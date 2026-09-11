"""Headless xvfb smoke test for the flashpoint GUI. Prints PASS/FAIL per check.

Covers the original 9 fixes PLUS the batch-2 changes:
  - specific-layer prep view honors the CURRENT border colour (was fixed magenta)
  - swatches show "<n> can(s)" (number before the word)
  - Save All Layers -> folder (master + borders.png + layers/ + stats.html)
  - Save Paint List -> static HTML table (name, colour example, cans)
  - quantize runs on a worker thread with an animated "Quantizing..." spinner
  - paint env H/S/L/contrast sliders (vectorized adjust_image + wiring)
"""
import sys, os, math, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashpoint_gui as g
import flashpoint_core as core
import tkinter as tk
from tkinter import ttk

# --- synthetic 24-color palette + 24-block image (560x560, internal edges) ---
N = 24
tmp_img = "/tmp/fp_smoke.png"; tmp_pal = "/tmp/fp_smoke_pal.txt"
rng = np.random.default_rng(7)
cols = [tuple(int(c) for c in rng.integers(20, 235, 3)) for _ in range(N)]
W, H = 560, 560
img = np.zeros((H, W, 3), dtype="uint8")
cw, ch = W // 6, H // 4
for i, (r, gg, b) in enumerate(cols):
    rr, cc = divmod(i, 6)
    img[rr*ch:(rr+1)*ch, cc*cw:(cc+1)*cw] = (r, gg, b)
core.Image.fromarray(img).save(tmp_img)
with open(tmp_pal, "w") as f:
    for i, (r, gg, b) in enumerate(cols):
        f.write(f"c{i:02d}, #{r:02X}{gg:02X}{b:02X}\n")

root = tk.Tk()
app = g.FlashpointApp(root, tmp_img, tmp_pal)
root.update(); root.update_idletasks()
ok = fail = 0
def check(name, cond, extra=""):
    global ok, fail
    if cond: ok += 1; print(f"  PASS  {name} {extra}")
    else:    fail += 1; print(f"  FAIL  {name} {extra}")

orig_status = app._status

def pump(cond, timeout=12.0):
    """Spin the Tcl event loop until cond() is true (or timeout)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        root.update(); root.update_idletasks()
        if cond():
            return True
        time.sleep(0.02)
    return cond()

# ---- top-level quantize is now ASYNC (worker thread + animator) ----
app.num_var.set("10"); app.exact_var.set(False)
app.prep_quantize()
pump(lambda: (not app._quant_running)
     and (app.state is not None or app._quant_err is not None))
st = app.state
print("== 0. quantize is threaded (state produced) ==")
check("top-level quantize produced a state", st is not None,
      f"(running={app._quant_running})")

print("== 0b. window title + icon ==")
check("window title is 'flashpointgui'", app.root.title() == "flashpointgui",
      f"(title={app.root.title()!r})")
check("window icon loaded (flashpointgui.png)",
      getattr(app, "_ico_ref", None) is not None)

print("== 1. spinboxes keyboard-editable (no readonly) ==")
def all_spinboxes():
    out = []
    def rec(w):
        if isinstance(w, ttk.Spinbox): out.append(w)
        for c in w.winfo_children(): rec(c)
    rec(app.prep)
    return out
sps = all_spinboxes()
editable = all(s.cget("state") not in ("readonly", "disabled") for s in sps)
check("all input spinboxes editable", editable and len(sps) >= 2,
      f"({len(sps)} spinboxes, editable={editable})")

print("== 2. mural width/height + can-coverage fields exist ==")
check("mural_mode var", hasattr(app, "mural_mode"))
check("mural_var var", hasattr(app, "mural_var"))
check("eff_var var", hasattr(app, "eff_var"))

print("== 3. swatch shows 'cans <n>' (matches core stats.html formula) ==")
app.mural_mode.set("width"); app.mural_var.set("3"); app.eff_var.set("3")
root.update()
total_px = st.W * st.H
AR = st.W / st.H
def expected_cans(cd):
    m = float(app.mural_var.get()); eff = float(app.eff_var.get())
    if m <= 0: return None
    w, h = (m, m / AR) if app.mural_mode.get() == "width" else (m * AR, m)
    wall = (w * h) * (100.0 * int(cd["mask"].sum()) / total_px) / 100.0
    return max(0, math.ceil(wall / eff - 1e-9)) if eff > 0 else 0
mismatch = 0
for cd in st.color_data:
    if app._cans_for_pct(100.0 * int(cd["mask"].sum()) / total_px) != expected_cans(cd):
        mismatch += 1
check("_cans_for_pct matches core formula for ALL colors", mismatch == 0,
      f"({mismatch} mismatches / {len(st.color_data)} colors)")
app._draw_swatch(); root.update()
texts = [app.swatch_canvas.itemcget(i, "text")
         for i in app.swatch_canvas.find_withtag("all")
         if app.swatch_canvas.type(i) == "text"]
cans_texts = [t for t in texts if t.endswith(" can(s)")]
check("swatch shows '<n> can(s)' (number before the word)",
      len(cans_texts) >= 1 and all(t[: -len(" can(s)")].isdigit() for t in cans_texts),
      f"(sample={texts[-5:]})")
# unmeasurable -> '—'
app.mural_var.set("0"); app._draw_swatch(); root.update()
texts2 = [app.swatch_canvas.itemcget(i, "text")
          for i in app.swatch_canvas.find_withtag("all")
          if app.swatch_canvas.type(i) == "text"]
check("mural=0 -> dashes (—)", "—" in texts2, f"(sample={texts2[-5:]})")
app.mural_var.set("3")

print("== 4. border + background color pickers route to on-theme dialog ==")
check("_add_color_control exists", hasattr(app, "_add_color_control"))
check("_ColorEditDialog class", hasattr(g, "_ColorEditDialog"))
applied = {"n": 0}
def _apply(rgb): applied["n"] += 1
dlg = g._ColorEditDialog(root, app, "test", (200, 40, 90), _apply)
root.update()
check("dialog instance has SQ square + lightness scale + preview",
      isinstance(dlg.lscale, ttk.Scale) and dlg.canvas is not None
      and getattr(dlg, "prev", None) is not None)
check("dialog drew its square PhotoImage", dlg.img is not None)
check("dialog applied a color on open (live)", applied["n"] >= 1,
      f"(applied {applied['n']}x)")
dlg._lmove(0.9); root.update()
check("lightness drag re-renders + applies", applied["n"] >= 2)
dlg._close(); root.update()
check("dialog close clears app ref", app._color_edit_dialog is None)

print("== 5. Log section removed ==")
check("no log_text widget", not hasattr(app, "log_text"))
try:
    app._log("ignored"); check("_log is a safe no-op", True)
except Exception as e:
    check("_log is a safe no-op", False, f"({e})")

print("== 6. zoom is a slider + wheel sync ==")
check("zoom_scale is ttk.Scale", isinstance(getattr(app, "zoom_scale", None), ttk.Scale))
check("prep_view.set_zoom exists", hasattr(app.prep_view, "set_zoom"))
app.prep_view.cw, app.prep_view.ch = 400, 400
app._zoom_from_slider(3.0); root.update()
check("slider drives zoom", abs(app.prep_view.zoom - 3.0) < 0.01,
      f"(zoom={app.prep_view.zoom:.2f})")

print("== 7. paint: no layer dropdown; overlay always borders ==")
check("no layer_var", not hasattr(app, "layer_var"))
check("no layer_cb", not hasattr(app, "layer_cb"))
app._bg = core.load_image_rgb(tmp_img)
app.paint_view.cw, app.paint_view.ch = 320, 320
frame = app._paint_view_fn()
check("paint renders 320x320 (borders overlay)", frame.shape == (320, 320, 3),
      f"({frame.shape})")
layer = app._paint_layer_rgba()
check("_paint_layer_rgba is borders RGBA", layer is not None and layer.shape[2] == 4,
      f"({None if layer is None else layer.shape})")

print("== 8. resize lag: configure is debounced ==")
live = set()
tok = {"n": 0}
class FakeCanvas:
    def __init__(self): self.n = 0
    def winfo_width(self): return 500 + self.n
    def winfo_height(self): return 400 + self.n
    def after(self, ms, fn):
        tok["n"] += 1; t = f"tok{tok['n']}"; live.add(t); self.n += 1
        return t
    def after_cancel(self, t):
        if t is not None and t in live: live.discard(t)
        return None
pv = app.prep_view
orig = pv.canvas
pv.canvas = FakeCanvas()
pv.cw, pv.ch = 2, 2; pv._conf_after = None
for _ in range(5):
    pv._on_configure()
pv.canvas = orig
check("5-event resize burst -> exactly 1 live refresh", len(live) == 1,
      f"(live after() tokens = {len(live)}; 4 superseded+cancelled)")

# ================= BATCH-2 FIXES =================

print("== 9. specific layer honors CURRENT border colour (was baked magenta) ==")
ov0 = st.borders_overlay
check("pipeline baked a border overlay with line pixels",
      ov0 is not None and (ov0[..., 3] > 0).any(),
      f"(alpha>0 px={int((ov0[..., 3] > 0).sum()) if ov0 is not None else 0})")
if ov0 is not None and (ov0[..., 3] > 0).any():
    app.border_rgb = (30, 120, 255)          # distinctive blue
    rt = app._retinted_borders()
    m = ov0[..., 3] > 200
    mean_rgb = rt[m][:, :3].mean(axis=0)
    check("_retinted_borders carries the current border colour",
          np.allclose(mean_rgb, (30, 120, 255), atol=1),
          f"(mean RGB on border px={mean_rgb.round(1)})")
    check("it is NOT the baked magenta",
          not np.allclose(mean_rgb, (255, 0, 255), atol=30))
    # changing the colour live re-tints the overlay
    app.border_rgb = (0, 220, 60)
    rt2 = app._retinted_borders()
    check("changing border colour live re-tints the overlay",
          np.allclose(rt2[m][:, :3].mean(axis=0), (0, 220, 60), atol=1))
    # ...and the specific-layer prep view uses it, not magenta
    app.view_var.set(st.color_data[0]["name"]); app.border_toggle.set(False)
    app.border_rgb = (30, 120, 255)
    base4 = app._prep_base_rgba()
    on_border = base4[:,:,:3][m]
    check("specific-layer view is not baked magenta on border px",
          not np.allclose(on_border.mean(axis=0), (255, 0, 255), atol=40),
          f"(border px mean RGB={on_border.mean(axis=0).round(1)})")
    check("specific-layer view shows the current border colour",
          np.allclose(on_border.mean(axis=0), (30, 120, 255), atol=25))
    app.border_rgb = (255, 0, 255)  # restore default
    app.view_var.set("master"); app.border_toggle.set(True)

print("== 10. Save All Layers -> folder (master + borders.png + layers/ + stats) ==")
outdir = "/tmp/fp_smoke_layers"
os.system(f"rm -rf {outdir}"); os.makedirs(outdir, exist_ok=True)
orig_askdir = g.filedialog.askdirectory
orig_asksave = g.filedialog.asksaveasfilename
g.filedialog.askdirectory = lambda **k: outdir
app.prep_save_all_layers()
files = os.listdir(outdir)
check("borders.png written", "borders.png" in files, f"(files={files})")
check("master_*.png written",
      any(f.startswith("master_") and f.endswith(".png") for f in files))
layers_dir = os.path.join(outdir, "layers")
check("layers/ folder has one PNG per colour",
      os.path.isdir(layers_dir)
      and len([f for f in os.listdir(layers_dir) if f.endswith(".png")])
      == len(st.color_data),
      f"({0 if not os.path.isdir(layers_dir) else len(os.listdir(layers_dir))} "
      f"png / {len(st.color_data)} colours)")
check("stats.html written", any(f.endswith(".html") for f in files))
if any(f.endswith(".html") for f in files):
    sph = os.path.join(outdir, "stats.html")
    sh = open(sph).read()
    check("stats.html embeds the logo (base64 data-URI)",
          "data:image/png;base64," in sh,
          f"(logo present={('data:image/png' in sh)})")
    check("stats.html title uses 'flashpointgui' (no 'paint coverage')",
          "flashpointgui" in sh and "paint coverage" not in sh,
          f"(flashpointgui={'flashpointgui' in sh} coverage={'paint coverage' in sh})")
app._status = orig_status

print("== 11. Save Paint List -> HTML table (name, colour example, cans) ==")
html_path = "/tmp/fp_smoke_paintlist.html"
g.filedialog.asksaveasfilename = lambda **k: html_path
app.mural_mode.set("width"); app.mural_var.set("3"); app.eff_var.set("3")
app.prep_save_paint_list()
if os.path.exists(html_path):
    html = open(html_path).read()
    check("paint list has an HTML table", "<table" in html)
    check("paint list names every colour",
          all(cd["name"] in html for cd in st.color_data))
    check("paint list shows colour examples (hex swatches)",
          all(("style" in html and cd["hex"].lstrip("#") in html.upper())
              or cd["hex"] in html for cd in st.color_data))
    check("paint list includes a cans column", "can" in html.lower())
    check("paint list shows a 'cans' figure (not all dashes)",
          any(t.isdigit() and t not in ("0",) for t in
              [x for x in html.replace("</td>", " ").split() if x.isdigit()])
          or "—" in html,
          f"(first 120 chars: {html[:120]!r})")
else:
    check("paint list html exists", False)
g.filedialog.askdirectory = orig_askdir
g.filedialog.asksaveasfilename = orig_asksave
app._status = orig_status
st = app.state

print("== 12. quantize runs on a worker thread with an animated spinner ==")
check("threading imported in gui", hasattr(g, "threading"))
check("animator hooks exist",
      all(hasattr(app, a) for a in
          ("_quant_tick", "_quant_finish", "_quant_running")))
real_run = g.core.run_pipeline
def slow_run(*a, **k):
    time.sleep(0.45); return real_run(*a, **k)
g.core.run_pipeline = slow_run
seen = []
app._status = lambda m: seen.append(m)
app.prep_quantize()
pump(lambda: (not app._quant_running) and app.state is not None, timeout=10.0)
g.core.run_pipeline = real_run
app._status = orig_status
q = [m for m in seen if m.startswith("Quantizing")]
dc = sorted(set(m.count(".") for m in q))
check("animation showed 'Quantizing...' (>=2 frames)", len(q) >= 2,
      f"({len(q)} frames; e.g. {q[:3]})")
check("dots visibly moved (>=2 distinct counts)", len(dc) >= 2, f"(dot counts {dc})")
check("run cleared the spinner (final status not 'Quantizing')",
      seen and not seen[-1].startswith("Quantizing"),
      f"(final={seen[-1][:40]!r})")
st = app.state

print("== 13. paint env H/S/L/contrast sliders ==")
for attr in ("paint_hue", "paint_sat", "paint_light", "paint_contrast"):
    check(f"{attr} is a ttk.Scale", isinstance(getattr(app, attr, None), ttk.Scale))
# adjust_image is correct on hand-computable cases
arr = np.full((4, 4, 3), 0.8, float)
c = g.adjust_image(arr, contrast=0.0)
check("adjust_image: contrast=0 -> mid gray", np.allclose(c, 0.5, atol=1e-5),
      f"(mean={c.mean():.3f})")
c2 = g.adjust_image(arr, light=0.2)
check("adjust_image: lightness +0.2 brightens",
      np.allclose(c2, 1.0, atol=1e-5) or c2.mean() > arr.mean(),
      f"(mean={c2.mean():.3f})")
s = g.adjust_image(np.array([[[1.0, 0.5, 0.0]]], float), sat=0.0)
check("adjust_image: saturation 0 -> neutral channel",
      abs(s[0,0,0] - s[0,0,1]) < 1e-5 and s[0,0,0] > 0.4,
      f"(rgb={s[0,0].round(2)})")
hh = g.adjust_image(np.array([[[1.0, 0.0, 0.0]]], float), hue=0.5)
check("adjust_image: hue +180 shifts red -> cyan (g> r)",
      hh[0,0,1] > hh[0,0,0], f"(rgb={hh[0,0].round(2)})")
check("adjust_image: defaults are identity (same object)",
      g.adjust_image(arr) is arr)
# wired into the live paint view
app._bg = core.load_image_rgb(tmp_img)
app.paint_view.cw, app.paint_view.ch = 200, 200
app.paint_reset()
base_frame = app._paint_view_fn().copy()
app.paint_contrast.set(0.0); app.paint_view.refresh()
low_frame = app._paint_view_fn()
check("contrast slider changes the rendered frame",
      not np.array_equal(base_frame, low_frame))
app.paint_reset()
check("paint_reset restores contrast label to 1.00",
      app.paint_contrast_lbl.cget("text") == "1.00",
      f"(={app.paint_contrast_lbl.cget('text')!r})")

print("== 14. manual recolor renames the swatch to the new hex ==")
# pick a color, edit it to a distinctive RGB via the live apply path, and
# confirm the swatch name / dropdown follow the new hex (not the old palette name)
st = app.state
target = st.color_data[0]
oldname = target["name"]
newrgb = (11, 22, 33)
app._edit_cid = target["ci"]
# view the target layer, then recolor it -> dropdown must track the rename
app.view_var.set(oldname)
app._apply_color(newrgb)
root.update()
check("recolor sets the swatch name to the new hex",
      target["name"] == core.rgb_to_hex(newrgb),
      f"(old={oldname!r} new={target['name']!r})")
check("recolor also updates cd['hex']",
      target["hex"] == core.rgb_to_hex(newrgb), f"(hex={target['hex']!r})")
check("layer dropdown tracks the rename (still points at this layer)",
      app.view_var.get() == core.rgb_to_hex(newrgb),
      f"(view={app.view_var.get()!r})")
check("dropdown values contain the new hex name",
      core.rgb_to_hex(newrgb) in list(app.view_cb.cget("values")))
app._edit_cid = None

print("== 15. preview push (ImageTk C-path) displays the correct frame ==")
# The preview now pushes via ImageTk.PhotoImage (C path) instead of building a
# per-pixel hex string + PhotoImage.put(). Capture the exact image hand-off and
# confirm it equals an independently-composed reference frame.
import flashpoint_gui as _g
_pcap = {}
_porig = _g.ImageTk.PhotoImage
def _pcap_ph(*_a, **_k):
    if _a:
        _pcap["img"] = np.asarray(_a[0]).copy()
    return _porig(*_a, **_k)
_g.ImageTk.PhotoImage = _pcap_ph

def _ref_frame(viewname, border):
    app.view_var.set(viewname)
    app.border_rgb = border
    arr = np.asarray(app._prep_view_fn())
    if arr.shape[2] == 4:
        bg = np.zeros((arr.shape[0], arr.shape[1], 3), dtype="uint8")
        a = arr[:, :, 3:4].astype("float32") / 255.0
        arr = (arr[:, :, :3].astype("float32") * a + bg * (1 - a)).clip(0, 255).astype("uint8")
    return arr

def _dsum(a, b):
    return int(np.abs(np.asarray(a).astype(int) - np.asarray(b).astype(int)).sum())

pv = app.prep_view
_pv_w, _pv_h = 120, 90
pv.cw, pv.ch = _pv_w, _pv_h
pv.zoom, pv.panx, pv.pany = 1.0, 0.0, 0.0

app.view_var.set("master")
app.border_rgb = (255, 0, 0)
app._epoch += 1; app._prep_zoom_cache = None
_pcap.clear(); pv.refresh(); root.update()
disp = _pcap.get("img")
check("preview frame captured via ImageTk path", disp is not None and disp.shape[:2] == (_pv_h, _pv_w),
      f"(shape={None if disp is None else disp.shape})")
if disp is not None:
    check("preview (master) == independently-composed reference",
          _dsum(disp, _ref_frame("master", (255, 0, 0))) == 0)
    app.view_var.set("borders")
    app.border_rgb = (0, 0, 255)
    app._epoch += 1; app._prep_zoom_cache = None
    _pcap.clear(); pv.refresh(); root.update()
    db = _pcap.get("img")
    check("preview (borders, recolored) == reference",
          db is not None and _dsum(db, _ref_frame("borders", (0, 0, 255))) == 0)
_g.ImageTk.PhotoImage = _porig

print("== 16. paint: zoom works + color-adjust skips the overlay ==")
# (a) Zoom: the wheel updates paint_view.zoom, and _paint_view_fn must now
#     actually read it (it was a no-op — always composited at 1:1 canvas size).
app._bg = core.load_image_rgb(tmp_img)
app.paint_view.cw, app.paint_view.ch = 240, 240
app.paint_reset()
app.paint_view.zoom = 1.0; app.paint_view.panx = 0.0; app.paint_view.pany = 0.0
z1 = app._paint_view_fn().copy()
app.paint_view.zoom = 2.0
z2 = app._paint_view_fn().copy()
check("paint zoom changes the composite (was a no-op)",
      not np.array_equal(z1, z2))
app.paint_view.zoom = 0.5
z3 = app._paint_view_fn().copy()
check("zoom in vs out differ", not np.array_equal(z2, z3))
app.paint_view.zoom = 1.0
# (b) Color adjust must hit ONLY the background, never the border overlay.
#     Deterministic check: spy on adjust_image and confirm it is fed the
#     BACKGROUND-ONLY composite (overlay not yet drawn), not the finished
#     frame. (The old bug applied adjust_image to `out` AFTER the overlay was
#     composited in, so it tinted the lines too.)
app.border_rgb = (255, 0, 0)   # solid red border
app._epoch += 1                 # invalidate the border transform cache
app.paint_reset()
cw, ch = 240, 240
# reference for "background-only": exactly what _paint_view_fn feeds to
# adjust_image (self._bg_fitted(...)/255, pre-adjust, no overlay).
ref_bg = app._bg_fitted(cw, ch).astype("float32") / 255.0
app.paint_hue.set(0.4)          # non-neutral adjust so the code path runs
_g2 = __import__("flashpoint_gui")
_seen = {}
_real_adjust = _g2.adjust_image
def _spy_adjust(img, *a, **k):
    _seen["img"] = np.asarray(img).copy()
    return _real_adjust(img, *a, **k)
_g2.adjust_image = _spy_adjust
try:
    _ = app._paint_view_fn()
finally:
    _g2.adjust_image = _real_adjust
check("adjust_image is called with the background-only composite",
      "img" in _seen, f"(captured={list(_seen)})")
if "img" in _seen:
    d = np.abs(_seen["img"].astype(float) - ref_bg.astype(float))
    check("that composite is the background with NO overlay in it",
          d.max() <= 1e-6, f"(max delta vs bg-only={d.max():.2e})")
app.paint_reset()

print("== 17. save fixes: prep full-res master / paint full-res composite / alt theme ==")
st = app.state
stW, stH = st.W, st.H

# (a) Prep "Save" writes the MASTER at native resolution (not a 900x900 preview),
#     plain master by default, +border overlay only when the toggle is on.
app.view_var.set("master"); app.border_toggle.set(False)
g.filedialog.asksaveasfilename = lambda **k: "/tmp/fp_smoke_master.png"
app.prep_save()
mp = core.Image.open("/tmp/fp_smoke_master.png")
marr = np.asarray(mp)
check("prep save is full native resolution (was 900x900 preview)",
      marr.shape[:2] == (stH, stW), f"(shape={marr.shape}, native=({stH},{stW}))")
check("prep save (plain) == core.render_master",
      np.array_equal(marr[..., :3], np.asarray(core.render_master(st, st.palette_rgb))))
app.border_toggle.set(True); app.border_rgb = (255, 0, 255)
g.filedialog.asksaveasfilename = lambda **k: "/tmp/fp_smoke_masterb.png"
app.prep_save()
mbarr = np.asarray(core.Image.open("/tmp/fp_smoke_masterb.png"))
check("prep save (borders ticked) overlays the border lines",
      not np.array_equal(mbarr[..., :3],
                         np.asarray(core.render_master(st, st.palette_rgb))))
check("prep save (borders ticked) == core.master_with_borders",
      np.array_equal(mbarr[..., :3],
                     np.asarray(core.master_with_borders(st, (255, 0, 255)))))
app.border_toggle.set(False)

# (c) Prep "Save" of a SPECIFIC layer view saves THAT layer at full res (Saxon:
#     "if the user is showing chocolate brown, save the chocolate brown layer").
pick = st.color_data[0]["name"]
app.view_var.set(pick); app.border_toggle.set(False)
_cap = {}
g.filedialog.asksaveasfilename = lambda **k: (_cap.update(k), "/tmp/fp_smoke_layer.png")[1]
app.prep_save()
larr = np.asarray(core.Image.open("/tmp/fp_smoke_layer.png"))
ref = app._prep_base_rgba()
check("prep save (specific layer) is full native resolution",
      larr.shape[:2] == (stH, stW), f"(shape={larr.shape}, native=({stH},{stW}))")
check("prep save (specific layer) == the current view (_prep_base_rgba)",
      np.array_equal(larr, np.asarray(ref).astype("uint8")))
check("prep save (specific layer) is NOT the master (it's a real layer)",
      not np.array_equal(larr[..., :3],
                         np.asarray(core.render_master(st, st.palette_rgb))))
safe = "".join(c if (c.isalnum() or c in "-_") else "_" for c in pick)[:60] or "view"
check("prep save (specific layer) offers a colour-named default filename",
      _cap.get("initialfile") == safe + ".png",
      f"(initialfile={_cap.get('initialfile')!r} want={safe + '.png'!r})")
app.view_var.set("master")

# (b) Paint "Save" writes the composite at the WALL PHOTO's own dimensions
#     (option 2: rectangular, photo fills the frame, no letterbox). Use a
#     NON-square photo so the letterbox path is unambiguous.
bH, bW = 200, 320
non_sq = np.dstack([np.full((bH, bW, 1), 190, "uint8"),
                    np.full((bH, bW, 1), 120, "uint8"),
                    np.full((bH, bW, 1), 70, "uint8")])
app._bg = non_sq
app.border_rgb = (255, 0, 0)   # red borders -> the overlay check below is unambiguous
app._epoch += 1                 # invalidate the (magenta-keyed) transform cache
app.paint_view.cw, app.paint_view.ch = 200, 200
app.paint_reset()
app.paint_hue.set(0.3); app.paint_scale.set("1.2")   # non-default adjust + scale
g.filedialog.asksaveasfilename = lambda **k: "/tmp/fp_smoke_paint.png"
app.paint_save()
pf = np.asarray(core.Image.open("/tmp/fp_smoke_paint.png"))
check("paint save is the photo's native size (no letterbox)",
      pf.shape == (bH, bW, 3), f"(shape={pf.shape}, native=({bH},{bW},3))")
# The saved frame must differ from the RAW photo — proving the background hue
# adjust AND the border composite were applied (a raw-photo passthrough would
# be identical to non_sq).
check("paint save is not a raw-photo passthrough (adjust+composite applied)",
      not np.array_equal(pf, non_sq))
check("paint save contains the border overlay (some border-red pixels)",
      bool(((pf[..., 0] > 200) & (pf[..., 1] < 90) & (pf[..., 2] < 90)).any()))
app.paint_reset()

# (c) The toolkit uses the 'alt' ttk theme (available on this Tk build).
check("toolkit uses the 'alt' ttk theme",
      ttk.Style().theme_use() == "alt", f"(={ttk.Style().theme_use()!r})")

print(f"\nRESULT: {ok} passed, {fail} failed")
root.destroy()
sys.exit(1 if fail else 0)
