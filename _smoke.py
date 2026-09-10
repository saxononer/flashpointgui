"""Headless xvfb smoke test for the 9 GUI fixes. Prints PASS/FAIL per check."""
import sys, os, math, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import flashpoint_gui as g
import flashpoint_core as core
import tkinter as tk
from tkinter import ttk

# --- synthetic 24-color palette + 24-block image ---
N = 24
tmp_img = "/tmp/fp_smoke.png"; tmp_pal = "/tmp/fp_smoke_pal.txt"
rng = np.random.default_rng(7)
cols = [tuple(int(c) for c in rng.integers(20, 235, 3)) for _ in range(N)]
W, H = 240, 240
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

app.num_var.set("10"); app.exact_var.set(False)
app.prep_quantize(); root.update()
st = app.state

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

print("== 3. swatch shows cans (matches core stats.html formula) ==")
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
# cans text present: numeric strings; '—' when unmeasurable
nums = [t for t in texts if t.isdigit()]
check("swatch renders numeric cans", len(nums) >= 1, f"(sample={texts[-4:]})")
# unmeasurable -> '—'
app.mural_var.set("0"); app._draw_swatch(); root.update()
texts2 = [app.swatch_canvas.itemcget(i, "text")
          for i in app.swatch_canvas.find_withtag("all")
          if app.swatch_canvas.type(i) == "text"]
check("mural=0 -> dashes (—)", "—" in texts2, f"(sample={texts2[-4:]})")
app.mural_var.set("3")

print("== 4. border + background color pickers route to on-theme dialog ==")
check("_add_color_control exists", hasattr(app, "_add_color_control"))
check("_ColorEditDialog class", hasattr(g, "_ColorEditDialog"))
check("vectorized HSL helper", hasattr(g, "_hsl_to_rgb_vec"))
# Instantiate the dialog and exercise the vectorized square render path.
applied = {"n": 0}
def _apply(rgb): applied["n"] += 1
dlg = g._ColorEditDialog(root, app, "test", (200, 40, 90), _apply)
root.update()
check("dialog instance has SQ square + lightness scale + preview",
      isinstance(dlg.lscale, ttk.Scale) and dlg.canvas is not None
      and getattr(dlg, "prev", None) is not None)
check("dialog drew its square PhotoImage (vectorized)", dlg.img is not None)
check("dialog applied a color on open (live)", applied["n"] >= 1,
      f"(applied {applied['n']}x)")
# drag the lightness slider -> re-render square + apply, no crash
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
check("preview has _conf_after attr", hasattr(app.prep_view, "_conf_after"))
# A resize DRAG fires a burst of Configure events with a *different* size each
# time. The live handler must cancel the previously-pending refresh and
# schedule exactly one new one, so at the end only ONE refresh is live.
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

print(f"\nRESULT: {ok} passed, {fail} failed")
root.destroy()
sys.exit(1 if fail else 0)
