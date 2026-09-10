"""Headless Xvfb verification of all 6 changes. Prints PASS/FAIL per check."""
import time, sys, os, numpy as np
sys.path.insert(0, ".")
import flashpoint_gui as g
import flashpoint_core as core

# --- synthetic 24-color palette + 24-block image (so num_colors is testable) ---
N = 24
tmp_img = "/tmp/fp_test.png"; tmp_pal = "/tmp/fp_test_palette.txt"
rng = np.random.default_rng(7)
# 24 well-separated colors
cols = []
for i in range(N):
    r,g_,b = (int(c) for c in rng.integers(20, 235, 3))
    cols.append((r,g_,b))
# image: 4x6 grid of blocks, each a distinct palette color (all 24 present)
W,H = 240,240
img = np.zeros((H,W,3), dtype="uint8")
cw,ch = W//6, H//4
for i,(r,g_,b) in enumerate(cols):
    rr,cc = divmod(i,6)
    img[rr*ch:(rr+1)*ch, cc*cw:(cc+1)*cw] = (r,g_,b)
core.Image.fromarray(img).save(tmp_img)
with open(tmp_pal,"w") as f:
    for i,(r,g_,b) in enumerate(cols):
        f.write(f"c{i:02d}, #{r:02X}{g_:02X}{b:02X}\n")

import tkinter as tk
from tkinter import ttk
root = tk.Tk()
app = g.FlashpointApp(root, tmp_img, tmp_pal)
root.update(); root.update_idletasks()
ok=fail=0
def check(name, cond, extra=""):
    global ok, fail
    if cond: ok+=1; print(f"  PASS  {name} {extra}")
    else:    fail+=1; print(f"  FAIL  {name} {extra}")

print(f"palette={len(app.palette)} colors, image uses all {N}")

print("== 1. QUANTIZE BUG: num_colors honored (exact=False default) ==")
app.num_var.set("10"); app.exact_var.set(False)
app.prep_quantize(); root.update()
used = len(app.state.color_data)
check("num=10 -> used <= 10 (was: whole palette)", used <= 10, f"(used {used})")
app.exact_var.set(True); app.prep_quantize(); root.update()
check("exact=True -> uses full palette", len(app.state.color_data)==N,
      f"(used {len(app.state.color_data)} == {N})")
app.exact_var.set(False); app.num_var.set("10"); app.prep_quantize(); root.update()

print("== 2. LOG BOX under input, receives quantize output ==")
txt = app.log_text.get("1.0","end") if hasattr(app,"log_text") else ""
check("log_text exists", hasattr(app,"log_text"))
check("log has [quantize] line", "[quantize]" in txt)
check("log has per-color lines", txt.count("·")>0, f"({txt.count(chr(0xb7))} color lines)")

print("== 3. BORDER + BACKGROUND color settings ==")
check("defaults", tuple(app.border_rgb)==(255,0,255) and tuple(app.bg_rgb)==(128,128,128))
app._set_border_rgb((0,200,90)); app.view_var.set("borders")
app.prep_view.zoom=1.0; app.prep_view.refresh(); root.update()
b1 = app._prep_view_fn().astype("int")
app._set_border_rgb((255,0,255)); app.prep_view.refresh(); root.update()
b2 = app._prep_view_fn().astype("int")
check("border color changes borders render", not np.array_equal(b1,b2))
gpx = int(np.sum((b1[...,1]>150)&(b1[...,0]<80)&(b1[...,2]<120)&(b1[...,3]>0)))
check("green border pixels visible", gpx>0, f"({gpx}px)")

print("== 4. HSL pop-up (Photoshop-style) on swatch click ==")
ci = app.state.color_data[0]["ci"]
app._swatch_click(ci); root.update()
dlg = app._color_edit_dialog
check("dialog opened", dlg is not None)
check("has square canvas + lightness slider", dlg is not None and hasattr(dlg,"canvas") and hasattr(dlg,"lscale"))
before = tuple(app.state.color_data[0]["rgb"])
class E: pass
e=E(); e.x=int(dlg.SQ*0.75); e.y=int(dlg.SQ*0.25); dlg._sq(e); root.update()
dlg._lmove(0.9); root.update()
after = tuple(app.state.color_data[0]["rgb"])
check("edit changed color live", before!=after, f"({before}->{after})")
dlg._close(); root.update()
check("dialog closed + app ref cleared", app._color_edit_dialog is None)
check("square PhotoImage created", dlg.img is not None)

print("== 5. PAINT scale/rotate are numeric fields ==")
check("paint_scale is ttk.Spinbox", isinstance(app.paint_scale, ttk.Spinbox), type(app.paint_scale).__name__)
check("paint_rot is ttk.Spinbox", isinstance(app.paint_rot, ttk.Spinbox), type(app.paint_rot).__name__)
app.paint_scale.set("2.0"); root.update()
check("paint view renders", app._paint_view_fn().shape==(app.paint_view.ch,app.paint_view.cw,3))
app.paint_reset(); root.update()
check("reset scale label", app.paint_scale_lbl.cget("text")=="1.00×")

print("== 6. PERFORMANCE (lag) ==")
a=(np.random.rand(760,520,3)*255).astype("uint8")
t0=time.time(); _=g._hex_rows(a); tv=time.time()-t0
t0=time.time(); _=[["#%02x%02x%02x"%(a[i,j,0],a[i,j,1],a[i,j,2]) for j in range(520)] for i in range(760)]; tc=time.time()-t0
check("vectorized _hex_rows faster", tv<tc*0.9, f"(vec {tv*1000:.0f}ms vs {tc*1000:.0f}ms)")
app.view_var.set("master"); app.prep_view.zoom=1.0
app.prep_view.refresh(); root.update()
t0=time.time()
for _ in range(20):
    app.prep_view.panx+=5; app.prep_view.pany+=3; _=app._prep_view_fn()
tpan=time.time()-t0
check("prep pan 20 steps fast (cache hit)", tpan<0.15, f"({tpan*1000:.0f}ms)")
app._bg=core.load_image_rgb(tmp_img)
t0=time.time()
for _ in range(30):
    app.paint_cx+=4; app.paint_cy+=2; _=app._paint_view_fn()
tdrag=time.time()-t0
check("paint drag 30 steps fast (cache hit)", tdrag<0.4, f"({tdrag*1000:.0f}ms)")

print(f"\nRESULT: {ok} passed, {fail} failed")
root.destroy()
sys.exit(1 if fail else 0)
