"""Open the GUI under Xvfb, populate it, grab a screenshot."""
import sys, os, numpy as np
sys.path.insert(0, ".")
import flashpoint_gui as g
import flashpoint_core as core
from PIL import Image as PILImage, ImageGrab

# 24-color test image + palette (reuse _verify's generator)
N = 24
tmp_img = "/tmp/fp_shot.png"; tmp_pal = "/tmp/fp_shot_palette.txt"
rng = np.random.default_rng(7)
cols = [tuple(int(c) for c in rng.integers(20, 235, 3)) for _ in range(N)]
W, H = 240, 240
img = np.zeros((H, W, 3), dtype="uint8")
cw, ch = W // 6, H // 4
for i, (r, gc, b) in enumerate(cols):
    rr, cc = divmod(i, 6)
    img[rr*ch:(rr+1)*ch, cc*cw:(cc+1)*cw] = (r, gc, b)
PILImage.fromarray(img).save(tmp_img)
with open(tmp_pal, "w") as f:
    for i, (r, gc, b) in enumerate(cols):
        f.write(f"c{i:02d}, #{r:02X}{gc:02X}{b:02X}\n")

import tkinter as tk
root = tk.Tk()
app = g.FlashpointApp(root, tmp_img, tmp_pal)
app.num_var.set("12")
root.update(); root.update_idletasks()
app.prep_quantize()
root.update(); root.update_idletasks()
# force preview draw
app.view_var.set("master")
app.prep_view.refresh(); app.prep_view.canvas.update()

def grab():
    root.update_idletasks(); root.update()
    disp = os.environ.get("DISPLAY", ":99")
    im = ImageGrab.grab(xdisplay=disp)
    im.save("/tmp/flashpoint_gui.png")
    print("saved", im.size)
    root.destroy()

root.after(2000, grab)
root.mainloop()
