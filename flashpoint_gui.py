"""
flashpoint_gui.py  —  tkinter GUI for the flashpoint posterizer.

Two workspaces (a Notebook):

  PREP    Build the poster.
          - inputs: image, palette, num colors, min area (paths shown inline)
          - [Quantize] runs the pipeline
          - a zoomable / pannable preview (zoom slider or wheel, drag = pan, reset)
          - a dropdown picks what to view: master / borders
          - a border-overlay toggle for judging complexity
          - [Save] writes the currently-viewed image to a file
          - a strip of the colors actually USED in the poster (can count each,
            derived from the mural size + can-coverage); click one -> an HSL
            editor nudges it and the preview updates live (the pipeline is
            cached, only the render re-runs).

  PAINT   Position the result over a wall photo.
          - [Upload background]
          - the overlay is always borders.png (the layer picker was removed)
          - move (drag the canvas) + scale + rotate sliders
          - opacity + blend-mode
          - [Save] the composited result

Rendering model: Tk 9.0 (x0n) has a flaky image decoder under a virtual
display, so we never use tk.PhotoImage(file=) or canvas.scale. Zoom / pan /
compositing are all done in numpy; the result is pushed into ONE persistent
PhotoImage via put() (raw pixels into an already-registered image = stable).

Run:  python flashpoint_gui.py [image.png] [palette.txt]
"""

import tkinter as tk
from tkinter import ttk, filedialog
from PIL import Image
import numpy as np
import os
import time
import math

import flashpoint_core as core


BLENDS = ["normal", "multiply", "screen", "overlay", "softlight",
          "darken", "lighten", "difference", "color_dodge", "color_burn",
          "hardlight", "luminosity"]


# ----------------------------------------------------------------------------
# HSL <-> RGB (standard HSL, not HLS)
# ----------------------------------------------------------------------------
def rgb_to_hsl(rgb):
    r, g, b = (v / 255.0 for v in rgb)
    mx, mn = max(r, g, b), min(r, g, b)
    l = (mx + mn) / 2.0
    if mx == mn:
        return (0.0, 0.0, l)
    d = mx - mn
    s = d / (2.0 - mx - mn) if l > 0.5 else d / (mx + mn)
    if mx == r:
        h = (g - b) / d + (6.0 if g < b else 0.0)
    elif mx == g:
        h = (b - r) / d + 2.0
    else:
        h = (r - g) / d + 4.0
    return (h / 6.0, s, l)


def hsl_to_rgb(h, s, l):
    h = h % 1.0
    if s == 0.0:
        v = round(l * 255)
        return (v, v, v)
    q = l * (1 + s) if l < 0.5 else l + s - l * s
    p = 2 * l - q

    def f(t):
        t = t % 1.0
        if t < 1/6:
            return p + (q - p) * 6 * t
        if t < 1/2:
            return q
        if t < 2/3:
            return p + (q - p) * (2/3 - t) * 6
        return p

    return (round(f(h + 1/3) * 255), round(f(h) * 255), round(f(h - 1/3) * 255))


# vectorized put()-push: precompute 256 two-hex-char strings, fancy-index them.
# ~6x faster than the per-pixel comprehension at realistic canvas sizes.
_HD = ['0','1','2','3','4','5','6','7','8','9','a','b','c','d','e','f']
_HEX2 = np.array([_HD[i] + _HD[j] for i in range(16) for j in range(16)],
                 dtype='U2')


def _hex_rows(arr):
    """(H,W,3) uint8 -> nested list of '#rrggbb' strings for tk PhotoImage.put.

    This Tk 9.0.3 build will not parse integer RGB lists in put() (it treats a
    bare number as a color name and raises 'invalid color name'), so we feed it
    hex strings. Vectorized: index a 256-entry hex table with the r/g/b planes
    instead of looping over pixels in Python.
    """
    a = np.ascontiguousarray(arr, dtype='uint8')
    r, g, b = a[..., 0].ravel(), a[..., 1].ravel(), a[..., 2].ravel()
    s = '#' + _HEX2[r] + _HEX2[g] + _HEX2[b]
    return s.reshape(a.shape[0], a.shape[1]).tolist()


def _resize(arr, w, h):
    return np.asarray(Image.fromarray(arr).resize((max(1, int(w)),
                                                   max(1, int(h))),
                                                   Image.BILINEAR))


def _hsl_to_rgb_vec(h, s, l):
    """Vectorized HSL->RGB for numpy arrays; returns (H,W,3) uint8.

    Replaces the per-pixel Python loop the old color square used (~3k scalar
    hsl_to_rgb calls per lightness change) — the square now computes in one
    numpy pass, so dragging lightness stays responsive.
    """
    h = np.mod(h, 1.0)
    c = (1.0 - np.abs(2.0 * l - 1.0)) * s
    hp = h * 6.0
    x = c * (1.0 - np.abs((hp % 2.0) - 1.0))
    m = l - c / 2.0
    sel = [hp < 1, hp < 2, hp < 3, hp < 4, hp < 5, hp < 6]
    r = np.select(sel, [c, x, 0, 0, x, c], default=c)
    g = np.select(sel, [0, c, c, x, 0, 0], default=0)
    b = np.select(sel, [0, 0, x, c, c, x], default=0)
    out = np.stack([r + m, g + m, b + m], axis=-1) * 255.0
    return np.clip(np.round(out), 0, 255).astype('uint8')


def _fit(img, cw, ch):
    """Center-fit an (h,w,c) image into a (ch,cw) box (bg 0). Returns (out,box)
    where box=(ox,oy,nw,nh)."""
    h, w = img.shape[:2]
    s = min(cw / max(1, w), ch / max(1, h))
    nw, nh = max(1, int(w * s)), max(1, int(h * s))
    z = _resize(img, nw, nh)
    out = np.zeros((ch, cw, z.shape[2]), dtype=z.dtype)
    ox, oy = (cw - nw) // 2, (ch - nh) // 2
    out[oy:oy + nh, ox:ox + nw] = z
    return out, (ox, oy, nw, nh)


class _Preview:
    """A zoomable/pannable canvas that pushes pixels into one persistent image.

    view_fn() -> ndarray of exactly (ch, cw, 3) or (ch, cw, 4) — the view at
    the CURRENT zoom/pan, already composited. Zoom + pan live here; view_fn
    reads them.
    """

    def __init__(self, canvas, view_fn, pan=True):
        self.canvas = canvas
        self.view_fn = view_fn
        self.cw = 2
        self.ch = 2
        self.zoom = 1.0
        self.panx = 0.0
        self.pany = 0.0
        self._img = None
        self._img_ref = None
        self._last = None
        self._conf_after = None
        canvas.bind("<Configure>", self._on_configure)
        canvas.bind("<Button-4>", lambda e: self._step(+1))
        canvas.bind("<Button-5>", lambda e: self._step(-1))
        canvas.bind("<MouseWheel>", lambda e: self._step(1 if e.delta > 0 else -1))
        if pan:
            canvas.bind("<ButtonPress-1>",
                        lambda e: setattr(self, "_last", (e.x, e.y)))
            canvas.bind("<B1-Motion>", self._drag)

    def _ensure_img(self):
        for _ in range(20):
            img = tk.PhotoImage(width=2, height=2)
            self._img_ref = img
            try:
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=img, anchor="nw")
                self._img = img
                return
            except tk.TclError:
                try:
                    self.canvas.delete("all")
                except Exception:
                    pass
                self.canvas.update()
                time.sleep(0.05)
        self._img = None
        self._img_ref = None

    def _on_configure(self, e=None):
        w, h = self.canvas.winfo_width(), self.canvas.winfo_height()
        if w > 2 and h > 2 and (w != self.cw or h != self.ch):
            self.cw, self.ch = w, h
            # Debounce: dragging a window border fires a burst of Configure
            # events; each would re-resize + re-put the whole frame (the resize
            # lag). Coalesce to a single refresh shortly after the last event.
            try:
                self.canvas.after_cancel(self._conf_after)
            except Exception:
                pass
            self._conf_after = self.canvas.after(40, self.refresh)

    def refresh(self):
        if self._img is None:
            self._ensure_img()
        if self._img is None:
            return
        try:
            arr = self.view_fn()
        except Exception:
            return
        arr = np.asarray(arr)
        if arr.shape[:2] != (self.ch, self.cw):
            return
        if arr.shape[2] == 4:
            bg = np.zeros((self.ch, self.cw, 3), dtype="uint8")
            a = (arr[:, :, 3:4].astype("float32") / 255.0)
            rgb = arr[:, :, :3].astype("float32")
            arr = (rgb * a + bg * (1 - a)).clip(0, 255).astype("uint8")
        self._img.config(width=self.cw, height=self.ch)
        self._img.put(_hex_rows(arr))

    # NOTE: this Tk 9.0.3 build rejects put() with integer RGB lists
    # ("invalid color name \"255\"") but accepts nested '#rrggbb' strings.

    def _step(self, direction):
        self._zoom_to(self.zoom * (1.2 ** direction))

    def _zoom_to(self, new):
        new = max(0.2, min(16.0, new))
        cx, cy = self.cw / 2, self.ch / 2
        old = self.zoom
        sx = (cx - self.panx) / old
        sy = (cy - self.pany) / old
        self.zoom = new
        self.panx = cx - sx * new
        self.pany = cy - sy * new
        self.refresh()

    def set_zoom(self, new):
        """Absolute zoom (slider) that keeps the canvas center fixed."""
        self._zoom_to(float(new))

    def _drag(self, e):
        if self._last is None:
            return
        self.panx += (e.x - self._last[0])
        self.pany += (e.y - self._last[1])
        self._last = (e.x, e.y)
        self.refresh()

    def reset(self):
        self.zoom = 1.0
        self.panx = 0.0
        self.pany = 0.0
        self.refresh()


class FlashpointApp:
    def __init__(self, root, image_path=None, palette_path=None):
        self.root = root
        root.title("flashpoint — posterizer")
        root.geometry("1200x860")
        root.minsize(920, 660)
        try:
            ttk.Style().theme_use("clam")
        except tk.TclError:
            pass

        self.image_rgb = None
        self.image_path = None
        self.palette = []
        self.palette_path = None
        self.state = None
        self._edit_cid = None
        self._bg = None
        self._bg_cache = (None, 0, 0, None)
        # content epoch: bumped whenever pixels change (quantize / recolor /
        # border+bg color). Zoom/pan and position do NOT change content.
        self._epoch = 0
        # render caches (the anti-lag core): the zoomed, centered base (prep)
        # and the scaled+rotated layer (paint) are built ONCE per content/zoom/
        # scale/rotate change, then pan/drag just cheaply copy/offset them.
        self._prep_zoom_cache = None
        self._paint_transform_cache = None
        # user-configurable line + background colors (default to the stable
        # core defaults: magenta lines, mid-gray bg)
        self.border_rgb = (255, 0, 255)
        self.bg_rgb = (128, 128, 128)

        self.nb = ttk.Notebook(root)
        self.nb.pack(fill="both", expand=True, padx=8, pady=8)
        self.prep = ttk.Frame(self.nb)
        self.paint = ttk.Frame(self.nb)
        self.nb.add(self.prep, text="  Prep  ")
        self.nb.add(self.paint, text="  Paint  ")
        self._build_prep()
        self._build_paint()

        if image_path:
            self._load_image(image_path)
        if palette_path:
            self._load_palette(palette_path)

    def _status(self, text):
        self.status.config(text=text)

    def _log(self, text):
        """No-op: the Log box was removed. Kept so old call sites stay valid."""
        return

    # ---- color pickers (border / background) ----
    def _add_color_control(self, parent, name, get_rgb, set_rgb):
        """A clickable swatch + hex label that opens the on-theme HSL color
        editor (same dialog as the poster-color strip) and applies changes
        live via set_rgb. get_rgb supplies the CURRENT color when reopening.

        The old path used tk.colorchooser.askcolor, which is flaky on this Tk
        9.0 / Wayland build (often opens nothing) and is off-theme anyway; the
        shared in-app editor sidesteps both.
        """
        hexc = core.rgb_to_hex(get_rgb())
        box = ttk.Frame(parent); box.pack(side="left", padx=8)
        sw = tk.Canvas(box, width=28, height=20, highlightthickness=0, bg="#1a1a1a")
        sw.pack(side="left")
        rect = sw.create_rectangle(0, 0, 28, 20, fill=hexc, outline="#555", width=1)
        lbl = ttk.Label(box, text=hexc, width=9)
        lbl.pack(side="left")

        def open_picker(_=None):
            if self._color_edit_dialog is not None:
                try:
                    if self._color_edit_dialog.winfo_exists():
                        self._color_edit_dialog.destroy()
                except Exception:
                    pass
                self._color_edit_dialog = None

            def apply(rgb):
                h = core.rgb_to_hex(rgb)
                sw.itemconfig(rect, fill=h)
                lbl.config(text=h)
                set_rgb(rgb)

            self._color_edit_dialog = _ColorEditDialog(
                self.root, self, name, tuple(get_rgb()), apply)

        sw.bind("<Button-1>", open_picker)
        lbl.bind("<Button-1>", open_picker)

    def _set_border_rgb(self, rgb):
        self.border_rgb = rgb
        self._epoch += 1
        self._prep_zoom_cache = None
        self._paint_transform_cache = None
        if self.state is not None:
            self.prep_view.refresh()
            self.paint_view.refresh()

    def _set_bg_rgb(self, rgb):
        self.bg_rgb = rgb
        self._epoch += 1
        self._prep_zoom_cache = None
        self._paint_transform_cache = None
        if self.state is not None:
            self.prep_view.refresh()
            self.paint_view.refresh()

    # ================================================================== PREP
    def _build_prep(self):
        p = self.prep

        inp = ttk.LabelFrame(p, text="Input")
        inp.pack(fill="x", padx=6, pady=(6, 2))
        r1 = ttk.Frame(inp); r1.pack(fill="x", padx=6, pady=(4, 2))
        ttk.Button(r1, text="Load image…",
                   command=self.prep_pick_image).pack(side="left")
        self.img_lbl = ttk.Label(r1, text="no image", foreground="#888")
        self.img_lbl.pack(side="left", padx=8)
        ttk.Button(r1, text="Load palette…",
                   command=self.prep_pick_palette).pack(side="left", padx=(16, 0))
        self.pal_lbl = ttk.Label(r1, text="no palette", foreground="#888")
        self.pal_lbl.pack(side="left", padx=8)

        r2 = ttk.Frame(inp); r2.pack(fill="x", padx=6, pady=(2, 2))
        ttk.Label(r2, text="Num colors:").pack(side="left")
        self.num_var = tk.StringVar(value="10")
        ttk.Spinbox(r2, textvariable=self.num_var, from_=2, to=64, width=5,
                    increment=1).pack(side="left", padx=(4, 8))
        self.exact_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(r2, text="Full palette (ignore num colors)",
                        variable=self.exact_var).pack(side="left", padx=(0, 16))
        ttk.Label(r2, text="Min area:").pack(side="left")
        self.min_var = tk.StringVar(value="50")
        ttk.Spinbox(r2, textvariable=self.min_var, from_=0, to=100000, width=7,
                    increment=1).pack(side="left", padx=(4, 16))
        ttk.Button(r2, text="Quantize", command=self.prep_quantize,
                   style="Accent.TButton").pack(side="left", padx=(16, 0))

        r2b = ttk.Frame(inp); r2b.pack(fill="x", padx=6, pady=(2, 2))
        ttk.Label(r2b, text="Mural:").pack(side="left")
        self.mural_mode = tk.StringVar(value="width")
        ttk.Radiobutton(r2b, text="width", variable=self.mural_mode, value="width",
                        command=self._swatch_changed).pack(side="left")
        ttk.Radiobutton(r2b, text="height", variable=self.mural_mode, value="height",
                        command=self._swatch_changed).pack(side="left", padx=(0, 8))
        ttk.Label(r2b, text="(m):").pack(side="left")
        self.mural_var = tk.StringVar(value="3")
        mural_sp = ttk.Spinbox(r2b, textvariable=self.mural_var, from_=0, to=10000,
                               increment=0.1, width=6)
        mural_sp.pack(side="left", padx=(4, 12))
        mural_sp.bind("<Return>", self._swatch_changed)
        mural_sp.bind("<FocusOut>", self._swatch_changed)
        ttk.Label(r2b, text="Can coverage (m²/can):").pack(side="left")
        self.eff_var = tk.StringVar(value="3")
        eff_sp = ttk.Spinbox(r2b, textvariable=self.eff_var, from_=0, to=10000,
                             increment=0.1, width=6)
        eff_sp.pack(side="left", padx=(4, 0))
        eff_sp.bind("<Return>", self._swatch_changed)
        eff_sp.bind("<FocusOut>", self._swatch_changed)

        r3 = ttk.Frame(inp); r3.pack(fill="x", padx=6, pady=(2, 6))
        ttk.Label(r3, text="Border color:").pack(side="left")
        self._add_color_control(r3, "Border", lambda: self.border_rgb, self._set_border_rgb)
        ttk.Label(r3, text="Background:").pack(side="left", padx=(20, 0))
        self._add_color_control(r3, "Background", lambda: self.bg_rgb, self._set_bg_rgb)

        body = ttk.Frame(p)
        body.pack(fill="both", expand=True, padx=6, pady=2)
        self.prep_canvas = tk.Canvas(body, bg="#1a1a1a", highlightthickness=0)
        self.prep_canvas.pack(side="left", fill="both", expand=True)

        # Create the preview BEFORE the View panel so the zoom slider's
        # build-time .set() has a live target (its command fires immediately).
        self._zoom_syncing = False
        self.prep_view = _Preview(self.prep_canvas, self._prep_view_fn)

        vc = ttk.LabelFrame(body, text="View")
        vc.pack(side="right", fill="y", padx=(8, 2))
        ttk.Label(vc, text="Show:").grid(row=0, column=0, columnspan=2, sticky="w",
                                         padx=6, pady=(6, 0))
        self.view_var = tk.StringVar(value="master")
        self.view_cb = ttk.Combobox(vc, textvariable=self.view_var, state="readonly",
                                    width=18, values=["master", "borders"])
        self.view_cb.grid(row=1, column=0, columnspan=2, sticky="ew", padx=6)
        self.view_cb.bind("<<ComboboxSelected>>", lambda e: self.prep_view.reset())
        self.border_toggle = tk.BooleanVar(value=False)
        ttk.Checkbutton(vc, text="Overlay borders", variable=self.border_toggle,
                        command=lambda: self.prep_view.reset()).grid(
            row=2, column=0, columnspan=2, sticky="w", padx=6, pady=6)
        ttk.Separator(vc, orient="horizontal").grid(row=3, column=0, columnspan=2,
                                                    sticky="ew", padx=6, pady=4)
        ttk.Label(vc, text="Zoom").grid(row=4, column=0, columnspan=2, sticky="w", padx=6)
        self.zoom_scale = ttk.Scale(vc, from_=0.2, to=8.0, orient="horizontal",
                                    command=self._zoom_from_slider)
        self.zoom_scale.grid(row=5, column=0, columnspan=2, sticky="ew", padx=6)
        self.zoom_scale.set(1.0)
        self.zoom_lbl = ttk.Label(vc, text="zoom 1.00×")
        self.zoom_lbl.grid(row=6, column=0, columnspan=2, sticky="w", padx=6)
        ttk.Button(vc, text="Reset view",
                   command=lambda: self.prep_view.reset()).grid(
            row=7, column=0, columnspan=2, pady=(4, 4), sticky="ew", padx=6)
        ttk.Button(vc, text="Save view…", command=self.prep_save).grid(
            row=8, column=0, columnspan=2, pady=(2, 8), sticky="ew", padx=6)

        strip = ttk.LabelFrame(p, text="Colors in poster (click a color to adjust)")
        strip.pack(fill="x", padx=6, pady=(2, 2))
        self.swatch_canvas = tk.Canvas(strip, height=72, highlightthickness=0)
        self.swatch_sb = ttk.Scrollbar(strip, orient="horizontal",
                                       command=self.swatch_canvas.xview)
        self.swatch_canvas.config(xscrollcommand=self.swatch_sb.set)
        self.swatch_sb.pack(side="right", fill="y")
        self.swatch_canvas.pack(side="left", fill="both", expand=True)
        self._swatch_order = []
        self._color_edit_dialog = None

        self.status = tk.Label(p, text="Load an image and a palette, then Quantize.",
                               anchor="w", relief="sunken", background="#eee")
        self.status.pack(fill="x", padx=6, pady=(0, 4))

    def prep_pick_image(self):
        path = filedialog.askopenfilename(title="Select image",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if path:
            self._load_image(path)

    def prep_pick_palette(self):
        path = filedialog.askopenfilename(title="Select palette",
            filetypes=[("Palette", "*.txt")])
        if path:
            self._load_palette(path)

    def _load_image(self, path):
        try:
            self.image_rgb = core.load_image_rgb(path)
            self.image_path = path
            self.img_lbl.config(text=os.path.basename(path), foreground="#222")
            self._status(f"Image: {os.path.basename(path)} "
                         f"({self.image_rgb.shape[1]}×{self.image_rgb.shape[0]})")
            if self.palette:
                self.prep_quantize()
        except Exception as e:
            self._status(f"Image failed: {e}")

    def _load_palette(self, path):
        try:
            self.palette = core.load_palette(path)
            self.palette_path = path
            self.pal_lbl.config(text=os.path.basename(path), foreground="#222")
            self._status(f"Palette: {os.path.basename(path)} ({len(self.palette)} colors)")
            if self.image_rgb is not None:
                self.prep_quantize()
        except Exception as e:
            self._status(f"Palette failed: {e}")

    def prep_quantize(self):
        if self.image_rgb is None or not self.palette:
            self._status("Need an image AND a palette first.")
            return
        self._status("Quantizing…")
        self.root.update()
        try:
            num = int(float(self.num_var.get()))
            mina = int(float(self.min_var.get()))
        except ValueError:
            self._status("Num colors / min area must be whole numbers.")
            return
        exact = bool(self.exact_var.get())
        t0 = time.time()
        try:
            self.state = core.run_pipeline(
                self.image_rgb, self.palette,
                num_colors=num, exact=exact, min_area=mina,
                border_color=core.rgb_to_hex(self.border_rgb),
                bg_color=core.rgb_to_hex(self.bg_rgb),
                stem=os.path.splitext(os.path.basename(self.image_path or "image"))[0],
                palette_name=os.path.basename(self.palette_path or "palette"),
            )
        except Exception as e:
            self.state = None
            self._status(f"Pipeline failed: {e}")
            return
        self._edit_cid = None
        self._epoch += 1
        self._prep_zoom_cache = None
        self._paint_transform_cache = None
        self._build_view_menu()
        self._build_swatch()
        self.prep_view.reset()
        K = len(self.palette)
        used = len(self.state.color_data)
        mode = "full palette" if exact else f"top-{num} dominant"
        self._log(f"[quantize] {os.path.basename(self.image_path)} -> "
                  f"{self.state.W}x{self.state.H} · {mode}")
        self._log(f"           {used} of {K} palette color(s) survived "
                  f"({K - used} had no region) in {time.time() - t0:.2f}s")
        for cd in sorted(self.state.color_data, key=lambda c: -int(c['mask'].sum())):
            pct = 100.0 * int(cd['mask'].sum()) / (self.state.W * self.state.H)
            self._log(f"           · {cd['name']:<14} {cd['hex']}  {pct:5.1f}%")
        self._status(self._quantized_status(used))

    # ---- cans math (mirrors the interactive stats.html report) ----
    # The mural's second dimension is derived from the image's aspect ratio, so
    # only ONE real-world number (width OR height in metres) is needed. The
    # per-color estimate is ceil(color_wall_area / can_coverage), exactly as the
    # CLI's stats.html computes it.
    def _mural_dims(self):
        """Return (m, eff, ar) or None if no usable mural measurement.

        m    = the user's mural dimension in metres (width or height)
        eff  = can coverage in m^2/can
        ar   = image aspect ratio (width / height)
        """
        st = self.state
        if st is None:
            return None
        try:
            m = float(self.mural_var.get())
        except (ValueError, tk.TclError):
            m = 0.0
        try:
            eff = float(self.eff_var.get())
        except (ValueError, tk.TclError):
            eff = 0.0
        if m <= 0:
            return None
        W, H = st.W, st.H
        ar = (W / H) if H else 1.0
        return m, eff, ar

    def _cans_for_pct(self, pct):
        """Cans for a colour covering `pct`% of the image. Returns an int, or
        None when no mural measurement is set (rendered as "—")."""
        d = self._mural_dims()
        if d is None:
            return None
        m, eff, ar = d
        total = m * (m / ar) if self.mural_mode.get() == "width" else (m * ar) * m
        wall = total * (pct / 100.0)
        if eff <= 0:
            return 0
        return max(0, math.ceil(wall / eff - 1e-9))

    def _total_cans(self):
        """Sum of cans across all surviving colours, or None if not measurable."""
        st = self.state
        if st is None:
            return None
        total_px = st.W * st.H
        t = 0
        anymeas = False
        for cd in st.color_data:
            pct = 100.0 * int(cd["mask"].sum()) / total_px
            c = self._cans_for_pct(pct)
            if c is None:
                return None
            anymeas = True
            t += c
        return t if anymeas else None

    def _quantized_status(self, used):
        base = f"Quantized — {used} colors used. Slider to zoom, drag to pan."
        t = self._total_cans()
        if t is not None:
            base += f"  ·  {t} can(s) total"
        return base

    def _swatch_changed(self, _=None):
        """Mural measure / efficiency / width-height changed: redraw the can
        counts under each swatch and refresh the status total."""
        if self.state is None:
            return
        self._draw_swatch()
        self._status(self._quantized_status(len(self.state.color_data)))

    def _zoom_from_slider(self, v):
        """Zoom slider -> preview. Guarded so the build-time .set(1.0) and any
        pre-realization event are no-ops (canvas still 2x2)."""
        if self.prep_view is None or self.prep_view.cw <= 2:
            return
        self._zoom_syncing = True
        try:
            self.prep_view.set_zoom(float(v))
        finally:
            self._zoom_syncing = False

    def _build_view_menu(self):
        vals = ["master", "borders"] + [cd["name"] for cd in self.state.color_data]
        self.view_cb.config(values=vals)
        if self.view_var.get() not in vals:
            self.view_var.set("master")

    def _prep_base_rgba(self):
        """Native-resolution (H,W,4) view of the selected item.

        Always returns a 4-channel array; the core render helpers return a mix
        of 3- and 4-channel, so we normalize here.
        """
        st = self.state
        v = self.view_var.get()
        if v == "borders":
            base = core.render_borders_rgba(st, self.border_rgb)
        elif v == "master" or v not in [cd["name"] for cd in st.color_data]:
            if v == "master" and self.border_toggle.get():
                base = core.master_with_borders(st, self.border_rgb)
            else:
                base = core.render_master(st, st.palette_rgb)
        else:
            ov = st.borders_overlay if st.borders_overlay is not None \
                else np.zeros((st.H, st.W, 4), dtype=np.float32)
            cd = next(c for c in st.color_data if c["name"] == v)
            base = core.render_layer_png(cd["mask"],
                                         np.array(cd["rgb"], dtype="uint8"),
                                         st.W, st.H, self.bg_rgb, ov)
        base = np.asarray(base)
        if base.ndim == 3 and base.shape[2] == 3:
            a = np.full(base.shape[:2], 255, dtype="uint8")
            base = np.dstack([base, a])
        return base.astype("uint8")

    def _prep_view_fn(self):
        st = self.state
        cw, ch = self.prep_view.cw, self.prep_view.ch
        if st is None:
            return np.zeros((ch, cw, 4), dtype="uint8")
        z = self.prep_view.zoom
        self.zoom_lbl.config(text=f"zoom {z:.2f}×")
        # Keep the zoom slider in sync when zoom changes via wheel/drag.
        zs = getattr(self, "zoom_scale", None)
        if zs is not None and not self._zoom_syncing:
            try:
                if abs(float(zs.get()) - z) > 0.01:
                    self._zoom_syncing = True
                    zs.set(z)
                    self._zoom_syncing = False
            except tk.TclError:
                self._zoom_syncing = False
        # Cache the zoomed+centered base. Content changes bump _epoch; zoom/
        # canvas size are part of the key; pan is applied cheaply on top.
        key = (self._epoch, self.view_var.get(), self.border_toggle.get(),
               round(z, 4), cw, ch)
        cached = self._prep_zoom_cache
        if cached is None or cached[0] != key:
            base = self._prep_base_rgba()
            h, w = base.shape[:2]
            nw, nh = max(1, int(w * z)), max(1, int(h * z))
            zz = _resize(base, nw, nh)
            out, _ = _fit(zz, cw, ch)          # (ch,cw,4), centered
            cached = (key, out)
            self._prep_zoom_cache = cached
        out = cached[1]
        panx = int(self.prep_view.panx)
        pany = int(self.prep_view.pany)
        res = np.zeros((ch, cw, 4), dtype="uint8")
        dst_x0 = max(0, panx)
        dst_x1 = min(cw, cw + panx)
        dst_y0 = max(0, pany)
        dst_y1 = min(ch, ch + pany)
        src_x0 = dst_x0 - panx
        src_y0 = dst_y0 - pany
        if dst_x1 > dst_x0 and dst_y1 > dst_y0:
            res[dst_y0:dst_y1, dst_x0:dst_x1] = \
                out[src_y0:src_y0 + (dst_y1 - dst_y0),
                    src_x0:src_x0 + (dst_x1 - dst_x0)]
        return res

    def prep_save(self):
        if self.state is None:
            self._status("Nothing to save — run Quantize first.")
            return
        cw, ch = 900, 900
        base = self._prep_base_rgba()
        fit, _ = _fit(base, cw, ch)
        fname = self.view_var.get().replace("/", "_") + ".png"
        path = filedialog.asksaveasfilename(title="Save view",
                                            defaultextension=".png",
                                            initialfile=fname,
                                            filetypes=[("PNG", "*.png")])
        if path:
            try:
                Image.fromarray(fit, mode="RGBA").save(path)
                self._status(f"Saved → {path}")
            except Exception as e:
                self._status(f"Save failed: {e}")

    # ---- swatches ----
    def _build_swatch(self):
        self._swatch_order = [cd["ci"] for cd in
                              sorted(self.state.color_data,
                                     key=lambda c: -int(c["mask"].sum()))]
        self._draw_swatch()

    def _draw_swatch(self):
        cv = self.swatch_canvas
        cv.delete("all")
        if not self._swatch_order or self.state is None:
            cv.config(scrollregion=(0, 0, 100, 72))
            return
        cd_by_ci = {cd["ci"]: cd for cd in self.state.color_data}
        total = self.state.W * self.state.H
        x = 6
        for ci in self._swatch_order:
            cd = cd_by_ci[ci]
            hexc = core.rgb_to_hex(tuple(cd["rgb"]))
            pct = 100.0 * int(cd["mask"].sum()) / total
            sel = self._edit_cid == ci
            r = cv.create_rectangle(x, 8, x + 58, 44, fill=hexc,
                                    outline="#ff0" if sel else "#444",
                                    width=2 if sel else 1)
            cv.create_text(x + 29, 51, text=cd["name"][:13],
                           font=("TkDefaultFont", 7))
            cans = self._cans_for_pct(pct)
            canstxt = "—" if cans is None else str(cans)
            cv.create_text(x + 29, 63, text=canstxt,
                           font=("TkDefaultFont", 7), fill="#555")
            cv.tag_bind(r, "<Button-1>", lambda e, ci=ci: self._swatch_click(ci))
            x += 66
        cv.config(scrollregion=(0, 0, x + 8, 72))

    def _swatch_click(self, ci):
        st = self.state
        cd = next(c for c in st.color_data if c["ci"] == ci)
        self._edit_cid = ci
        self._draw_swatch()
        if self._color_edit_dialog is not None:
            try:
                if self._color_edit_dialog.winfo_exists():
                    self._color_edit_dialog.destroy()
            except Exception:
                pass
            self._color_edit_dialog = None
        self._color_edit_dialog = _ColorEditDialog(
            self.root, self, cd["name"], tuple(cd["rgb"]), self._apply_color)
        self._status(f"Adjusting '{cd['name']}' — preview updates live.")

    def _apply_color(self, rgb):
        """Live color edit callback from the picker (ci fixed while open)."""
        ci = self._edit_cid
        if ci is None or self.state is None:
            return
        st = self.state
        cd = next(c for c in st.color_data if c["ci"] == ci)
        cd["rgb"] = rgb
        cd["hex"] = core.rgb_to_hex(rgb)
        idx = st.active_idx[ci]
        st.palette_rgb[idx] = np.array(rgb, dtype="uint8")
        self.palette[idx][1] = rgb
        # content changed: invalidate the zoomed-base cache
        self._epoch += 1
        self._prep_zoom_cache = None
        if self.view_var.get() in ("master", cd["name"]):
            self.prep_view.refresh()
        self._draw_swatch()


    # ================================================================== PAINT
    def _build_paint(self):
        p = self.paint
        self.paint_canvas = tk.Canvas(p, bg="#1a1a1a", highlightthickness=0)
        self.paint_canvas.pack(side="left", fill="both", expand=True)

        # create the preview FIRST so slider .set() callbacks during build
        # (which call paint_view.refresh()) have a live target.
        self.paint_cx = 0.0
        self.paint_cy = 0.0
        self._centered = False
        self.paint_view = _Preview(self.paint_canvas, self._paint_view_fn, pan=False)
        self.paint_view.canvas.bind("<ButtonPress-1>", self._paint_press)
        self.paint_view.canvas.bind("<B1-Motion>", self._paint_move)
        self._paint_last = None

        vc = ttk.LabelFrame(p, text="Placement")
        vc.pack(side="right", fill="y", padx=(8, 2), pady=2)

        ttk.Button(vc, text="Upload background…",
                   command=self.paint_pick_bg).grid(
            row=0, column=0, columnspan=2, sticky="ew", padx=6, pady=(6, 2))
        ttk.Label(vc, text="Overlay: borders.png (always)").grid(
            row=1, column=0, columnspan=2, sticky="w", padx=6, pady=(2, 0))
        ttk.Separator(vc, orient="horizontal").grid(row=2, column=0, columnspan=2,
                                                    sticky="ew", padx=6, pady=6)

        self.paint_scale_lbl = ttk.Label(vc, text="1.00×")
        self.paint_scale = self._numfield(vc, 3, "Scale", "1.0", self.paint_scale_lbl, "x")
        self.paint_rot_lbl = ttk.Label(vc, text="0°")
        self.paint_rot = self._numfield(vc, 4, "Rotate", "0", self.paint_rot_lbl, "deg")
        self.paint_op_lbl = ttk.Label(vc, text="1.00")
        self.paint_opacity = self._slider(vc, 5, "Opacity", 0.0, 1.0, 1.0,
                                          self.paint_op_lbl,
                                          lambda v: self.paint_view.refresh())
        ttk.Label(vc, text="Blend:").grid(row=6, column=0, sticky="w", padx=6)
        self.blend_var = tk.StringVar(value="normal")
        self.blend_cb = ttk.Combobox(vc, textvariable=self.blend_var, state="readonly",
                                     width=16, values=BLENDS)
        self.blend_cb.grid(row=6, column=1, sticky="ew", padx=6)
        self.blend_cb.bind("<<ComboboxSelected>>", lambda e: self.paint_view.refresh())
        ttk.Button(vc, text="Reset placement",
                   command=self.paint_reset).grid(
            row=7, column=0, columnspan=2, sticky="ew", padx=6, pady=(10, 2))
        ttk.Button(vc, text="Save composition…",
                   command=self.paint_save).grid(
            row=8, column=0, columnspan=2, sticky="ew", padx=6, pady=(2, 8))

    def _slider(self, parent, row, label, lo, hi, val, lbl, on_change):
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6)
        sc = ttk.Scale(parent, from_=lo, to=hi, orient="horizontal",
                       command=on_change)
        sc.set(val)
        sc.grid(row=row, column=1, sticky="ew", padx=6)
        lbl.grid(row=row + 1, column=0, columnspan=2, sticky="w", padx=6)
        return sc

    def _numfield(self, parent, row, label, val, lbl, unit):
        """A numeric entry (label + spinbox). Updates lbl and refreshes on edit.
        Used for paint Scale / Rotate (exact entry beats a drag slider)."""
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", padx=6)
        var = tk.StringVar(value=val)
        en = ttk.Spinbox(parent, textvariable=var, from_=-1000.0, to=1000.0,
                         increment=0.5, width=8)
        en.grid(row=row, column=1, sticky="ew", padx=6)

        def _upd(_=None):
            try:
                f = float(var.get())
            except (ValueError, tk.TclError):
                return
            lbl.config(text=("" if unit == "deg" else "") +
                       (f"{f:.2f}{unit}" if unit == "x"
                        else f"{f:.0f}°"))
            self.paint_view.refresh()

        en.bind("<Return>", _upd)
        en.bind("<FocusOut>", _upd)
        return en

    def paint_pick_bg(self):
        path = filedialog.askopenfilename(title="Background / wall photo",
            filetypes=[("Images", "*.png *.jpg *.jpeg *.webp *.bmp")])
        if path:
            try:
                self._bg = core.load_image_rgb(path)
                self._centered = False
                self.paint_view.refresh()
            except Exception as e:
                self._status(f"Background failed: {e}")

    def _paint_press(self, e):
        if not self._centered:
            self.paint_cx = self.paint_view.cw / 2
            self.paint_cy = self.paint_view.ch / 2
            self._centered = True
        self._paint_last = (e.x, e.y)

    def _paint_move(self, e):
        if self._paint_last is None:
            return
        self.paint_cx += (e.x - self._paint_last[0])
        self.paint_cy += (e.y - self._paint_last[1])
        self._paint_last = (e.x, e.y)
        self.paint_view.refresh()

    def paint_reset(self):
        self.paint_cx = self.paint_view.cw / 2
        self.paint_cy = self.paint_view.ch / 2
        self.paint_scale.set("1.0")
        self.paint_rot.set("0")
        self.paint_opacity.set(1.0)
        self.paint_scale_lbl.config(text="1.00×")
        self.paint_rot_lbl.config(text="0°")
        self.paint_op_lbl.config(text="1.00")
        self._paint_transform_cache = None
        self.paint_view.refresh()

    def _bg_fitted(self, cw, ch):
        if self._bg is None:
            return np.full((ch, cw, 3), 30, dtype="uint8")
        key = (self._bg.shape[0], self._bg.shape[1])
        cached, kcw, kch, ckey = self._bg_cache
        if cached is not None and (kcw, kch) == (cw, ch) and key == ckey:
            return cached
        fit, _ = _fit(self._bg, cw, ch)
        self._bg_cache = (fit.astype("uint8"), cw, ch, key)
        return fit

    def _paint_layer_rgba(self):
        st = self.state
        # The paint overlay is always borders.png (the layer picker was removed).
        return core.render_borders_rgba(st, self.border_rgb)

    def _paint_view_fn(self):
        cw, ch = self.paint_view.cw, self.paint_view.ch
        bg = self._bg_fitted(cw, ch).astype("float32") / 255.0
        if self.state is None:
            return (bg * 255).clip(0, 255).astype("uint8")
        layer = self._paint_layer_rgba()
        scale = float(self.paint_scale.get())
        rot = float(self.paint_rot.get())
        opacity = float(self.paint_opacity.get())
        blend = self.blend_var.get()
        self.paint_scale_lbl.config(text=f"{scale:.2f}×")
        self.paint_rot_lbl.config(text=f"{rot:.0f}°")
        self.paint_op_lbl.config(text=f"{opacity:.2f}")
        if layer is None:
            return (bg * 255).clip(0, 255).astype("uint8")

        # The scaled+rotated layer is the expensive part (PIL resize + rotate).
        # Cache it by content/layer/scale/rot so dragging (position) and
        # opacity/blend changes are cheap re-composites, not re-transforms.
        tkey = (self._epoch, "borders", round(scale, 4), round(rot, 3))
        cached = self._paint_transform_cache
        if cached is None or cached[0] != tkey:
            rgb = layer[:, :, :3]
            alpha = layer[:, :, 3]
            h, w = rgb.shape[:2]
            base = min(cw, ch)
            target = base * scale
            nw = max(1, int(w * target / max(1, min(w, h))))
            nh = max(1, int(h * target / max(1, min(w, h))))
            zrgb = _resize(rgb, nw, nh).astype("uint8")
            zalpha = _resize(alpha.astype("uint8"), nw, nh).astype("uint8")
            if abs(rot) > 0.01:
                zrgb = np.asarray(Image.fromarray(zrgb).rotate(
                    rot, resample=Image.BILINEAR, expand=True))
                zalpha = np.asarray(Image.fromarray(zalpha).rotate(
                    rot, resample=Image.BILINEAR, expand=True))
            cached = (tkey, zrgb.astype("float32") / 255.0,
                      zalpha.astype("float32") / 255.0)
            self._paint_transform_cache = cached
        _, zrgb, zalpha = cached

        zh, zw = zrgb.shape[:2]
        x0, y0 = int(self.paint_cx - zw / 2), int(self.paint_cy - zh / 2)
        sx0, sy0 = max(0, x0), max(0, y0)
        ex, ey = min(cw, x0 + zw), min(ch, y0 + zh)
        out = bg.copy()
        if sx0 < ex and sy0 < ey:
            fx, fy = sx0 - x0, sy0 - y0
            w2, h2 = ex - sx0, ey - sy0
            a = (zalpha[fy:fy + h2, fx:fx + w2] * opacity)[..., None]
            top = zrgb[fy:fy + h2, fx:fx + w2]
            bot = out[sy0:ey, sx0:ex]
            sub = top if blend == "normal" else core._blend(top, bot, blend)
            out[sy0:ey, sx0:ex] = bot * (1 - a) + sub * a
        return (out * 255).clip(0, 255).astype("uint8")

    def paint_save(self):
        if self.state is None:
            self._status("Run Quantize in the Prep tab first.")
            return
        path = filedialog.asksaveasfilename(title="Save composition",
                                            defaultextension=".png",
                                            initialfile="paint.png",
                                            filetypes=[("PNG", "*.png")])
        if path:
            try:
                Image.fromarray(self._paint_view_fn()).save(path)
                self._status(f"Saved → {path}")
            except Exception as e:
                self._status(f"Save failed: {e}")


class _ColorEditDialog(tk.Toplevel):
    """Photoshop-style color editor for a single poster color.

    Layout: a big square (Hue across X, Saturation up Y) + a single Lightness
    slider + live preview. Changes apply to the poster immediately (live), so
    there is no OK/Cancel — close when done. Dark theme to match the app.
    """

    SQ = 220  # square side in px
    BG = "#262626"

    def __init__(self, master, app, name, rgb, apply_cb):
        super().__init__(master)
        self.app = app
        self.title(f"Adjust color — {name}")
        self.resizable(False, False)
        self.configure(padx=10, pady=10, bg=self.BG)
        self.apply_cb = apply_cb
        h, s, l = rgb_to_hsl(rgb)
        self.h = h % 1.0
        self.s = s
        self.l = l

        top = tk.Frame(self, bg=self.BG); top.pack(fill="x")
        self.canvas = tk.Canvas(top, width=self.SQ, height=self.SQ,
                                highlightthickness=1, highlightbackground="#555")
        self.canvas.pack(side="left")
        self.canvas.bind("<ButtonPress-1>", self._sq)
        self.canvas.bind("<B1-Motion>", self._sq)

        side = tk.Frame(self, bg=self.BG); side.pack(side="left", padx=(10, 0))
        self.prev = tk.Canvas(side, width=72, height=72, highlightthickness=1,
                              highlightbackground="#555", bg=self.BG)
        self.prev.pack(pady=(0, 6))
        self.hexlbl = tk.Label(side, text="", bg=self.BG, fg="#e6e6e6",
                               font=("TkFixedFont", 10))
        self.hexlbl.pack(pady=(0, 10))
        ttk.Label(side, text="Lightness").pack(anchor="w")
        self.lscale = ttk.Scale(side, orient="vertical", length=self.SQ - 40,
                                from_=0.0, to=1.0, command=self._lmove)
        self.lscale.pack(pady=(2, 0))

        self.img = None
        self._sq_cache_l = None
        self._prev_rect = self.prev.create_rectangle(0, 0, 72, 72, outline="")
        self._marker = self.canvas.create_oval(0, 0, 6, 6, width=2, outline="white")
        self.canvas.tag_raise(self._marker)
        self._sync()
        self.protocol("WM_DELETE_WINDOW", self._close)

    def _sq(self, e):
        x = max(0, min(self.SQ - 1, e.x)) / (self.SQ - 1)
        y = 1.0 - (max(0, min(self.SQ - 1, e.y)) / (self.SQ - 1))  # up = more sat
        self.h = x
        self.s = y
        self._sync()

    def _lmove(self, v):
        self.l = float(v)
        self._sq_cache_l = None  # square only depends on lightness
        self._sync()

    def _rgb(self):
        return tuple(max(0, min(255, x)) for x in hsl_to_rgb(self.h, self.s, self.l))

    def _sync(self):
        hexc = core.rgb_to_hex(self._rgb())
        self.hexlbl.config(text=hexc)
        self.prev.itemconfig(self._prev_rect, fill=hexc)
        if self._sq_cache_l != self.l:
            self._sq_cache_l = self.l
            self._draw_square()
        # marker at (h, s) — moves on hue/sat without a square redraw
        mx = int(self.h * (self.SQ - 1))
        my = int((1.0 - self.s) * (self.SQ - 1))
        self.canvas.coords(self._marker, mx - 5, my - 5, mx + 5, my + 5)
        self.apply_cb(self._rgb())

    def _draw_square(self):
        # hue(X) × saturation(Y) field at the current lightness, computed in a
        # single numpy pass (no per-pixel Python loop) so lightness drags stay
        # responsive even while the preview re-composites behind the dialog.
        N = self.SQ
        xs = np.linspace(0.0, 1.0, N)
        ys = np.linspace(1.0, 0.0, N)  # row 0 = top = s=1
        Hg, Sg = np.meshgrid(xs, ys)
        Lg = np.full((N, N), self.l, dtype=np.float64)
        arr = _hsl_to_rgb_vec(Hg, Sg, Lg)
        if self.img is None:
            self.img = tk.PhotoImage(width=N, height=N)
        self.img.put(_hex_rows(arr))
        self.canvas.delete("sq")
        self.canvas.create_image(0, 0, image=self.img, anchor="nw", tags="sq")
        self.canvas.tag_lower("sq")
        self.canvas.tag_raise(self._marker)

    def _close(self):
        self.destroy()
        try:
            if self.app is not None:
                self.app._color_edit_dialog = None
        except Exception:
            pass



def main():
    import sys
    img = sys.argv[1] if len(sys.argv) > 1 else None
    pal = sys.argv[2] if len(sys.argv) > 2 else None
    root = tk.Tk()
    FlashpointApp(root, img, pal)
    root.mainloop()


if __name__ == "__main__":
    main()
