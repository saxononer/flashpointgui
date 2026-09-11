#!/usr/bin/env python3
"""
flashpoint_core.py — importable engine for the flashpoint posterizer.

This is the algorithm, pulled out of the original flashpoint.py so that both
a CLI (flashpoint.py) and a GUI (flashpoint_gui.py) can drive it. Nothing here
imports tkinter or touches the command line — it is pure numpy + PIL + scipy.

Two cost tiers (the whole reason this module exists):

  * EXPENSIVE  — Lab convert, nearest-color assignment, region cleanup,
                 contour trace, border polyline dedupe. These depend only on the
                 RELATIVE positions of the palette colors in Lab space. A palette
                 color swap that does not change which colors are "dominant"
                 leaves the assignment + masks + contours UNCHANGED.

  * CHEAP      — turning (assignment, masks, contours) into actual pixels: the
                 master image, the borders overlay, the per-color layers, and the
                 photo compositing. These depend on the EXACT rgb values.

So the GUI can swap a palette color and re-render in milliseconds: it reuses the
cached expensive state and only recomputes the cheap render path. run_pipeline()
re-does the expensive tier (fast enough, and always correct); render_* does the
cheap tier on a PipelineState.

Deterministic: byte-identical re-runs, same as the original.
"""

import os
import re
import json
import sys
import time
import base64
from collections import Counter

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage as _ndi
from scipy.spatial import cKDTree


def fail(msg):
    print(f"ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


# ==========================================================================
# Palette
# ==========================================================================

HEX_ENTRY_RE = re.compile(r'^\s*([^,]+?)\s*,\s*#([0-9a-fA-F]{6})\s*$')
BARE_HEX_RE = re.compile(r'^\s*#[0-9a-fA-F]{6}\s*$')


def hex_to_rgb(h):
    return (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def rgb_to_hex(rgb):
    r, g, b = (int(v) for v in rgb)
    return f"#{r:02X}{g:02X}{b:02X}"


def load_palette(path):
    """Parse 'name, #rrggbb' lines. Returns list of [name, (r,g,b)].

    NOTE: returns lists (mutable) rather than the original's tuples so the GUI
    can edit a color's rgb in place.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw_lines = f.readlines()
    except OSError as e:
        fail(f"Cannot read palette file {path!r}: {e}")

    colors = []
    seen_names = {}
    for lineno, raw in enumerate(raw_lines, start=1):
        line = raw.rstrip("\r\n")
        stripped = line.strip()
        if stripped == "":
            continue
        if line.lstrip().startswith("#"):
            if BARE_HEX_RE.match(line):
                fail(f"Palette error, line {lineno}: expected 'name, #rrggbb' "
                     f"format, got '{line}'.")
            continue
        m = HEX_ENTRY_RE.match(line)
        if not m:
            fail(f"Palette error, line {lineno}: expected 'name, #rrggbb' "
                 f"format, got '{line}'.")
        name = m.group(1).strip()
        hexcode = m.group(2).upper()
        if name == "":
            fail(f"Palette error, line {lineno}: expected 'name, #rrggbb' "
                 f"format, got '{line}'.")
        if name in seen_names:
            fail(f"Palette error, line {lineno}: duplicate color name {name!r} "
                 f"(first seen on line {seen_names[name]}).")
        seen_names[name] = lineno
        colors.append([name, hex_to_rgb(hexcode)])

    if len(colors) < 2:
        fail(f"Palette error: fewer than 2 valid colors found (got {len(colors)}).")
    return colors


def _parse_color(s):
    s = s.strip()
    if s.startswith("#"):
        s = s[1:]
    if len(s) != 6:
        raise ValueError(f"bad color {s!r}")
    return (int(s[0:2], 16), int(s[2:4], 16), int(s[4:6], 16))


# ==========================================================================
# Color space: sRGB -> CIELAB (D65), inline numpy
# ==========================================================================

def _srgb_to_lab_f(rgb):
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    M = np.array([
        [0.4124564, 0.3575761, 0.1804375],
        [0.2126729, 0.7151522, 0.0721750],
        [0.0193339, 0.1191920, 0.9503041],
    ])
    xyz = linear @ M.T
    Xn, Yn, Zn = 0.95047, 1.0, 1.08883
    x = xyz[..., 0] / Xn
    y = xyz[..., 1] / Yn
    z = xyz[..., 2] / Zn
    delta = 6.0 / 29.0
    d3 = delta ** 3
    fx = np.where(x > d3, np.cbrt(x), x / (3.0 * delta ** 2) + 4.0 / 29.0)
    fy = np.where(y > d3, np.cbrt(y), y / (3.0 * delta ** 2) + 4.0 / 29.0)
    fz = np.where(z > d3, np.cbrt(z), z / (3.0 * delta ** 2) + 4.0 / 29.0)
    L = 116.0 * fy - 16.0
    a = 500.0 * (fx - fy)
    b = 200.0 * (fy - fz)
    return np.stack([L, a, b], axis=-1)


def rgb_to_lab(rgb_u8):
    img = rgb_u8.astype(np.float64) / 255.0
    return _srgb_to_lab_f(img).astype(np.float32)


# ==========================================================================
# Nearest-color assignment
# ==========================================================================

def assign_nearest(lab_pixels, lab_colors):
    """Nearest in Euclidean Lab. Returns int32 index array (N,) into lab_colors."""
    tree = cKDTree(lab_colors.astype(np.float64))
    _, idx = tree.query(lab_pixels.astype(np.float64))
    return idx.astype(np.int32)


# ==========================================================================
# Connected components + cleanup
# ==========================================================================

def label_components4(mask):
    structure = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=np.int32)
    labels, num = _ndi.label(mask, structure=structure)
    return labels.astype(np.int32), int(num)


def _dilate8(mask):
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def cleanup_regions(assigned, active_idx, min_area):
    if min_area <= 0:
        return assigned
    H, W = assigned.shape
    A = len(active_idx)
    result = assigned.copy()
    max_passes = 25
    for _pass in range(max_passes):
        changed = False
        for ci in range(A):
            comp_mask = (result == ci)
            if not comp_mask.any():
                continue
            labels, num = label_components4(comp_mask)
            if num < 2:
                continue
            sizes = np.bincount(labels.ravel(), minlength=num + 1)
            sizes = sizes[1:]
            small_labels = [l for l in range(1, num + 1) if sizes[l - 1] < min_area]
            if not small_labels:
                continue
            small_mask = np.isin(labels, small_labels)
            neigh = _dilate8(small_mask)
            outside = neigh & ~small_mask
            nb_counts = Counter()
            nb_vals = result[outside]
            for v, cnt in zip(nb_vals, np.repeat(1, nb_vals.size)):
                nb_counts[int(v)] += 1
            if nb_counts:
                target = max(nb_counts.items(), key=lambda kv: (kv[1], -kv[0]))[0]
            else:
                target = 0
            if target != ci:
                result[small_mask] = target
                changed = True
        if not changed:
            break
    return result


# ==========================================================================
# Boundary tracing -> closed pixel-aligned contours
# ==========================================================================

def marching_squares_closed(mask):
    H, W = mask.shape
    if H == 0 or W == 0 or not mask.any():
        return []

    def filled(y, x):
        return 0 <= y < H and 0 <= x < W and bool(mask[y, x])

    edges = []
    idx = {}

    def add(A, B, lp):
        i = len(edges)
        edges.append((A, B, lp))
        idx.setdefault(A, []).append(i)

    for y in range(H):
        for x in range(W):
            if not mask[y, x]:
                continue
            if not filled(y - 1, x):
                add((x + 1, y), (x, y), (x, y))
            if not filled(y + 1, x):
                add((x, y + 1), (x + 1, y + 1), (x, y))
            if not filled(y, x - 1):
                add((x, y), (x, y + 1), (x, y))
            if not filled(y, x + 1):
                add((x + 1, y + 1), (x + 1, y), (x, y))

    if not edges:
        return []

    def next_edge(corner, lp_in):
        cands = idx[corner]
        best = None
        best_key = None
        for i in cands:
            lp = edges[i][2]
            dist = abs(lp[0] - lp_in[0]) + abs(lp[1] - lp_in[1])
            if dist > 1:
                continue
            key = (0 if dist == 0 else 1, i)
            if best_key is None or key < best_key:
                best = i
                best_key = key
        if best is not None:
            return best
        return min(cands)

    visited = [False] * len(edges)
    contours = []
    start_ids = sorted(range(len(edges)), key=lambda i: (edges[i][0], edges[i][2]))
    for si in start_ids:
        if visited[si]:
            continue
        A, B, lp = edges[si]
        visited[si] = True
        path = [A, B]
        cur = B
        cur_lp = lp
        guard = 0
        while guard <= len(edges) + 2:
            nxt = next_edge(cur, cur_lp)
            if visited[nxt]:
                break
            visited[nxt] = True
            _, B2, lp2 = edges[nxt]
            path.append(B2)
            cur = B2
            cur_lp = lp2
            guard += 1
        dedup = [path[0]]
        for p in path[1:]:
            if p != dedup[-1]:
                dedup.append(p)
        if len(dedup) >= 3 and dedup[0] == dedup[-1]:
            contours.append(dedup)
    return contours


def border_polylines(contours_by_color, W, H):
    hsegs = set()
    vsegs = set()
    for contours in contours_by_color:
        for cont in contours:
            for i in range(len(cont) - 1):
                ax, ay = cont[i]
                bx, by = cont[i + 1]
                if ay == by:
                    x, y = min(ax, bx), ay
                    if y == 0 or y == H:
                        continue
                    hsegs.add((x, y))
                else:
                    x, y = ax, min(ay, by)
                    if x == 0 or x == W:
                        continue
                    vsegs.add((x, y))
    polylines = []
    rows = {}
    for (x, y) in hsegs:
        rows.setdefault(y, set()).add(x)
    for y in sorted(rows):
        xs = sorted(rows[y])
        i = 0
        while i < len(xs):
            j = i
            while j + 1 < len(xs) and xs[j + 1] == xs[j] + 1:
                j += 1
            x0, x1 = xs[i], xs[j] + 1
            polylines.append([(x0, y), (x1, y)])
            i = j + 1
    cols = {}
    for (x, y) in vsegs:
        cols.setdefault(x, set()).add(y)
    for x in sorted(cols):
        ys = sorted(cols[x])
        i = 0
        while i < len(ys):
            j = i
            while j + 1 < len(ys) and ys[j + 1] == ys[j] + 1:
                j += 1
            y0, y1 = ys[i], ys[j] + 1
            polylines.append([(x, y0), (x, y1)])
            i = j + 1
    return polylines


# ==========================================================================
# Render (cheap tier)
# ==========================================================================

def render_borders_overlay(polylines, W, H, color, stroke_width=2.0, ss=4):
    """Border polylines -> transparent (H,W,4) float32 RGBA, supersampled."""
    S = ss
    big = Image.new("RGBA", (W * S, H * S), (0, 0, 0, 0))
    dr = ImageDraw.Draw(big)
    wbig = max(1, int(round(stroke_width * S)))
    rbig = wbig // 2
    for line in polylines:
        if len(line) < 2:
            continue
        xy = [(x * S, y * S) for (x, y) in line]
        for a, b in zip(xy, xy[1:]):
            dr.line([a, b], fill=color + (255,), width=wbig, joint="curve")
        for (px, py) in (xy[0], xy[-1]):
            dr.ellipse([px - rbig, py - rbig, px + rbig, py + rbig],
                       fill=color + (255,))
    small = big.resize((W, H), Image.BILINEAR)
    return np.asarray(small, dtype=np.float32)


def render_master(state, palette_rgb):
    """(H,W,3) uint8: the flat posterized image at the current palette rgb."""
    active_rgb = palette_rgb[state.active_idx]
    return active_rgb[state.assigned]


def render_layer_png(color_mask, rgb, W, H, bg_rgb, borders_overlay):
    base = np.zeros((H, W, 3), dtype=np.float32)
    base[:, :, 0] = bg_rgb[0]
    base[:, :, 1] = bg_rgb[1]
    base[:, :, 2] = bg_rgb[2]
    for c in range(3):
        base[:, :, c] = np.where(color_mask, float(rgb[c]), base[:, :, c])
    a = borders_overlay[:, :, 3:4] / 255.0
    out = borders_overlay[:, :, :3] * a + base * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


def render_borders_rgba(state, border_rgb):
    """(H,W,4) uint8: transparent line-work in border_rgb (same as original)."""
    W, H = state.W, state.H
    if state.borders_overlay is not None:
        bimg = np.zeros((H, W, 4), dtype=np.uint8)
        bimg[:, :, 0] = border_rgb[0]
        bimg[:, :, 1] = border_rgb[1]
        bimg[:, :, 2] = border_rgb[2]
        bimg[:, :, 3] = np.clip(state.borders_overlay[:, :, 3], 0, 255).astype(np.uint8)
    else:
        bimg = np.zeros((H, W, 4), dtype=np.uint8)
    return bimg


def master_with_borders(state, border_rgb):
    """(H,W,3) uint8: master posterized image with the border line-work composited
    on top (magenta lines over the flat fills). This is what the GUI composites
    over the photo."""
    W, H = state.W, state.H
    master = render_master(state, state.palette_rgb)
    if state.borders_overlay is None:
        return master
    base = master.astype(np.float32)
    ov = state.borders_overlay
    # re-tint the overlay to the chosen border color (alpha from overlay)
    a = ov[:, :, 3:4] / 255.0
    col = np.zeros_like(ov[:, :, :3])
    col[:, :, 0] = border_rgb[0]
    col[:, :, 1] = border_rgb[1]
    col[:, :, 2] = border_rgb[2]
    out = col * a + base * (1.0 - a)
    return np.clip(out, 0, 255).astype(np.uint8)


# ==========================================================================
# Photo compositing (the "blend layers over a camera photo" feature)
# ==========================================================================

# Each blend mode is a numpy op applied per-pixel. `top` = posterized layer
# (master+borders), `bottom` = photo, both (H,W,3) float32 in [0,1].
# `opacity` in [0,1] mixes the result back toward the photo.

def _clamp01(x):
    return np.clip(x, 0.0, 1.0)


def _blend(top, bottom, mode):
    t = _clamp01(top)
    b = _clamp01(bottom)
    if mode == "normal":
        out = t
    elif mode == "multiply":
        out = t * b
    elif mode == "screen":
        out = 1.0 - (1.0 - t) * (1.0 - b)
    elif mode == "overlay":
        out = np.where(b < 0.5, 2.0 * t * b, 1.0 - 2.0 * (1.0 - t) * (1.0 - b))
    elif mode == "softlight":
        out = np.where(b <= 0.5,
                       b - (1.0 - 2.0 * t) * b * (1.0 - b),
                       b + (2.0 * t - 1.0) * (b - b * b))
    elif mode == "darken":
        out = np.minimum(t, b)
    elif mode == "lighten":
        out = np.maximum(t, b)
    elif mode == "difference":
        out = np.abs(t - b)
    elif mode == "color_dodge":
        # b / (1 - t), with t==1 -> 1
        with np.errstate(divide="ignore", invalid="ignore"):
            out = b / np.where(t < 1.0, 1.0 - t, 1.0)
        out = np.where(t >= 1.0, 1.0, out)
    elif mode == "color_burn":
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 1.0 - np.where((1.0 - b) > 0.0, (1.0 - b) / (t + 1e-12), 1.0)
        out = np.where(t <= 0.0, 0.0, out)
    elif mode == "hardlight":
        out = np.where(t < 0.5, 2.0 * t * b, 1.0 - 2.0 * (1.0 - t) * (1.0 - b))
    elif mode == "luminosity":
        # keep top's lightness, bottom's hue/sat (approx via Rec.709 luma)
        tl = 0.2126 * t[..., 0] + 0.7152 * t[..., 1] + 0.0722 * t[..., 2]
        bl = 0.2126 * b[..., 0] + 0.7152 * b[..., 1] + 0.0722 * b[..., 2]
        out = b + (tl - bl)[..., None]
    else:
        raise ValueError(f"unknown blend mode {mode!r}")
    return _clamp01(out)


def composite_over_photo(state, photo_rgb, border_rgb, mode="normal", opacity=1.0):
    """Composite the posterized layers (master + borders) over `photo_rgb` using
    the given blend mode. photo_rgb: (H,W,3) uint8 (resized to state size).
    Returns (H,W,3) uint8."""
    if photo_rgb.shape[:2] != (state.H, state.W):
        photo_rgb = np.asarray(
            Image.fromarray(photo_rgb).resize((state.W, state.H), Image.BILINEAR),
            dtype=np.uint8)
    top = master_with_borders(state, border_rgb).astype(np.float32) / 255.0
    bottom = photo_rgb.astype(np.float32) / 255.0
    out = _blend(top, bottom, mode)
    photo_f = photo_rgb.astype(np.float32) / 255.0
    out = out * opacity + photo_f * (1.0 - opacity)
    return _clamp01(out).astype(np.uint8)


# ==========================================================================
# Pipeline state
# ==========================================================================

class PipelineState:
    """Holds the expensive-computed results of run_pipeline().

    Re-render the cheap tier (master, borders, layers, composites) from any
    PipelineState without re-running quantization — this is what makes live
    color-swap instant.
    """
    __slots__ = ("rgb", "W", "H", "palette", "palette_rgb", "names",
                 "active_idx", "assigned", "final_present", "color_data",
                 "all_contours", "polylines", "borders_overlay", "_lab_img",
                 "_stem", "_palette_name")

    def rgb_lab(self, rgb_u8):
        return rgb_to_lab(rgb_u8)


def run_pipeline(rgb, palette, num_colors=10, exact=False, min_area=50,
                 border_color="#ff00ff", bg_color="#808080",
                 stem="image", palette_name="unknown"):
    """Run the full expensive pipeline.

    rgb: (H,W,3) uint8.
    palette: list of [name, (r,g,b)] (mutable lists fine).
    Returns a PipelineState.
    """
    W, H = rgb.shape[1], rgb.shape[0]
    names = [c[0] for c in palette]
    palette_rgb = np.array([c[1] for c in palette], dtype=np.uint8)  # (K,3)
    K = len(palette)

    if num_colors > K:
        num_colors = K

    lab_img = rgb_to_lab(rgb)                       # (H,W,3)
    lab_flat = lab_img.reshape(-1, 3)
    lab_palette = rgb_to_lab(palette_rgb)           # (K,3)

    # ---- choose active colors ---------------------------------------
    if exact:
        active_idx = list(range(K))
    else:
        step = 4
        cap = 250_000
        sub = lab_img[::step, ::step, :]
        if sub.size // 3 > cap:
            r, c = sub.shape[0], sub.shape[1]
            scale = max(1, int((r * c) / cap) ** 0.5)
            sub = sub[::scale, ::scale, :]
        sub_flat = sub.reshape(-1, 3).astype(np.float32)
        winner = assign_nearest(sub_flat, lab_palette)
        counts = Counter(int(w) for w in winner)
        ranked = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
        top = [idx for idx, _ in ranked[:num_colors]]
        active_idx = sorted(top)

    A = len(active_idx)
    active_lab = lab_palette[active_idx]

    # ---- quantize ---------------------------------------------------
    idx = assign_nearest(lab_flat, active_lab)
    assigned = idx.reshape(H, W).astype(np.int32)

    # ---- cleanup ----------------------------------------------------
    if min_area > 0:
        assigned = cleanup_regions(assigned, active_idx, min_area)

    active_rgb = palette_rgb[active_idx]
    final_present = sorted(np.unique(assigned).tolist())

    # ---- contours + per-color data ----------------------------------
    all_contours = []
    color_data = []
    active_names = [names[i] for i in active_idx]
    for ci in final_present:
        cname = active_names[ci]
        mask = (assigned == ci)
        if not mask.any():
            continue
        contours = marching_squares_closed(mask)
        all_contours.append(contours)
        r, g, b = (int(v) for v in active_rgb[ci])
        hex_up = f"#{r:02X}{g:02X}{b:02X}"
        color_data.append({"ci": ci, "name": cname, "mask": mask,
                           "hex": hex_up, "rgb": (r, g, b)})

    polylines = border_polylines(all_contours, W, H)

    st = PipelineState()
    st.rgb = rgb
    st.W, st.H = W, H
    st.palette = palette
    st.palette_rgb = palette_rgb
    st.names = names
    st.active_idx = active_idx
    st.assigned = assigned
    st.final_present = final_present
    st.color_data = color_data
    st.all_contours = all_contours
    st.polylines = polylines
    st._lab_img = lab_img
    # borders overlay rendered with the CURRENT border color; GUI re-renders
    # cheaply per color via master_with_borders (which re-tints on the fly).
    st.borders_overlay = (render_borders_overlay(polylines, W, H, (255, 0, 255))
                          if polylines else None)
    st._stem = stem
    st._palette_name = palette_name
    return st


# ==========================================================================
# paintlist.html — interactive paint-coverage report (self-contained copy)
# ==========================================================================

STATS_HTML = """<!doctype html>
<html lang="en"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>flashpointgui - __TITLE__</title>
<style>
  body { font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
         font-size: 14px; line-height: 1.5; color: #000; background: #fff;
         margin: 0; padding: 24px; }
  h1 { font-size: 16px; font-weight: 600; margin: 0 0 4px; }
  .sub { font-size: 12px; color: #444; margin-bottom: 20px; }
  .controls { margin-bottom: 8px; }
  .controls label { display: inline-block; min-width: 160px; }
  .controls input[type=number] { width: 120px; font: inherit; }
  .seg { display: inline-flex; }
  .seg button { font: inherit; padding: 2px 10px; border: 1px solid #000; cursor: pointer;
                background: #fff; }
  .seg button.on { background: #000; color: #fff; }
  table { border-collapse: collapse; margin-top: 12px; }
  th, td { border: 1px solid #888; padding: 4px 10px; text-align: right; }
  th:first-child, td:first-child { text-align: left; }
  th { font-weight: 600; }
  .swatch { display: inline-block; width: 12px; height: 12px; vertical-align: -1px;
            margin-right: 8px; border: 1px solid #000; }
  tfoot td { font-weight: 600; }
</style>
</head>
<body>
  __LOGO__
  <h1>flashpointgui - __TITLE__</h1>
  <div class="sub">__DIM__ &middot; __NCOLORS__ colors &middot; palette: __PALETTE__</div>

  <div class="controls">
    The value below is the
    <span class="seg" id="mode">
      <button data-m="width" class="on">width</button>
      <button data-m="height">height</button>
    </span><br>
    Mural <span id="mlabel">width</span> (m):
    <input type="number" id="measure" min="0" step="0.1" value="3" inputmode="decimal">
    <br>Can coverage (m&sup2;/can):
    <input type="number" id="eff" min="0.001" step="0.1" value="3" inputmode="decimal">
  </div>
  <div>Mural will be <b id="dims">&mdash;</b></div>

  <table>
    <thead><tr><th>Color</th><th>Pixel %</th><th>Cans</th></tr></thead>
    <tbody id="rows"></tbody>
    <tfoot><tr><td>Total</td><td>100</td><td id="tCans">&mdash;</td></tr></tfoot>
  </table>

<script>
const AR = __AR__;              // image aspect ratio = width / height
const ROWS = __ROWS__;          // [{name, hex, pct}]
const $ = s => document.querySelector(s);
let mode = "width";

function dimsFrom(m){
  return mode === "width" ? { w:m, h:m / AR } : { w:m * AR, h:m };
}
function fmt(x){ return (Math.round(x * 100) / 100).toLocaleString("en-AU"); }
function ceil(x){ return Math.ceil(x - 1e-9); }

function render(){
  const m = parseFloat($("#measure").value) || 0;
  const eff = parseFloat($("#eff").value) || 0;
  const d = dimsFrom(m);
  const total = d.w * d.h;
  $("#dims").textContent = m > 0
    ? (fmt(d.w) + " m wide " + "\u00d7 " + fmt(d.h) + " m tall") : "\u2014";

  let totCans = 0;
  const body = ROWS.map(r => {
    const wall = total * (r.pct / 100);
    const cans = (eff > 0 && m > 0) ? ceil(wall / eff) : 0;
    totCans += cans;
    return `<tr>
      <td><span class="swatch" style="background:${r.hex}"></span>${r.name}</td>
      <td>${Math.round(r.pct)}</td>
      <td>${m > 0 ? cans : "\u2014"}</td>
    </tr>`;
  }).join("");
  $("#rows").innerHTML = body;
  $("#tCans").textContent = m > 0 ? totCans : "\u2014";
}

function setMode(m){
  mode = m;
  $("#mode").querySelectorAll("button").forEach(b => b.classList.toggle("on", b.dataset.m === m));
  $("#mlabel").textContent = (m === "width" ? "width" : "height");
  render();
}
$("#mode").addEventListener("click", e => { if (e.target.dataset.m) setMode(e.target.dataset.m); });
$("#measure").addEventListener("input", render);
$("#eff").addEventListener("input", render);
render();
</script>
</body></html>
"""


_LOGO_URI_CACHE = None


def _logo_data_uri():
    """Base64 data-URI of the flashpointgui logo (shipped alongside this file),
    or '' if it's missing. Inlined so paintlist.html stays fully self-contained
    (opens anywhere, no sidecar image)."""
    global _LOGO_URI_CACHE
    if _LOGO_URI_CACHE is None:
        uri = ""
        try:
            p = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "flashpointgui.png")
            with open(p, "rb") as f:
                uri = "data:image/png;base64," + base64.b64encode(f.read()).decode("ascii")
        except Exception:
            uri = ""
        _LOGO_URI_CACHE = uri
    return _LOGO_URI_CACHE


def _logo_html():
    """The <img> tag for the logo banner in paintlist.html, or '' if no logo."""
    uri = _logo_data_uri()
    if not uri:
        return ""
    return ('<img src="%s" alt="flashpointgui" width="56" height="56" '
            'style="display:block;margin:0 0 12px;">' % uri)


def write_stats_html(path, image_w, image_h, rows, title="poster", palette="unknown"):
    ar = (image_w / image_h) if image_h else 1.0
    data = [{"name": n, "hex": hx, "pct": float(p)} for (n, hx, p) in rows]
    html = (STATS_HTML
            .replace("__TITLE__", title)
            .replace("__DIM__", f"{image_w} \u00d7 {image_h} px")
            .replace("__NCOLORS__", str(len(data)))
            .replace("__PALETTE__", palette)
            .replace("__AR__", repr(round(ar, 8)))
            .replace("__ROWS__", json.dumps(data))
            .replace("__LOGO__", _logo_html()))
    with open(path, "w", encoding="utf-8") as f:
        f.write(html)
    return path


# ==========================================================================
# Output writing (shared by CLI + GUI "save")
# ==========================================================================

def sanitize_name(name):
    return re.sub(r'[^A-Za-z0-9_-]', '_', name)


def write_outputs(state, outdir, border_color="#ff00ff", bg_color="#808080",
                  stroke_width=2.0):
    """Write master PNG, borders.png, layers/, paintlist.html from a PipelineState.
    Returns (master_path, borders_path, [layer dicts], stats_path).
    """
    border_rgb = _parse_color(border_color)
    bg_rgb = _parse_color(bg_color)
    W, H = state.W, state.H
    os.makedirs(outdir, exist_ok=True)
    layers_dir = os.path.join(outdir, "layers")
    os.makedirs(layers_dir, exist_ok=True)
    stem = getattr(state, "_stem", "image")

    master = render_master(state, state.palette_rgb)
    master_path = os.path.join(outdir, f"master_{stem}.png")
    Image.fromarray(master, mode="RGB").save(master_path)

    total_px = W * H
    if state.borders_overlay is not None:
        bimg = render_borders_rgba(state, border_rgb)
    else:
        bimg = np.zeros((H, W, 4), dtype=np.uint8)
    borders_path = os.path.join(outdir, "borders.png")
    Image.fromarray(bimg, mode="RGBA").save(borders_path)

    # per-color layers (bg -> fill -> border overlay), re-tinted to border_color
    ov = state.borders_overlay
    zero_ov = np.zeros((H, W, 4), dtype=np.float32)
    written = []
    collision = {}
    for cd in state.color_data:
        cname = cd["name"]
        base = sanitize_name(cname) or "color"
        if base not in collision:
            collision[base] = 0
            fname = base + ".png"
        else:
            n = collision[base]
            while True:
                n += 1
                cand = f"{base}_{n}.png"
                if cand not in collision:
                    break
            fname = cand
            collision[base] = n
        lay = render_layer_png(cd["mask"], np.array(cd["rgb"], dtype=np.uint8),
                               W, H, bg_rgb, ov if ov is not None else zero_ov)
        Image.fromarray(lay, mode="RGB").save(os.path.join(layers_dir, fname))
        written.append({
            "name": cname, "file": fname, "hex": cd["hex"], "rgb": cd["rgb"],
            "pct": 100.0 * int(cd["mask"].sum()) / total_px,
            "pixels": int(cd["mask"].sum()),
        })

    stats_path = os.path.join(outdir, "paintlist.html")
    write_stats_html(stats_path, W, H,
                     [(L["name"], L["hex"], L["pct"]) for L in written],
                     title=stem,
                     palette=getattr(state, "_palette_name", "unknown"))
    return master_path, borders_path, written, stats_path


def load_image_rgb(path):
    """Open any image path -> (H,W,3) uint8 RGB, alpha composited onto white."""
    img = Image.open(path)
    mode = img.mode
    if mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255, 255, 255, 255))
        bg.alpha_composite(img)
        img = bg
    img = img.convert("RGB")
    return np.asarray(img, dtype=np.uint8)
