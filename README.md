# flashpointgui

[github.com/saxononer/flashpointgui](https://github.com/saxononer/flashpointgui)

Linux Python GUI for posterizing an image into a limited, named palette and
tracing its color boundaries into clean, simplified SVG polylines (and PNG
layers).

It is a refactor of `flashpoint.py` (099-Flashpoint2) split into three layers
so the algorithm is testable and a GUI can sit on top of it:

| File | Role |
|------|------|
| `flashpoint_core.py` | The engine. `run_pipeline()` + the cheap `render_*` / `composite_over_photo` / `save_outputs` functions. Pure numpy/scipy/PIL — no UI. |
| `flashpoint_gui.py`  | The tkinter app. Live palette editing + camera-photo blend. |
| `flashpoint.py`      | Thin CLI wrapper — same command line as the original. |

## Install

```bash
python3 -m venv .venv
.venv/bin/pip install numpy scipy pillow
```

(Any Python 3.9+ with those three deps. tkinter is in the stdlib.)

## CLI

```bash
python flashpoint.py --input photo.png --palette colors.txt --output out/
```

`colors.txt` = one `name=HEX` per line, top-down by brightness. Optional flags:
`--exact` (skip k-means pre-clustering), `--num-colors N` (k-means count),
`--min-area N` (region cleanup), `--border-color HEX`, `--bg-color HEX`.

Outputs (mirrors the original): `master_<name>.png`, `borders_<name>.png`,
`stats.html`, and `layers/<hex>_<name>.png` (one per color, transparent).

## GUI

```bash
python flashpoint_gui.py [image.png] [palette.txt]
```

- **Load image / Load palette / Load both** — pick files (or pass as args).
- **Palette list** — each row is a color; select one to edit it.
- **Live recolor** — change the HEX (or hit Pick color) and the preview
  repaints *instantly*: the pipeline is cached, only the render re-runs.
- **Re-run pipeline** — forces a full recompute (after changing image, min-area,
  exact, or reordering the palette).
- **Camera / photo blend** — Load photo, pick a blend mode (normal, multiply,
  screen, overlay, soft-light, hard-light, color-dodge, color-burn, darken,
  lighten), set opacity, toggle "blend over photo". Exports the composite.
- **Show borders** — overlay the traced region contours on the preview.
- **Save all** — writes master / borders / stats.html / layers, same as the CLI.

## The two-tier model

The pipeline (Lab convert → nearest-color → region cleanup → contour trace)
depends only on the palette's *relative* positions, not its exact RGB. So:

- **Recolor** = cheap (repaint existing regions). This is the "edit on the fly"
  path — instant.
- **Re-run pipeline** = expensive (nearest-color + region grow + contour trace).
  Needed only when the image or palette *membership/order* changes.

## Known quirk: Tk 9.0 preview under a virtual display

`tk.PhotoImage(file=...)` is intermittently flaky under Xvfb + Tk 9.0
(`image ... does not exist`), per-process and unrelated to timing. On a real
display it works on the first try. The GUI ships a safety net: if the file
path flakes, it permanently falls back to updating one persistent image via
`put()` (slower but deterministic). No code change needed — it self-heals.
