#!/usr/bin/env python3
"""
flashpoint.py — CLI for the flashpoint posterizer (v3, refactored).

Same command-line contract as the original:
  python flashpoint.py --input image.png --palette colors.txt --output outdir \
      [--min-area 50] [--num-colors 10] [--exact] \
      [--border-color #ff00ff] [--bg-color #808080]

All the algorithm lives in flashpoint_core.py; this file just parses args,
runs the pipeline, writes outputs, and prints the summary. The GUI
(flashpoint_gui.py) drives the same core with a different front-end.
"""

import argparse
import os
import sys
import time

import flashpoint_core as core


def build_parser():
    p = argparse.ArgumentParser(
        description="Palette-constrained posterizer: snap a PNG to a palette in CIELAB "
                    "and emit a master PNG plus per-color PNG layers (color fill with "
                    "the border line-work superimposed).")
    p.add_argument("--input", required=True, help="Source PNG (RGB or RGBA).")
    p.add_argument("--palette", required=True, help="Palette text file.")
    p.add_argument("--output", required=True, help="Output directory (created if missing).")
    p.add_argument("--min-area", type=int, default=50,
                   help="Merge regions smaller than this many pixels (0 disables).")
    p.add_argument("--num-colors", type=int, default=10,
                   help="Limit to the N dominant palette colors (default 10).")
    p.add_argument("--exact", action="store_true",
                   help="Skip dominant-color selection; map against the full palette.")
    p.add_argument("--border-color", default="#ff00ff",
                   help="Line color for the border line-work (default #ff00ff).")
    p.add_argument("--bg-color", default="#808080",
                   help="Background fill behind each color in the layer PNGs (default #808080).")
    return p


def main():
    args = build_parser().parse_args()
    t0 = time.time()

    print("[load] loading image ...", flush=True)
    try:
        rgb = core.load_image_rgb(args.input)
    except Exception as e:
        core.fail(f"Cannot open image {args.input!r}: {e}")
    W, H = rgb.shape[1], rgb.shape[0]
    if max(W, H) > 8000:
        print(f"[load] WARN: image is {W}x{H} (longest side > 8000 px); processing anyway.",
              file=sys.stderr, flush=True)
    print(f"[load] {args.input} -> {W}x{H} RGB.", flush=True)

    print("[quantize] loading palette ...", flush=True)
    palette = core.load_palette(args.palette)  # list of [name, (r,g,b)]
    stem = os.path.splitext(os.path.basename(args.input))[0]
    pal_name = os.path.basename(args.palette)

    state = core.run_pipeline(
        rgb, palette,
        num_colors=args.num_colors,
        exact=args.exact,
        min_area=args.min_area,
        border_color=args.border_color,
        bg_color=args.bg_color,
        stem=stem,
        palette_name=pal_name,
    )

    print(f"[write] writing outputs to {args.output} ...", flush=True)
    master_path, borders_path, layers, report_path = core.write_outputs(
        state, args.output,
        border_color=args.border_color,
        bg_color=args.bg_color,
    )

    dt = time.time() - t0
    print(flush=True)
    print(f"Summary ({dt:.2f}s):")
    print(f"  master:  {master_path}  ({W}x{H})")
    print(f"  borders: {borders_path}  ({len(state.polylines)} line runs)")
    for L in layers:
        print(f"  layer:   layers/{L['file']}  ({L['name']} {L['hex']}, "
              f"{L['pct']:.1f}% of image)")
    print(f"  report:  {report_path}")
    K = len(palette)
    print(f"  {len(layers)} layer(s) written (color fill + border line-work); "
          f"{K - len(layers)} palette color(s) had no surviving region.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
