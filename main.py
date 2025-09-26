import cv2
from typing import List
import matplotlib.pyplot as plt
import argparse
import numpy as np

# import local writer
from gcode import write_gcode, Segment
from tone import load_tone, detect_solid_blobs
from hatch import hatch_layer, export_preview_svg
from outline import extract_centerlines_lineart
from plan import order_segments_greedy

def main():
    ap = argparse.ArgumentParser(description="Shaded image → hatch → G-code (servo)") 
    ap.add_argument("image", help="input image path (grayscale preferred)") 
    ap.add_argument("outfile", help="output G-code path, e.g., output.nc") 
    ap.add_argument("--canvas_w", type=float, default=180.0, help="canvas width mm") 
    ap.add_argument("--canvas_h", type=float, default=260.0, help="canvas height mm") 
    ap.add_argument("--ppm", type=float, default=5.0, help="pixels per mm for tone field") 
    ap.add_argument("--angles", type=float, nargs="*", default=[22.5, 67.5], help="hatch angles deg") 
    ap.add_argument("--dmin", type=float, default=0.45, help="minimum line spacing in mm (dark)") 
    ap.add_argument("--dmax", type=float, default=2.2, help="maximum line spacing in mm (light)") 
    ap.add_argument("--gamma", type=float, default=1.0, help="tone curve exponent (>=1 darkens)") 
    ap.add_argument("--probe", type=float, default=0.35, help="probe step along line (mm)") 
    ap.add_argument("--alpha", type=float, default=1.0, help="keep when desired<=d_nom*alpha") 
    ap.add_argument("--feed", type=float, default=1500.0, help="XY drawing feed (mm/min)") 
    ap.add_argument("--downS", type=int, default=900, help="servo PWM for pen down (M3 S..)") 
    ap.add_argument("--upS", type=int, default=0, help="servo PWM for pen up (M3 S..)") 
    ap.add_argument("--dwell", type=float, default=0.10, help="dwell after servo toggle (sec)") 
    ap.add_argument("--preview_svg", default=None, help="optional SVG preview path") 
    ap.add_argument("--render_scale", type=float, default=1.0, help="compute hatches on a larger virtual canvas, then downscale coords for output")
    ap.add_argument("--outline_buf_mm", type=float, default=0.30, help="Do not hatch within this distance from outlines")
    ap.add_argument("--shade_thresh", type=float, default=None, help="Optional [0..1] tone cutoff to restrict hatching to darker regions")
    args = ap.parse_args()

    # after parsing args:
    virt_w = args.canvas_w * args.render_scale
    virt_h = args.canvas_h * args.render_scale
    virt_ppm = args.ppm * args.render_scale
    
    img = load_tone(args.image, virt_w, virt_h, virt_ppm, False, args.gamma)
    cv2.imshow("Tone Image",img)
    cv2.waitKey(0)

    centerlines, bw = extract_centerlines_lineart(
        img, virt_ppm,
        blur_ksize=0,
        binarize="otsu",          # or "adaptive" for uneven lighting
        erode_px=3,               # try 0→2 depending on stroke thickness
        simplify_eps_mm=0.02,
        prune_spur_mm=0.05,
    )

    # 1) Foreground proxy for blob search:
    # Dilate outline mask so thin strokes define a 'filled' bird region.
    buf_px = max(1, int(round(args.outline_buf_mm * args.ppm)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*buf_px+1, 2*buf_px+1))
    fg_mask = cv2.dilate(bw, k)
    cv2.imshow("Dilate Image", fg_mask)
    cv2.waitKey(0)

    # 2) Solid blobs (eye, tiny darks)
    # 'img' should be grayscale [0..1] in image coords (y-down).
    blobs = detect_solid_blobs(img, fg_mask,
                            min_area_px=int(100),   # ~0.5 mm^2
                            max_area_px=int(100000),   # ~25  mm^2
                            min_solidity=0.9)
    
    cv2.imshow("Blob Image", blobs)
    cv2.waitKey(0)

    # 3) Optional: “darker tone” mask for areas like feet shadows
    # Use a stricter percentile to avoid hatching grey paper.
    vals = img[fg_mask > 0]
    t_dark = np.percentile(vals, 20)  # darkest 20% inside bird
    darker = (img <= t_dark).astype("uint8") * 255
    cv2.imshow("Darker Mask", darker)
    cv2.waitKey(0)

    # 4) Shade mask = blobs (must-fill) ∪ darker (soft shading)
    shade_mask_raw = cv2.bitwise_or(blobs, darker)
    cv2.imshow("Shade Mask Raw", shade_mask_raw)
    cv2.waitKey(0)

    # 5) Don’t hatch over outlines → carve a moat
    outline_block = cv2.dilate(bw, k)                   # reuse same buffer kernel
    shade_mask = cv2.bitwise_and(shade_mask_raw, outline_block)
    cv2.imshow("Shade Mask", shade_mask)
    cv2.waitKey(0)

    # # # ---- build shading mask (image coords, top-left origin) ----
    # # buf_px = max(1, int(round(args.outline_buf_mm * virt_ppm)))
    # # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*buf_px+1, 2*buf_px+1))
    # # cv2.imshow("Kernel Image", kernel)
    # # cv2.waitKey(0)

    # # # bw is strokes=255, background=0 (from outline.py)
    # # outline_block = cv2.dilate(bw, kernel)
    # # cv2.imshow("Outline Image", outline_block)
    # # cv2.waitKey(0)

    # # # start with "keep everywhere"
    # # shade_mask = (255 - outline_block)

    # # # optionally restrict to darker tones (e.g., hatch only where img <= 0.7)
    # # if args.shade_thresh is not None:
    # #     tone_mask = ( (img <= float(args.shade_thresh)).astype("uint8") * 255 )
    # #     shade_mask = cv2.bitwise_and(shade_mask, tone_mask)
    # #     cv2.imshow("Shade Mask", shade_mask)
    # #     cv2.waitKey(0)

    # Build layers 
    hatched: List[Segment] = [] 
    for ang in args.angles: 
        layer = hatch_layer( img, virt_w, virt_h, virt_ppm, angle_deg=ang, 
                            d_min=args.dmin, d_max=args.dmax, 
                            gamma_tone=args.gamma, d_nom=None, 
                            probe_step_mm=args.probe, keep_alpha=args.alpha,
                            mask=shade_mask) 
        hatched.extend(layer)

    # Order segments 
    outlines_ord = order_segments_greedy(centerlines)
    hatch_ord    = order_segments_greedy(hatched)
    combined_ord = outlines_ord + hatch_ord
    scale = 1.0 / args.render_scale
    ordered = [[(x*scale, y*scale) for (x,y) in poly] for poly in combined_ord]
    
    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    # draw rectangle
    ax.plot([0, args.canvas_w, args.canvas_w, 0, 0],
            [0, 0, args.canvas_h, args.canvas_h, 0])

    for pts in ordered:
        x_vals, y_vals = zip(*pts)
        ax.plot(x_vals, y_vals)

    plt.show()
    
    # Optional preview  
    try: 
        export_preview_svg(args.preview_svg, ordered, args.canvas_w, args.canvas_h) 
    except Exception as e: 
        print("[warn] SVG preview failed:", e)

    # Write G-code (servo) 
    write_gcode( 
        ordered, 
        outfile=args.outfile, 
        xy_feed=args.feed, 
        pen_down_s=args.downS, 
        pen_up_s=args.upS, 
        dwell_after_toggle=args.dwell, 
        header_comment=f"Shaded hatch from {args.image}", ) 
    print(f"✔ G-code saved to {args.outfile} |  outlines: {len(outlines_ord)}  hatch segments: {len(hatch_ord)}")

if __name__ == "__main__":
    main()