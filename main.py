import cv2
from typing import List
import argparse

# import local writer
from gcode import write_gcode, Segment
from tone import load_tone
from hatch import hatch_layer, export_preview_svg, extract_centerlines_lineart, build_shade_mask
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
    ap.add_argument("--render_scale", type=float, default=1.0,
                help="compute hatches on a larger virtual canvas, then downscale coords for output")
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
        blur_ksize=3,
        binarize="otsu",          # or "adaptive" for uneven lighting
        erode_px=1,               # try 0→2 depending on stroke thickness
        simplify_eps_mm=0.02,
        prune_spur_mm=0.05,
    )

    shade_mask = build_shade_mask(
            img, bw, thresh_rel=0.5, edge_dilate_px=1, blur_ksize=2
        )

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
    
    # Optional preview  
    try: 
        export_preview_svg(args.preview_svg, ordered, args.canvas_w, args.canvas_h) 
    except Exception as e: 
        print("[warn] SVG preview failed:", e)

    # Write G-code (servo) 
    write_gcode( 
        outlines_ord, 
        outfile=args.outfile, 
        xy_feed=args.feed, 
        pen_down_s=args.downS, 
        pen_up_s=args.upS, 
        dwell_after_toggle=args.dwell, 
        header_comment=f"Shaded hatch from {args.image}", ) 
    print(f"✔ G-code saved to {args.outfile} |  outlines: {len(outlines_ord)}  hatch segments: {len(hatch_ord)}")

if __name__ == "__main__":
    main()