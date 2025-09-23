import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon, box
import math
from typing import List, Tuple
import svgwrite
import cv2

# import local writer
from gcode import Segment

def desired_spacing(intensity: float, d_min: float, d_max: float, gamma: float) -> float:
# intensity I in [0,1], 0=black -> d_min; 1=white -> d_max
    return d_max - (d_max - d_min) * (1.0 - intensity) ** gamma

def hatch_layer(
I: np.ndarray,
canvas_w_mm: float,
canvas_h_mm: float,
px_per_mm: float,
angle_deg: float = 22.5,
d_min: float = 0.45,
d_max: float = 2.2,
gamma_tone: float = 1.0,
d_nom: float | None = None,
probe_step_mm: float = 0.35,
keep_alpha: float = 1.0,
mask: np.ndarray | None = None,
) -> List[Segment]:
    
    if d_nom is None:
        d_nom = 0.5 * (d_min + d_max)

    # Direction and normal
    th = math.radians(angle_deg)
    ux, uy = math.cos(th), math.sin(th)
    nx, ny = -math.sin(th), math.cos(th) # left-normal

    # World-rect polygon (mm): origin at (0,0) bottom-left
    rect = box(0.0, 0.0, canvas_w_mm, canvas_h_mm)

    # Determine how many lines to cover the bbox diagonal
    # Span across the rectangle's projection on normal axis
    # Compute bounds of projection for corners
    corners = [(0, 0), (canvas_w_mm, 0), (canvas_w_mm, canvas_h_mm), (0, canvas_h_mm)]
    proj_vals = [cx * nx + cy * ny for (cx, cy) in corners]
    proj_min, proj_max = min(proj_vals), max(proj_vals)
    span = proj_max - proj_min
    n_lines = int(math.ceil(span / d_nom)) + 3 # small padding

    segments: List[Segment] = []

    # Helper: sample image intensity at world (mm) coords
    Hpx, Wpx = I.shape

    def sample_I(x_mm: float, y_mm: float) -> float:
        # map mm->px, origin bottom-left -> image origin top-left
        cx = x_mm * px_per_mm
        cy = y_mm * px_per_mm
        px = np.clip(cx, 0, Wpx - 1 - 1e-6)
        py = np.clip((Hpx - 1) - cy, 0, Hpx - 1 - 1e-6)
        return float(I[int(py), int(px)])
    
    def sample_mask(x_mm: float, y_mm: float) -> bool:
        if mask is None:
            return True
        Hpx, Wpx = I.shape
        cx = x_mm * px_per_mm
        cy = y_mm * px_per_mm
        px = int(np.clip(cx, 0, Wpx - 1))
        py = int(np.clip((Hpx - 1) - cy, 0, Hpx - 1))
        return mask[py, px] > 0

    # For each parallel line index, compute anchor point at projection value
    for k in range(0, n_lines):
        proj = proj_min + k * d_nom
        # Build an infinite line via a long segment, then clip
        # Pick a point on this line: p0 satisfies dot(p0, n) = proj
        # Choose p0 = proj * n (on the normal), then two far points along +/- u
        p0x, p0y = proj * nx, proj * ny
        L = max(Wpx, Hpx) * 2.0 + span # long enough
        a = (p0x, p0y)
        b = (p0x + ux * L, p0y + uy * L)
        inf_line = LineString([a, b])
        clipped = rect.intersection(inf_line)
        if clipped.is_empty:
            continue
        # Handle MultiLine by iterating; but for a convex rect it should be LineString
        line_geoms = []
        if clipped.geom_type == "MultiLineString":
            line_geoms = list(clipped.geoms)
        else:
            line_geoms = [clipped]

        for geom in line_geoms:
            (x1, y1), (x2, y2) = list(geom.coords)
            seg_len = math.hypot(x2 - x1, y2 - y1)
            n_steps = max(2, int(math.ceil(seg_len / probe_step_mm)))
            # Build polyline pieces based on local gating
            run: Segment = []
            prev_keep = False
            for i in range(n_steps + 1):
                t = i / n_steps
                x = x1 + (x2 - x1) * t
                y = y1 + (y2 - y1) * t
                Ixy = sample_I(x, y)
                d_tar = desired_spacing(Ixy, d_min, d_max, gamma_tone)
                keep = (d_tar <= d_nom * keep_alpha) and sample_mask(x, y)

                if keep:
                    if not prev_keep:
                    # start new run
                        run = [(x, y)]
                    else:
                        run.append((x, y))
                if (not keep or i == n_steps) and prev_keep and run:
                    # close run
                    # simplify tiny runs
                    if len(run) >= 2:
                        segments.append(run)
                    run = []
                prev_keep = keep

    # --- Plot ---
    fig, ax = plt.subplots(figsize=(8, 5))
    # draw rectangle
    ax.plot([0, canvas_w_mm, canvas_w_mm, 0, 0],
            [0, 0, canvas_h_mm, canvas_h_mm, 0])

    for pts in segments:
        x_vals, y_vals = zip(*pts)
        ax.plot(x_vals, y_vals)

    plt.show()

    return segments

def export_preview_svg(path, polys, canvas_w_mm, canvas_h_mm):
    W = canvas_w_mm
    H = canvas_h_mm
    dwg = svgwrite.Drawing(path, size=(f"{W}mm", f"{H}mm"))
    dwg.add(dwg.rect(insert=(0,0), size=(f"{W}mm", f"{H}mm"), fill="white"))
    for poly in polys:
        pts = [(x, H-y) for (x,y) in poly]  # flip Y for image coords
        dwg.add(dwg.polyline(points=pts, stroke="black", fill="none", stroke_width=0.2))
    dwg.save()

# def extract_outlines(
#     I: np.ndarray,
#     px_per_mm: float,
#     canny_low: int = 80,
#     canny_high: int = 180,
#     simplify_eps_mm: float = 0.20,
# ) -> tuple[list[Segment], np.ndarray]:
#     """
#     Returns (outline_segments_mm, edge_map_px).
#     - I is grayscale [0,1] with shape (Hpx, Wpx)
#     - outlines are returned as list[Segment] in mm coords (origin bottom-left, like your hatch)
#     """
#     Hpx, Wpx = I.shape
#     img8 = (I * 255.0).astype(np.uint8)
#     edges = cv2.Canny(img8, canny_low, canny_high)
#     cv2.imshow("Canny Image", edges)
#     cv2.waitKey(0)

#     contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

#     # helper: px→mm with origin at bottom-left (consistent with your preview/export)
#     def px_to_mm(pt):
#         x_px, y_px = float(pt[0]), float(pt[1])
#         x_mm = x_px / px_per_mm
#         y_mm = (Hpx - 1 - y_px) / px_per_mm
#         return (x_mm, y_mm)

#     eps_px = simplify_eps_mm * px_per_mm
#     outlines: list[Segment] = []
#     for c in contours:
#         if len(c) < 2:
#             continue
#         # simplify with RDP in px domain (polyline)
#         c2 = cv2.approxPolyDP(c, epsilon=eps_px, closed=False)
#         pts_px = c2.reshape(-1, 2)
#         if len(pts_px) < 2:
#             continue
#         poly_mm: Segment = [px_to_mm(p) for p in pts_px]
#         outlines.append(poly_mm)

#     # --- Plot ---
#     fig, ax = plt.subplots(figsize=(8, 5))
#     # draw rectangle

#     for pts in outlines:
#         x_vals, y_vals = zip(*pts)
#         ax.plot(x_vals, y_vals)

#     plt.show()

#     return outlines, edges

#     # Draw contours on a copy of the original image
#     # img_contours = img8.copy()
#     # cv2.drawContours(img_contours, contours, -1, (0, 255, 0), 2) # Draw all contours in green with thickness 2

#     # cv2.imshow("Contour Image", img_contours)
#     # cv2.waitKey(0)


def build_shade_mask(
    I: np.ndarray,
    edge_map_px: np.ndarray,
    thresh_rel: float = 0.55,      # lower => more hatched area
    edge_dilate_px: int = 2,       # exclude a band around edges
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Returns uint8 mask same size as I: 255 where hatching is allowed, 0 elsewhere.
    """
    g = I.astype(np.float32)
    if blur_ksize > 1:
        k = (blur_ksize | 1)
        g = cv2.GaussianBlur(g, (k, k), 0)

    # Darker-than threshold
    mask = (g <= thresh_rel).astype(np.uint8) * 255
    cv2.imshow("Mask Image1", mask)
    cv2.waitKey(0)

    # 3) Ensure thin, binary edges
    edges = (edge_map_px > 0).astype(np.uint8) * 255
    cv2.imshow("Mask Image2", edges)
    cv2.waitKey(0)

    # 4) Distance transform: distance (in pixels) to nearest edge
    inv = cv2.bitwise_not(edges)                 # edges=0, background=255
    dist = cv2.distanceTransform(inv, cv2.DIST_L2, 5)  # float32, pixels

    gap_px = max(1, int(round(edge_dilate_px)))  # min 1 px
    keep_far = (dist >= gap_px).astype(np.uint8) * 255

    # 5) Keep only shaded pixels that are also far from edges
    mask = cv2.bitwise_and(mask, keep_far)
    cv2.imshow("Mask Image3", mask)
    cv2.waitKey(0)

    # # Clean small specks
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,
    #                         cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3)))
    
    # cv2.imshow("Mask Image4", mask)
    # cv2.waitKey(0)
    return mask