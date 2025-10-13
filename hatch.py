import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box
import math
from typing import List, Tuple
import svgwrite
import cv2

# import local writer
from gcode import Segment

def desired_spacing(intensity: float, d_min: float, d_max: float, gamma: float) -> float:
# intensity I in [0,1], 0=black -> d_min; 1=white -> d_max
    return d_max - (d_max - d_min) * (1.0 - intensity) ** gamma

def hatch_layer_medium(
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
                    if len(run) >= 2:
                        # keep only endpoints if short; otherwise simplify
                        if len(run) <= 4:
                            comp = [run[0], run[-1]]
                        else:
                            comp = simplify_polyline(run, eps=0.03)  # ~0.03 mm tolerance
                        # Drop very short runs
                        if len(comp) >= 2:
                            x1,y1 = comp[0]; x2,y2 = comp[-1]
                            if math.hypot(x2-x1, y2-y1) >= 0.10:  # min 0.1 mm segment
                                segments.append(comp)
                    run = []
                
                prev_keep = keep

    # # --- To Plot ---
    # fig, ax = plt.subplots(figsize=(8, 5))
    # # draw rectangle
    # ax.plot([0, canvas_w_mm, canvas_w_mm, 0, 0],
    #         [0, 0, canvas_h_mm, canvas_h_mm, 0])

    # for pts in segments:
    #     x_vals, y_vals = zip(*pts)
    #     ax.plot(x_vals, y_vals)

    # plt.show()

    return segments

def simplify_polyline(pts, eps=0.03):  # eps in mm; ~pen width fraction
    import math
    if len(pts) <= 2:
        return pts
    # RDP-lite using OpenCV if available; else manual
    import numpy as np, cv2
    arr = np.array(pts, dtype=np.float32).reshape(-1,1,2)
    # Convert eps (mm) directly; we’re in mm coords already
    simp = cv2.approxPolyDP(arr, epsilon=eps, closed=False).reshape(-1,2).tolist()
    # Merge nearly collinear triples (angle ~180°)
    out = [simp[0]]
    for i in range(1, len(simp)-1):
        x0,y0 = out[-1]; x1,y1 = simp[i]; x2,y2 = simp[i+1]
        v1x,v1y = x1-x0, y1-y0
        v2x,v2y = x2-x1, y2-y1
        # area (cross) small & both segments non-trivial => remove middle
        cross = abs(v1x*v2y - v1y*v2x)
        if cross < 1e-3:  # tune: ~0.001 mm²
            continue
        out.append((x1,y1))
    out.append(simp[-1])
    return out

def export_preview_svg(path, polys, canvas_w_mm, canvas_h_mm):
    W = canvas_w_mm
    H = canvas_h_mm
    dwg = svgwrite.Drawing(path, size=(f"{W}mm", f"{H}mm"))
    dwg.add(dwg.rect(insert=(0,0), size=(f"{W}mm", f"{H}mm"), fill="white"))
    for poly in polys:
        pts = [(x, H-y) for (x,y) in poly]  # flip Y for image coords
        dwg.add(dwg.polyline(points=pts, stroke="black", fill="none", stroke_width=0.2))
    dwg.save()

def hatch_layer_high(
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

    # # --- To Plot ---
    # fig, ax = plt.subplots(figsize=(8, 5))
    # # draw rectangle
    # ax.plot([0, canvas_w_mm, canvas_w_mm, 0, 0],
    #         [0, 0, canvas_h_mm, canvas_h_mm, 0])

    # for pts in segments:
    #     x_vals, y_vals = zip(*pts)
    #     ax.plot(x_vals, y_vals)

    # plt.show()

    return segments