from typing import Iterable, List, Tuple


Point = Tuple[float, float]
Segment = List[Point] # polyline




def write_gcode(
    segments: List[Segment],
    outfile: str,
    xy_feed: float = 100.0,
    z_feed: float = 600.0, # unused for servo, kept for compatibility
    pen_down_s: int = 900,
    pen_up_s: int = 0,
    dwell_after_toggle: float = 0.70,
    travel_lift_between: bool = True, # always true for servo toggles
    header_comment: str = "Pencil Shaded Render",
    origin_mm: Tuple[float, float] = (0.0, 0.0),
):
    """
    segments: list of polylines (each is list of (x,y) in mm) in desired plotting order
    Emits absolute (G90) metric (G21) G-code. Z axis is not used; servo toggles via M3 S{value}.
    """
    with open(outfile, "w", encoding="utf-8") as f:
        w = f.write
        w(f"( {header_comment} )\n")
        w("G21\n") # mm
        w("G90\n") # absolute
        # ensure pen up
        w(f"M3 S{pen_up_s}\n")
        w(f"G4 P{dwell_after_toggle:.3f}\n")
        w("G92 X0 Y0 Z0\n")


        for poly in segments:
            if not poly:
                continue
            # travel to first point (pen up)
            x0, y0 = poly[0]
            x0 += origin_mm[0]
            y0 += origin_mm[1]
            w(f"G0 X{round(x0,3)} Y{round(y0,3)}\n")
            # pen down
            w(f"M3 S{pen_down_s}\n")
            w(f"G4 P{dwell_after_toggle:.3f}\n")
            # draw
            for (x, y) in poly[1:]:
                x += origin_mm[0]
                y += origin_mm[1]
                w(f"G1 X{round(x,3)} Y{round(y,3)} F{int(xy_feed)}\n")
            # pen up
            w(f"M3 S{pen_up_s}\n")
            w(f"G4 P{dwell_after_toggle:.3f}\n")


        # end
        w(f"M3 S{pen_up_s}\n")
        w(f"G4 P{dwell_after_toggle:.3f}\n")
        w("G0 X0 Y0\n")
        w("M2\n")

def chain_touching_segments(segments, tol=0.02):
    """Join consecutive polylines whose endpoints touch (within tol, mm)."""
    def same(p, q): return abs(p[0]-q[0]) <= tol and abs(p[1]-q[1]) <= tol
    out = []
    cur = []
    for seg in segments:
        if not seg or len(seg) < 2:
            continue
        if not cur:
            cur = seg[:]
        else:
            if same(cur[-1], seg[0]):           # end of cur meets start of seg
                cur.extend(seg[1:])
            else:
                out.append(cur)
                cur = seg[:]
    if cur:
        out.append(cur)
    return out
