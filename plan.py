import math
import numpy as np
from typing import List

# import local writer
from gcode import Segment

def stitch_polylines(polys, snap=0.02, angle_tol_deg=12.0, min_attach_len=0.05):
    """
    Merge polylines whose endpoints coincide within `snap` and whose directions
    are angle-compatible within `angle_tol_deg`. Returns a new list of longer polylines.
    polys: list[list[(x,y)]], coordinates in mm, each with len >= 2
    """
    import math
    from collections import defaultdict

    def key(p):  # grid-snap to make near-coincident points identical
        return (round(p[0]/snap), round(p[1]/snap))

    def length2(a, b):
        dx, dy = b[0]-a[0], b[1]-a[1]
        return dx*dx + dy*dy

    def dir_vec_last(poly):
        # direction of the last segment of the chain
        x1, y1 = poly[-2]; x2, y2 = poly[-1]
        return (x2-x1, y2-y1)

    def dir_vec_forward(poly):
        # forward direction at the start of a candidate
        x1, y1 = poly[0]; x2, y2 = poly[1]
        return (x2-x1, y2-y1)

    def dir_vec_backward(poly):
        # backward direction at the end of a candidate (if we reverse it)
        x1, y1 = poly[-1]; x2, y2 = poly[-2]
        return (x2-x1, y2-y1)

    def angle_deg(u, v):
        ux, uy = u; vx, vy = v
        nu = math.hypot(ux, uy); nv = math.hypot(vx, vy)
        if nu == 0 or nv == 0: return 180.0
        c = max(-1.0, min(1.0, (ux*vx + uy*vy)/(nu*nv)))
        return abs(math.degrees(math.acos(c)))

    n = len(polys)
    used = [False]*n

    # Build an endpoint index for both start and end
    # key -> list of (poly_idx, end_type) where end_type in {"start","end"}
    touch = defaultdict(list)
    for i, p in enumerate(polys):
        if len(p) < 2: 
            used[i] = True
            continue
        touch[key(p[0])].append((i, "start"))
        touch[key(p[-1])].append((i, "end"))

    out = []
    for i, p in enumerate(polys):
        if used[i] or len(p) < 2:
            continue

        used[i] = True
        chain = p[:]  # seed the chain with this polyline

        # grow forward from chain end
        extended = True
        while extended:
            extended = False
            end_key = key(chain[-1])

            # candidates that touch our end_key
            candidates = touch.get(end_key, [])
            if not candidates: 
                break

            best = None  # (angle, j, how) where how in {"forward","reverse"}
            base_dir = dir_vec_last(chain)

            for (j, end_type) in candidates:
                if used[j] or j == i:
                    continue
                q = polys[j]
                # we can attach q either as forward (if its start touches us)
                # or as reverse (if its end touches us)
                if end_type == "start":
                    # end(chain) -> start(q)
                    if len(q) >= 2 and length2(q[0], q[1]) >= (min_attach_len**2):
                        a = angle_deg(base_dir, dir_vec_forward(q))
                        if a <= angle_tol_deg and (best is None or a < best[0]):
                            best = (a, j, "forward")
                else:  # end_type == "end"
                    # end(chain) -> end(q)  (we'll reverse q)
                    if len(q) >= 2 and length2(q[-1], q[-2]) >= (min_attach_len**2):
                        a = angle_deg(base_dir, dir_vec_backward(q))
                        if a <= angle_tol_deg and (best is None or a < best[0]):
                            best = (a, j, "reverse")

            if best is not None:
                _, j, how = best
                used[j] = True
                q = polys[j]
                if how == "forward":
                    # avoid duplicating the junction point q[0] == chain[-1] (within snap)
                    chain.extend(q[1:])
                else:
                    rq = list(reversed(q))
                    chain.extend(rq[1:])
                extended = True

        out.append(chain)

    return out


def order_segments_greedy(polys: List[Segment]) -> List[Segment]:
    """Greedy nearest-end ordering with segment reversal allowed. O(N^2) but simple.
    Suitable until you hit many tens of thousands of segments.
    """
    if not polys:
        return []
    remaining = [(p[0], p[-1], p) for p in polys] # (start, end, poly)
    used = [False] * len(remaining)
    ordered: List[Segment] = []


    # pick the longest as a good starting chain
    lens = [math.hypot(px[1][0]-px[0][0], px[1][1]-px[0][1]) for px in [(p[0], p[-1]) for _, _, p in remaining]]
    cur_idx = int(np.argmax(lens))
    used[cur_idx] = True
    _, cur_end, cur_poly = remaining[cur_idx]
    ordered.append(cur_poly)


    for _ in range(len(remaining) - 1):
        best = None
        best_dist = 1e18
        best_rev = False
        best_idx = -1
        for j, (s, e, p) in enumerate(remaining):
            if used[j]:
                continue
            # distance from current end to this poly start/end
            d1 = (cur_end[0]-s[0])**2 + (cur_end[1]-s[1])**2
            d2 = (cur_end[0]-e[0])**2 + (cur_end[1]-e[1])**2
            if d1 < best_dist:
                best_dist, best, best_rev, best_idx = d1, p, False, j
            if d2 < best_dist:
                best_dist, best, best_rev, best_idx = d2, p[::-1], True, j
        used[best_idx] = True
        ordered.append(best if not best_rev else best)
        cur_end = ordered[-1][-1]
    return ordered