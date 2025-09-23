import math
import numpy as np
from typing import List

# import local writer
from gcode import Segment

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