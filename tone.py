import cv2
import numpy as np
from typing import Tuple

def load_tone(path: str, canvas_w_mm: float, canvas_h_mm: float, px_per_mm: float,
            clahe: bool = True, gamma: float = 1.0):
    # Import Image as Gray Scale Image

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    
    # Normalize pixels in range [0,1]
    img = img.astype(np.float32) / 255.0

    # Change image resolution according to canvas width and height
    Wpx = int(round(canvas_w_mm * px_per_mm))
    Hpx = int(round(canvas_h_mm * px_per_mm))
    img = cv2.resize(img, (Wpx, Hpx), interpolation=cv2.INTER_AREA)

    # Local Contrast using CLAHE
    if clahe:
        c = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = c.apply((img * 255.0).astype(np.uint8)).astype(np.float32) / 255.0

    if abs(gamma - 1.0) > 1e-6:
        img = np.clip(img, 1e-6, 1.0) ** (1.0 / gamma)

    return img

def detect_solid_blobs(
    gray01: np.ndarray,        # grayscale [0..1], image coords (y-down)
    fg_mask: np.ndarray,       # foreground approx, 0/255
    min_area_px: int = 20,     # tune for your ppm
    max_area_px: int = 2000,   # prevent grabbing big patches
    min_solidity: float = 0.90,# filled not ring-like
    circularity_hint: Tuple[float,float] = (0.5, 1.3),  # optional
) -> np.ndarray:               # 0/255 mask
    # threshold darker-than-local background *within* foreground
    vals = gray01[fg_mask > 0]
    if len(vals) == 0:
        return np.zeros_like(fg_mask)
    t = np.percentile(vals, 25)         # lower quartile works nicely for line art
    dark = (gray01 <= t).astype("uint8") * 255
    cand = cv2.bitwise_and(dark, fg_mask)
    cv2.imshow("CAND Image", cand)
    cv2.waitKey(0)

    # close tiny gaps so the pupil is solid
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, k, iterations=1)
    cv2.imshow("CAND Image (After filing gaps)", cand)
    cv2.waitKey(0)

    num, lab, stats, cent = cv2.connectedComponentsWithStats(cand, connectivity=8)
    out = np.zeros_like(fg_mask)
    for i in range(1, num):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area_px <= area <= max_area_px): 
            continue
        # compute solidity and circularity
        comp = (lab == i).astype("uint8") * 255
        cv2.imshow(f"Comp {i}", comp)
        cv2.waitKey(0)
        cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: 
            continue
        cnt = max(cnts, key=cv2.contourArea)
        area_c = cv2.contourArea(cnt)
        per   = cv2.arcLength(cnt, True) + 1e-6
        # solidity = area / convex_hull_area
        hull = cv2.convexHull(cnt)
        # sol  = area_c / (cv2.contourArea(hull) + 1e-6)
        # circ = (per*per) / (4*np.pi*area_c + 1e-6)  # =1 for perfect circle
        # if sol >= min_solidity and (circularity_hint[0] <= circ <= circularity_hint[1]):
        out[lab == i] = 255
    return out