import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# import local writer
from gcode import Segment

# ------------------ Centerline (skeleton) extraction ------------------
def extract_centerlines_lineart(
    I: np.ndarray,
    px_per_mm: float,
    blur_ksize: int = 3,            # 0/1 = off; 3/5 smooths noise
    binarize: str = "otsu",         # "otsu" or "adaptive"
    adaptive_block: int = 35,       # for adaptive
    adaptive_C: int = 5,            # for adaptive
    erode_px: int = 0,              # 0–2 to shrink thick strokes a touch
    simplify_eps_mm: float = 0.20,  # RDP tol in mm
    prune_spur_mm: float = 0.5,     # drop very short dangling pieces
) -> tuple[list[Segment], np.ndarray]:
    """
    Centerlines for black-on-white line art.
    Returns (centerline_segments_mm, binary_strokes_px)
    """
    Hpx, Wpx = I.shape
    img8 = (np.clip(I, 0, 1) * 255.0).astype(np.uint8)

    # 1) (Optional) denoise small grain
    if blur_ksize and blur_ksize > 1:
        k = blur_ksize | 1
        img8 = cv2.GaussianBlur(img8, (k, k), 0)
        cv2.imshow("Gaussian Blur", img8)
        cv2.waitKey(0)

    # 2) Binarize: strokes=255, background=0
    if binarize == "adaptive":
        bw = cv2.adaptiveThreshold(
            img8, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV,
            adaptive_block | 1, adaptive_C
        )
        cv2.imshow("Adaptive Threshold", bw)
        cv2.waitKey(0)
    else:
        _, bw = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        cv2.imshow("Threshold", bw)
        cv2.waitKey(0)

    # 3) (Optional) shrink very thick lines slightly before thinning
    if erode_px and erode_px > 0:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (erode_px*2+1, erode_px*2+1))
        bw = cv2.erode(bw, k)
        cv2.imshow("Erode Image", bw)
        cv2.waitKey(0)

    # 4) Thinning (skeletonization) => 1-px centerlines
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize((bw > 0).astype(bool)).astype(np.uint8) * 255
        cv2.imshow("Skeletonize Image", skel)
        cv2.waitKey(0)
    except Exception:
        try:
            from cv2.ximgproc import thinning, THINNING_ZHANGSUEN
            skel = thinning(bw, THINNING_ZHANGSUEN)
            cv2.imshow("ZHANGSUEN Skeletonize Image", skel)
            cv2.waitKey(0)
        except Exception:
            raise RuntimeError("Install scikit-image or opencv-contrib-python for thinning.")

    # 5) Vectorize skeleton -> polylines in px
    from skimage import measure
    contours = measure.find_contours(skel.astype(np.uint8), level=0.5)  # list of (row, col)

    # 6) RDP simplify in px + convert to mm (origin bottom-left)
    eps_px = simplify_eps_mm * px_per_mm

    def px_to_mm(pt):
        x_px, y_px = float(pt[0]), float(pt[1])
        x_mm = x_px / px_per_mm
        y_mm = (Hpx - 1 - y_px) / px_per_mm
        return (x_mm, y_mm)

    segs: list[Segment] = []
    for arr in contours:
        pts_px = np.stack([arr[:,1], arr[:,0]], axis=1).astype(np.float32)  # (x,y)
        if len(pts_px) < 2:
            continue
        c = pts_px.reshape(-1,1,2)
        c2 = cv2.approxPolyDP(c, epsilon=eps_px, closed=False)
        p2 = c2.reshape(-1, 2)
        if len(p2) < 2:
            continue
        poly_mm: Segment = [px_to_mm(p) for p in p2]
        segs.append(poly_mm)

    # 7) Prune tiny dangling spur segments (optional cleanup)
    if prune_spur_mm and prune_spur_mm > 0:
        keep: list[Segment] = []
        thr = prune_spur_mm
        for s in segs:
            if len(s) < 2:
                continue
            # length in mm
            L = 0.0
            for (x1,y1),(x2,y2) in zip(s, s[1:]):
                L += math.hypot(x2-x1, y2-y1)
            if L >= thr:
                keep.append(s)
        segs = keep

    # # --- To Plot ---
    # fig, ax = plt.subplots(figsize=(8, 5))

    # for pts in segs:
    #     x_vals, y_vals = zip(*pts)
    #     ax.plot(x_vals, y_vals)

    # plt.show()

    # cv2.imshow("Black and White Image", bw)
    # cv2.waitKey(0)

    return segs, bw  # bw returned so you can preview/check binarization