import cv2
import numpy as np

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