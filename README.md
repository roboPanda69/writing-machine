# Image‑to‑G‑code (Outline + Selective Hatching) for CNC/XY Plotter

A lightweight Python pipeline that turns images into clean plotter strokes:

* **Centerline/Outline layer first** (single‑stroke line art)
* **Selective hatching** only in darker regions (with an adjustable “gap from edges” so shading doesn’t touch contours)
* **Greedy path ordering** to reduce pen‑up travel
* **GRBL‑safe G‑code** for servo pens (`M03 S0` Pen Down and `M03 S900` Pen Up)
* Optional **SVG preview** in mm

This repo is meant for CNCjs/GRBL‑style XY plotters (e.g., writing machines, pen plotters, DIY CNC).

---

## Project structure

```
image2gcode/
  ├─ main.py          # CLI: image -> toolpaths -> G-code (draw order: outlines/centerlines, then hatch)
  ├─ gcode.py         # GRBL-safe G-code writer (servo pen via M03 Sxxx + dwell)
  ├─ tone.py          # For Image Pre-Processing 
  ├─ hatch.py         # For creating hatch around dark areas
  ├─ plan.py          # For Greedy path ordering to reduce pen‑up travel
  ├─ outline.py       # For creating outline around the diagram
  └─ README.md
```

> **Note**: If you keep everything in `main.py + gcode.py`, that’s fine—the README still applies.

---

## Installation

**Python**: 3.10–3.12 recommended

```bash
pip install numpy opencv-python shapely svgwrite scikit-image
# optional (alternative thinning backend):
pip install opencv-contrib-python
```

---

## Core idea / pipeline (Work in Progress)

1. **Load & scale** the image to match your physical canvas (mm → px via `px_per_mm`).
2. **Centerlines / Outlines**:

   * For **line art** (black on white): *binarize → (optional) erode → skeletonize → vectorize → simplify*.
   * For **photo edges**: *Canny → contour simplify* (produces outlines, not centerlines).
3. **Shading mask** (for selective hatching): threshold dark regions and keep only pixels **≥ gap** away from edges/centerlines using a **distance transform**.
4. **Hatch generation**: lay parallel lines at chosen angles; sample the tone; keep runs only where the local tone demands; clip to mask.
5. **Path planning**: greedy nearest‑end ordering (with reversal allowed).
6. **G‑code**: absolute metric (`G21 G90`), servo pen control (`M03 Sdown/Sup` + `G4 P…`), feedrates in mm/min.

ASCII overview:

```
 image → scale →  [centerlines / outlines]
                    │
                    ├──► distance gap (mm) ┐
                    ▼                      │
          shading threshold (dark regions) ├──► hatch mask → hatch lines → order → G-code
                                           │
                                      keep far from edges
```

---

## Usage (CLI)

**Typical run (outline/centerline + selective hatch):**

```bash
python main.py input.jpg output.gcode \
  --canvas_w 60 --canvas_h 90 --ppm 5 \
  --angles 22.5 67.5 --dmin 0.45 --dmax 2.2 --gamma 1.0 \
  --probe 0.35 --alpha 1.0 \
  --feed 1500 --downS 720 --upS 0 --dwell 0.10 \
  --preview_svg out.svg
```

**Outline‑only / Centerline‑only preview:**

* Temporarily skip the hatch loop or set an absurdly large hatch spacing.

**Dense cross‑hatch look:**

```bash
--angles 0 45 90 135 --dmin 0.35 --dmax 1.8 --probe 0.25
```

> The CLI flags above match the original `main.py` contract. The *centerline* step is inside the code; see the next section to toggle it.

---

## Centerlines vs Outlines

You have both approaches available. Pick one per project (or mix if you want):

### A) Centerlines for line art (recommended for sketches/ink)

* **Why**: yields a single stroke down the middle of black lines—no “double edges”.
* **Pipeline**: `blur → binarize (Otsu/Adaptive) → erode (optional) → skeletonize (thinning) → vectorize → RDP simplify`.
* **Function**: `extract_centerlines_lineart(I, ...) -> list[Segment]`.

**Key knobs**

* `binarize`: `"otsu"` (even lighting) or `"adaptive"` (uneven paper/lighting)
* `erode_px`: 0–2 to shrink thick strokes so the skeleton is stable
* `simplify_eps_mm`: 0.15–0.30 mm (detail ↔ speed)
* `prune_spur_mm`: 0.5–1.0 mm to drop tiny dangling hairs

### B) Outlines from photo edges

* **Why**: traces object boundaries; good when you want visible contours, then hatch shadows.
* **Pipeline**: `Canny → contour simplify (RDP) → mm`.
* **Function**: `extract_outlines(I, ...) -> list[Segment]`.

**Key knobs**

* `canny_low/high`: 70–100 / 150–200 are solid starts
* `simplify_eps_mm`: 0.15–0.30 mm

> **Switching mode**: In `main.py`, call either `extract_centerlines_lineart(...)` or `extract_outlines(...)` before the hatching step. Both return `List[Segment]` in **mm** (origin at bottom‑left), so the rest of the pipeline is unchanged.

---

## Selective hatching & the edge gap (distance transform)

To avoid shading right on top of your nice lines, we build a **hatch mask** that:

1. selects **dark** regions (by threshold), and
2. **keeps only pixels at least `edge_gap_mm` away** from any edge/centerline using a **distance transform**.

**Why distance transform?** Dilation often over‑kills in dense drawings (mask goes to black). Distance‑based gating is stable and resolution‑proof.

**Good starting values**

* `thresh_rel = 0.55` (lower → hatch fewer regions; higher → hatch more midtones)
* `edge_gap_mm = 0.3–0.6` (≈ one thin pen width)

> If you’re using the pixel version, translate: `gap_px = round(edge_gap_mm * px_per_mm)`.

---

## Hatching controls

* `--angles`: one or more angles in degrees (e.g., `22.5 67.5`).
* `--dmin / --dmax`: min/max line spacing (mm). Darker areas target `dmin`.
* `--gamma`: tone curve shaping (≥1 darkens; try 1.0–1.6).
* `--probe`: sampling step along each hatch line (mm). Smaller -> smoother but slower.
* `--alpha`: gating slack (keep runs where desired\_spacing ≤ `d_nom * alpha`).

**Speed vs Quality quick tips**

* Faster: increase `dmin/dmax` and `probe`; reduce number of angles; raise `simplify_eps_mm`.
* Darker/richer: decrease `dmin`; add a second angle (crosshatch); lower `thresh_rel` so more area is shaded.

---

## G‑code output (servo pen)

`gcode.py::write_gcode(...)` produces GRBL‑friendly code:

* `G21` (mm) + `G90` (absolute), `G92 X0 Y0 Z0`
* **Pen up/down via servo**:

  * Pen **up**: `M03 S{pen_up_s}` → `G4 P{dwell}`
  * Pen **down**: `M03 S{pen_down_s}` → `G4 P{dwell}`
* Travel uses `G0`; drawing uses `G1 F{xy_feed}`

**Typical values**

* `pen_down_s = 720`, `pen_up_s = 0`
* `dwell_after_toggle = 0.10–1.00 s` (depends on servo speed)

> If you prefer Z moves instead of a servo, the writer can be adapted to use `G1 Z...`—the rest of the pipeline stays the same.

---

## CNCjs / GRBL setup checklist

1. **Units**

   * Add `G21` near the top of every file (the writer already does this).
   * In GRBL: `$$` then ensure `$13=0` (report **mm**). In CNCjs UI, set **Millimeters**.
2. **Steps/mm calibration**

   * Command a 50.00 mm move; measure actual travel.
   * Update `$100` (X) / `$101` (Y) accordingly.
3. **Servo mapping**

   * Map `M03 Sxxx` to your pen up/down angles in your GRBL build / servo controller. Tune `S` values and `G4` dwell until motion is reliable.

If the viewer looks 25.4× off, it’s a **units mismatch** (mm vs inch).

---

## Examples / recipes

**1) Outline + light cross‑hatch (60×90 mm)**

```bash
python main.py img.jpg out.nc --canvas_w 60 --canvas_h 90 --ppm 5 \
  --angles 22.5 67.5 --dmin 0.45 --dmax 2.2 --probe 0.35 --feed 1500 \
  --downS 720 --upS 0 --dwell 0.50 --preview_svg out.svg
```

**2) Centerlines‑only (ink sketch look)**

* In `main.py`, call `extract_centerlines_lineart(...)` and **skip** the hatch loop.

**3) Dense shading (posterized shadows)**

```bash
--angles 0 45 90 135 --dmin 0.35 --dmax 1.8 --gamma 1.3 --probe 0.25
```

---

## Troubleshooting

**Centerlines scale doesn’t match hatch (e.g., 140×135 mm vs 30×35 mm)**

* Ensure both stages use the **same** `I, px_per_mm` from `load_tone(...)`, or make centerline conversion **canvas‑aware** (`px→mm` via `Wpx/Hpx` and `canvas_w/h`).

**Mask turns all‑black after adding edge band**

* Use **distance transform** gating (keep pixels with `distance ≥ edge_gap_mm`). Dilation can over‑grow in dense drawings.

**Mask turns black after “clean small specks”**

* Replace `MORPH_OPEN` with **area‑based despeckling** (connected components). Or make the open conditional/gentler.

**Double lines along edges**

* Use the **centerline (line‑art)** pipeline (binarize → thin) instead of Canny edges.

**Too slow / heavy G‑code**

* Increase `simplify_eps_mm`; increase `probe`; fewer angles; larger `dmin/dmax`.

**Hatch touches outlines**

* Increase `edge_gap_mm` (e.g., 0.4→0.6) or reduce `simplify_eps_mm` so outlines are truer; confirm mask uses **distance transform**.

---

## Contributing

Issues and PRs are welcome—especially for:

* A CLI flag to toggle centerlines vs outlines
* Optional Z‑axis pen control in `gcode.py`
* Spur pruning for skeletons and short‑segment merging
* Advanced TSP‑style path ordering

---

## License

This Read me was made with ChatGPT. Would review and update this in part by part manner.
