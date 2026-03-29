"""Crop figure bounding boxes from a slide PNG and save them to figures/.

Usage:
    python crop_figures.py

"""

from pathlib import Path
import numpy as np
from PIL import Image
from tqdm import tqdm

from lpmdataset.data_models import Folder, Presentation, Slide, SPEAKER_RESOLUTIONS

# --- Configuration -----------------------------------------------------------
FOLDER   = Folder.ML_1
OUT_DIR  = Path("figures")
# -----------------------------------------------------------------------------

def detect_pillarbox(
    img: Image.Image,
    threshold: int = 10,
    center_fraction: float = 0.5,
) -> tuple[int, int]:
    """Return (left_bar, right_bar) pixel widths of black pillarbox columns.

    Only the central *center_fraction* of rows are examined so that
    picture-in-picture overlays near the corners do not affect the result.
    Scans column-by-column from each edge inward until a column whose mean
    brightness exceeds *threshold*.
    """
    arr = np.array(img.convert("L"))  # grayscale, shape (h, w)
    h, w = arr.shape
    y0 = int(h * (1 - center_fraction) / 2)
    y1 = int(h * (1 + center_fraction) / 2)
    strip = arr[y0:y1, :]  # central horizontal strip

    left_bar = 0
    for x in range(w):
        if strip[:, x].mean() > threshold:
            left_bar = x
            break

    right_bar = 0
    for x in range(w - 1, -1, -1):
        if strip[:, x].mean() > threshold:
            right_bar = w - 1 - x
            break

    return left_bar, right_bar


def calibrate_pillarbox(presentation: Presentation, threshold: int = 10) -> tuple[int, int]:
    """Compute consensus pillarbox widths over all slides in *presentation*.

    Loads each slide PNG, runs :func:`detect_pillarbox` on the central strip,
    and returns the median (left, right) bar widths across all slides.
    """
    lefts, rights = [], []
    slides = list(presentation.slides())
    print(f"Calibrating pillarbox over {len(slides)} slides...", flush=True)
    for slide in slides:
        img = Image.open(slide.png)
        l, r = detect_pillarbox(img, threshold=threshold)
        lefts.append(l)
        rights.append(r)
    left_bar  = int(np.median(lefts))
    right_bar = int(np.median(rights))
    print(f"  per-slide left  bars: min={min(lefts)}  max={max(lefts)}  median={left_bar}")
    print(f"  per-slide right bars: min={min(rights)}  max={max(rights)}  median={right_bar}")
    return left_bar, right_bar


def crop_figures(slide: Slide):
    presentation = slide.presentation
    figures = slide.figures
    if figures.empty:
        print("No figures annotated for this slide.")
        return

    # Open the slide PNG and get its actual dimensions.
    img = Image.open(slide.png)
    png_w, png_h = img.size

    # Native annotation resolution (the resolution the bbox coords were recorded in).
    native_res = SPEAKER_RESOLUTIONS[str(FOLDER)]
    native_w, native_h = native_res.width, native_res.height

    # --- Debug info ----------------------------------------------------------
    print(f"Slide PNG        : {slide.png}")
    print(f"PNG size         : {png_w} x {png_h}")
    print(f"Native resolution: {native_w} x {native_h}  ({native_res})")
    print(f"Scale factors    : sx={png_w/native_w:.4f}  sy={png_h/native_h:.4f}")
    print()

    # Calibrate over all slides to get a robust pillarbox estimate.
    left_bar, right_bar = calibrate_pillarbox(presentation)
    content_w = png_w - left_bar - right_bar
    print(f"Pillarbox (calib): left={left_bar}px  right={right_bar}px  "
          f"=> content area {content_w} x {png_h}")
    if left_bar or right_bar:
        sx_content = content_w / native_w
        print(f"Scale (no pillar): sx={sx_content:.4f}  sy={png_h/native_h:.4f}")
    print()
    print(f"Figures on slide : {len(figures)}")
    print(figures.to_string())
    print("-" * 60)
    # -------------------------------------------------------------------------

    # Naive rescaling: scale directly from native resolution to PNG size.
    # (Pillarbox offset commented out until calibration is confirmed.)
    sx = png_w / native_w
    sy = png_h / native_h

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for i, row in figures.iterrows():
        # Scale bbox from native resolution to PNG coordinates.
        # left_bar offset commented out until pillarbox calibration is confirmed.
        left   = int(row["left"] * sx)  # + left_bar
        top    = int(row["top"] * sy)
        width  = int(row["width"] * sx)
        height = int(row["height"] * sy)

        # Clamp to image bounds.
        left   = max(0, min(left,          png_w - 1))
        top    = max(0, min(top,           png_h - 1))
        right  = max(0, min(left + width,  png_w))
        bottom = max(0, min(top  + height, png_h))

        crop = img.crop((left, top, right, bottom))

        label = str(row["label"]).replace(" ", "_").lower()
        out_path = OUT_DIR / f"{presentation.yt_id}_slide_{SLIDE_NO:03d}_fig{i:02d}_{label}.png"
        crop.save(out_path)
        print(f"Saved {out_path}  [{label}]  ({right-left}x{bottom-top}px)")

    print(f"\nDone — {len(figures)} figure(s) saved to '{OUT_DIR}/'")


if __name__ == "__main__":
    for presentation in data_models.iter_presentations(FOLDER):
        for slide in tqdm(
            presentation.iter_slides(),
            total=sum(1 for _ in presentation.iter_slides()),
        ):
            crop_figures(slide)
    main()
