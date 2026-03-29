import logging
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from lpmdataset import data_models
from lpmdataset.representations.heatmap import HeatMap


log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# =========================================================
# CONFIG
# =========================================================

SCREEN_W  = 1200
SCREEN_H  = 900
PLOTS_DIR = "results/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

REQUIRED_COLS = ["pred_x", "pred_y", "gold_x", "gold_y"]


# =========================================================
# LOAD FILES — recursive, handles all CSV filenames
# =========================================================

def load_result_files(results_root: str) -> list[str]:
    """
    Recursively collect every CSV under results_root.
    Deduplicates by (folder, slide_no) when both slide_NNN.csv
    and slide_NNN_trace.csv exist — prefers the shorter stem.
    """
    all_files: list[str] = []
    for root, _, fs in os.walk(results_root):
        for f in fs:
            if f.endswith(".csv"):
                all_files.append(os.path.join(root, f))

    seen: dict[tuple[str, int], str] = {}
    ungrouped: list[str] = []

    for path in all_files:
        folder = os.path.dirname(path)
        stem   = os.path.splitext(os.path.basename(path))[0]
        parts  = stem.split("_")
        try:
            slide_no = int(parts[1])
        except (IndexError, ValueError):
            ungrouped.append(path)
            continue

        key = (folder, slide_no)
        if key not in seen:
            seen[key] = path
        else:
            current_stem = os.path.splitext(os.path.basename(seen[key]))[0]
            if len(stem) < len(current_stem):
                seen[key] = path

    return list(seen.values()) + ungrouped


def _read_csv_safe(path: str) -> pd.DataFrame | None:
    """Read a prediction CSV. Returns None on error or missing columns."""
    try:
        df = pd.read_csv(path)
    except Exception as e:
        log.warning(f"Cannot read {path}: {e}")
        return None

    if not all(c in df.columns for c in REQUIRED_COLS):
        log.warning(f"Missing required columns in {path} — skipping")
        return None

    if len(df) < 1:
        return None

    df = df.astype({c: float for c in REQUIRED_COLS})

    #  AUTO-DETECT NORMALIZED DATA (CLIP)
    if df["pred_x"].max() <= 1.5 and df["gold_x"].max() <= 1.5:
        # assume normalized → convert to pixel space
        df["pred_x"] *= SCREEN_W
        df["pred_y"] *= SCREEN_H
        df["gold_x"] *= SCREEN_W
        df["gold_y"] *= SCREEN_H

    return df


# =========================================================
# TRAJECTORY METRICS
# =========================================================

def compute_metrics(df: pd.DataFrame):
    N = min(len(df["pred_x"]), len(df["gold_x"]))
    if N < 2:
        return None

    pred = df[["pred_x", "pred_y"]].values[:N].copy().astype(np.float32)
    gold = df[["gold_x", "gold_y"]].values[:N].copy().astype(np.float32)

    pred[:, 0] /= SCREEN_W;  pred[:, 1] /= SCREEN_H
    gold[:, 0] /= SCREEN_W;  gold[:, 1] /= SCREEN_H

    dist = np.linalg.norm(pred - gold, axis=1)
    ad   = float(np.mean(dist))
    rmse = float(np.sqrt(np.mean(dist ** 2)))

    bins = 20
    ph, _, _ = np.histogram2d(pred[:, 0], pred[:, 1], bins=bins,
                               range=[[0, 1], [0, 1]])
    gh, _, _ = np.histogram2d(gold[:, 0], gold[:, 1], bins=bins,
                               range=[[0, 1], [0, 1]])
    ph /= ph.sum() + 1e-8
    gh /= gh.sum() + 1e-8
    iou = float(np.minimum(ph, gh).sum() / (np.maximum(ph, gh).sum() + 1e-8))

    return ad, rmse, iou


# =========================================================
# REGION PROXY METRICS
# Purely CSV-based — no Slide/OCR reconstruction needed.
# Uses screen quadrants (2x2 grid) as a proxy for region.
# =========================================================

def compute_region_proxy_accuracy(files: list[str]) -> dict:
    """
    A prediction is 'correct' when pred and gold fall in the same
    screen quadrant. Requires only the four CSV columns — never
    touches the original dataset files.
    """
    all_match = 0
    all_total = 0
    per_file  = []

    for f in tqdm(files, total=len(files)):
        df = _read_csv_safe(f)
        if df is None:
            continue

        pred_quad = (
            (df["pred_x"] // (SCREEN_W / 2)).astype(int).astype(str)
            + "_"
            + (df["pred_y"] // (SCREEN_H / 2)).astype(int).astype(str)
        )
        gold_quad = (
            (df["gold_x"] // (SCREEN_W / 2)).astype(int).astype(str)
            + "_"
            + (df["gold_y"] // (SCREEN_H / 2)).astype(int).astype(str)
        )

        matches = int((pred_quad == gold_quad).sum())
        total   = len(df)

        per_file.append({"file": f, "region_accuracy": matches / total})
        all_match += matches
        all_total += total

    overall = all_match / (all_total + 1e-8)

    log.info("\n--- Region Proxy Metrics ---")
    log.info(f"Files evaluated: {len(per_file)}")
    log.info(f"Region Proxy Accuracy: {overall:.4f}")

    return {"per_file": per_file, "overall_accuracy": overall}

# =========================================================
# REGION TYPE METRICS
# Purely CSV-based — no Slide/OCR reconstruction needed.
# Uses screen quadrants (2x2 grid) as a proxy for region.
# =========================================================

def compute_region_type_accuracy(files: list[str]) -> dict:
    """Compute the rate at which predicted and gold points fall in the
    same region type (inside an OCR bounding box vs. outside).

    Returns a dict with per-file accuracies and overall statistics.
    """
    all_match = 0
    all_total = 0
    per_file  = []

    for f in files:
        df = _read_csv_safe(f)
        if df is None:
            continue

        required = ["pred_x","pred_y","gold_x","gold_y"]
        if not all(c in df.columns for c in required):
            print(f"Skipping (bad format): {f}")
            continue

        if len(df) < 1:
            continue

        try:
            slide = data_models.Slide.from_prediction_file(f)
        except Exception as exc:
            print(f"Skipping {f}: {exc}")
            continue

        pred_regions = df.apply(
            lambda row: slide.get_region_for_point(
                float(row['pred_x']), float(row['pred_y'])
            ), axis=1,
        )
        gold_regions = df.apply(
            lambda row: slide.get_region_for_point(
                float(row['gold_x']), float(row['gold_y'])
            ), axis=1,
        )

        matches = (pred_regions == gold_regions).sum()
        total = len(df)
        acc = matches / total
        per_file.append({"file": f, "region_type_accuracy": acc})
        all_match += matches
        all_total += total

    overall = all_match / (all_total + 1e-8)

    log.info("\n--- Region Type Metrics ---")
    log.info(f"Files evaluated: {len(per_file)}")
    log.info(f"Region Type Accuracy: {overall:.4f}")

    return {"per_file": per_file, "overall_accuracy": overall}
# =========================================================
# MOVEMENT METRICS
# =========================================================

def compute_movement_metrics(files: list[str], threshold: float = 1.0) -> dict:
    all_gold, all_pred = [], []

    for f in files:
        df = _read_csv_safe(f)
        if df is None or len(df) < 2:
            continue

        gold_dx = np.diff(df["gold_x"].values, prepend=df["gold_x"].iloc[0])
        gold_dy = np.diff(df["gold_y"].values, prepend=df["gold_y"].iloc[0])
        pred_dx = np.diff(df["pred_x"].values, prepend=df["pred_x"].iloc[0])
        pred_dy = np.diff(df["pred_y"].values, prepend=df["pred_y"].iloc[0])

        all_gold.append((np.sqrt(gold_dx ** 2 + gold_dy ** 2) > threshold).astype(int))
        all_pred.append((np.sqrt(pred_dx ** 2 + pred_dy ** 2) > threshold).astype(int))

    if not all_gold:
        log.warning("No valid movement data found.")
        return {"TP": 0, "TN": 0, "FP": 0, "FN": 0,
                "precision": 0.0, "recall": 0.0, "f1": 0.0, "accuracy": 0.0}

    all_gold = np.concatenate(all_gold)
    all_pred = np.concatenate(all_pred)

    TP = int(np.sum((all_pred == 1) & (all_gold == 1)))
    TN = int(np.sum((all_pred == 0) & (all_gold == 0)))
    FP = int(np.sum((all_pred == 1) & (all_gold == 0)))
    FN = int(np.sum((all_pred == 0) & (all_gold == 1)))

    precision = TP / (TP + FP + 1e-8)
    recall    = TP / (TP + FN + 1e-8)
    f1        = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy  = (TP + TN) / (TP + TN + FP + FN + 1e-8)

    log.info("\n--- Movement Metrics ---")
    log.info(f"TP:{TP}  TN:{TN}  FP:{FP}  FN:{FN}")
    log.info(f"P:{precision:.3f}  R:{recall:.3f}  F1:{f1:.3f}  Acc:{accuracy:.3f}")

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "precision": precision, "recall": recall,
        "f1": f1, "accuracy": accuracy,
    }


# =========================================================
# PLOTS
# =========================================================

def _save(fig: plt.Figure, name: str):
    path = os.path.join(PLOTS_DIR, name.replace(" ", "_") + ".png")
    fig.savefig(path, bbox_inches="tight", dpi=150)
    log.info(f"Saved → {path}")
    plt.close(fig)


def plot_confusion_matrix(metrics: dict, model_name: str):
    cm = np.array([
        [metrics["TN"], metrics["FP"]],
        [metrics["FN"], metrics["TP"]],
    ])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["No Move", "Move"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["No Move", "Move"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color="white", fontsize=12)
    ax.set_title(f"{model_name} — Confusion Matrix")
    ax.set_xlabel("Predicted");  ax.set_ylabel("Actual")
    fig.colorbar(im, ax=ax)
    _save(fig, f"{model_name}_confusion_matrix")


def plot_metric_bars(metrics: dict, model_name: str):
    names = ["Precision", "Recall", "F1", "Accuracy"]
    vals  = [metrics[k] for k in ["precision", "recall", "f1", "accuracy"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    bars = ax.bar(names, vals,
                  color=["#4C72B0", "#55A868", "#C44E52", "#8172B2"])
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                f"{v:.2f}", ha="center", va="bottom")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"{model_name} — Movement Metrics")
    ax.set_ylabel("Score")
    _save(fig, f"{model_name}_movement_metrics")


def plot_combined_heatmap(root: str, model_name: str, num_slides: int = 5):
    files = load_result_files(root)
    if not files:
        return

    chosen = np.random.choice(files, min(num_slides, len(files)), replace=False)
    gx, gy, px, py = [], [], [], []

    for f in chosen:
        df = _read_csv_safe(f)
        if df is None:
            continue
        gx.extend(df["gold_x"] / SCREEN_W);  gy.extend(df["gold_y"] / SCREEN_H)
        px.extend(df["pred_x"] / SCREEN_W);  py.extend(df["pred_y"] / SCREEN_H)

    if not gx:
        return

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    h1 = axes[0].hist2d(gx, gy, bins=40, range=[[0, 1], [0, 1]])
    axes[0].set_title(f"{model_name} — Gold")
    fig.colorbar(h1[3], ax=axes[0])
    h2 = axes[1].hist2d(px, py, bins=40, range=[[0, 1], [0, 1]])
    axes[1].set_title(f"{model_name} — Predicted")
    fig.colorbar(h2[3], ax=axes[1])
    fig.suptitle(f"{model_name} — Spatial Distribution")
    fig.tight_layout()
    _save(fig, f"{model_name}_spatial_distribution")


def spatial_metric_and_visualization(
    root: str,
    model_name: str,
    num_slides: int = 5,
    bins: int = 32,
) -> list[float]:
    files = load_result_files(root)
    if not files:
        log.warning("spatial_metric_and_visualization: no files found")
        return []

    chosen = np.random.choice(files, min(num_slides, len(files)), replace=False)

    # Pre-filter to only slides with valid data
    valid = [(f, df) for f in chosen
             if (df := _read_csv_safe(f)) is not None]

    if not valid:
        return []

    scores = []
    fig, axes = plt.subplots(len(valid), 2, figsize=(8, 4 * len(valid)))
    if len(valid) == 1:
        axes = np.array([axes])

    for i, (f, df) in enumerate(valid):
        try:
            def make_heatmap(x_col, y_col):
                hm = HeatMap(pd.DataFrame({
                    "x": df[x_col], "y": df[y_col],
                    "timestamp": np.arange(len(df)),
                }))
                hm.upsample()
                h, _, _ = hm.low_res(bins=bins)
                return h / (h.sum() + 1e-8)

            hist_g = make_heatmap("gold_x", "gold_y")
            hist_p = make_heatmap("pred_x", "pred_y")
        except Exception as e:
            log.warning(f"HeatMap failed for {f}: {e}")
            continue

        score = float(np.minimum(hist_g, hist_p).sum())
        scores.append(score)

        axes[i, 0].imshow(hist_g.T, origin="lower", aspect="auto")
        axes[i, 0].set_title(f"Slide {i + 1} — Gold")
        axes[i, 0].axis("off")

        axes[i, 1].imshow(hist_p.T, origin="lower", aspect="auto")
        axes[i, 1].set_title(f"Slide {i + 1} — Pred  Score:{score:.3f}")
        axes[i, 1].axis("off")

    fig.suptitle(f"{model_name} — Spatial Metric + Heatmaps", fontsize=14)
    fig.tight_layout()
    _save(fig, f"{model_name}_spatial_heatmaps")

    if scores:
        log.info(f"Spatial Histogram Intersection — Mean: {np.mean(scores):.4f}")

    return scores


# =========================================================
# EVALUATION
# =========================================================

def evaluate_model(root: str, name: str) -> dict:
    log.info(f"\n===== {name} =====")

    files = load_result_files(root)
    if not files:
        log.warning(f"No result files found under {root}")
        return {}

    log.info(f"Found {len(files)} result files")

    ads, rmses, ious = [], [], []
    for f in files:
        df = _read_csv_safe(f)
        if df is None:
            continue
        res = compute_metrics(df)
        if res is None:
            continue
        ad, rmse, iou = res
        ads.append(ad);  rmses.append(rmse);  ious.append(iou)

    if ads:
        log.info(
            f"Trajectory — AD:{np.mean(ads):.4f}  "
            f"RMSE:{np.mean(rmses):.4f}  IoU:{np.mean(ious):.4f}"
        )
    else:
        log.warning("No trajectory metrics computed.")

    movement_metrics = compute_movement_metrics(files)
    region_proxy_metrics = compute_region_proxy_accuracy(files)
    region_type_metrics = compute_region_type_accuracy(files)

    plot_confusion_matrix(movement_metrics, name)
    plot_metric_bars(movement_metrics, name)
    plot_combined_heatmap(root, name)

    return {
        "trajectory": {
            "AD":   float(np.mean(ads))   if ads else 0.0,
            "RMSE": float(np.mean(rmses)) if rmses else 0.0,
            "IoU":  float(np.mean(ious))  if ious else 0.0,
        },
        "movement": movement_metrics,
        "region_proxy": region_proxy_metrics,
        "region_type": region_type_metrics,
    }


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    BASE = os.environ["RESULTS_DIR"]

    MODELS = {
        "Clip": os.path.join(BASE, "clip"),
        "Mouse Only": os.path.join(BASE, "mouse_only"),
        "OCR_ASR_Semanctic": os.path.join(BASE, "ocr_asr_semantic_org"),
        "OCR Only":   os.path.join(BASE, "ocr_only"),
        "OCR + ASR":  os.path.join(BASE, "ocr_asr"),
        "ViLT": os.path.join(BASE, "vilt_b32_finetuned_vqa"),
        "ViLT hierarchical": os.path.join(BASE, "vilt_b32_finetuned_vqa_modified_hierarchical_patches"),
        "LayoutLMv3": os.path.join(BASE, "layoutlmv3_base_finetuned_rvlcdip"),
    }

    for name, root in MODELS.items():
        if not os.path.exists(root):
            log.warning(f"Skipping {name} — path not found: {root}")
            continue
        evaluate_model(root, name)
        spatial_metric_and_visualization(root, name)
