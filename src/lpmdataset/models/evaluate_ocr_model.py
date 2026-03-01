import numpy as np
import pandas as pd
import torch
from src.lpmdataset.representations import heatmap
from src.lpmdataset.modalities import mouse


TOP_K_BOXES = 80


# ------------------------------------------------------------
# Convert region IDs → predicted (x,y) coordinates
# ------------------------------------------------------------
def region_ids_to_coords(region_ids, centers):
    coords = [centers[r] for r in region_ids]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return np.array(xs), np.array(ys)


# ------------------------------------------------------------
# Predict a full region sequence autoregressively
# ------------------------------------------------------------
def predict_region_sequence(model, x0, centers, steps=300):
    model.eval()
    seq = x0.unsqueeze(0)   # shape (1, SEQ_LEN, 3*K)
    preds = []

    for _ in range(steps):
        with torch.no_grad():
            logits = model(seq)
            region_id = logits.argmax(dim=1).item()
            preds.append(region_id)

        # Build next input
        onehot = torch.zeros(1, 1, TOP_K_BOXES)
        onehot[0, 0, region_id] = 1

        geom = seq[:, -1, TOP_K_BOXES:]  # geometry stays constant
        geom = geom.unsqueeze(1)

        next_x = torch.cat([onehot, geom], dim=2)
        seq = torch.cat([seq[:, 1:], next_x], dim=1)

    return preds


# ------------------------------------------------------------
# Build predicted heatmap
# ------------------------------------------------------------
def build_pred_heatmap(xs, ys):
    df = pd.DataFrame({
        "timestamp": np.arange(len(xs)),
        "x": xs,
        "y": ys
    })
    hm = heatmap(df)
    hm.upsample()
    return hm


# ------------------------------------------------------------
# Build ground truth heatmap
# ------------------------------------------------------------
def build_gt_heatmap(gt_path):
    df = mouse.load_trace_data(gt_path)
    hm = heatmap(df)
    hm.upsample()
    return hm


# ------------------------------------------------------------
# Evaluate OCR model prediction using Wasserstein distance
# ------------------------------------------------------------
def evaluate_prediction(pred_region_ids, centers, gt_trace_path):
    xs, ys = region_ids_to_coords(pred_region_ids, centers)

    pred_hm = build_pred_heatmap(xs, ys)
    gt_hm = build_gt_heatmap(gt_trace_path)

    dist = pred_hm.distance_to(gt_hm)
    return dist, pred_hm, gt_hm


# ------------------------------------------------------------
# Full evaluation wrapper
# ------------------------------------------------------------
def evaluate_ocr_baseline(model, dataset, pairs, idx=0, steps=300):
    x, y, centers, ocr_path = dataset[idx]
    centers = centers.numpy()

    # Get ground truth mouse trace path
    gt_trace_path = pairs[idx][1]

    # Predict region sequence
    pred_region_ids = predict_region_sequence(model, x, centers, steps=steps)

    # Evaluate
    dist, pred_hm, gt_hm = evaluate_prediction(pred_region_ids, centers, gt_trace_path)

    print("Wasserstein distance:", dist)

    # Optional visualization
    pred_hm.show(title="Predicted Heatmap")
    gt_hm.show(title="Ground Truth Heatmap")

    return dist
