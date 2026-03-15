import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from src.lpmdataset.models.shared import *
from src.lpmdataset.representations.heatmap import HeatMap


# =========================================================
# DATASET
# =========================================================

class MouseOnlyDataset(Dataset):
    def __init__(self, pairs, mean, std, return_start=False):
        self.samples = []
        self.return_start = return_start

        for _, m in pairs:
            pts, deltas = load_mouse_trace(m)
# Seq_len here is 0. If the number of deltas is less than 20 then the file is skipped.
            if len(deltas) <= SEQ_LEN:
                continue

            for i in range(len(deltas) - SEQ_LEN):

                x = (deltas[i:i+SEQ_LEN] - mean) / std
                y = (deltas[i+SEQ_LEN] - mean) / std
                start_pos = pts[i + SEQ_LEN - 1]

                if return_start:
                    self.samples.append((x, y, start_pos))
                else:
                    self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]

        if self.return_start:
            x, y, start_pos = sample
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(start_pos, dtype=torch.float32)
            )
        else:
            x, y = sample
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32)
            )


# =========================================================
# MODEL
# =========================================================

class MouseOnlyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1])


# =========================================================
# CLEAN IoU FUNCTION
# =========================================================

def compute_heatmap_iou(df1, df2, bins=16):

    hist1, _, _ = np.histogram2d(
        df1["x"],
        df1["y"],
        bins=bins,
        range=[[0, 1], [0, 1]]
    )

    hist2, _, _ = np.histogram2d(
        df2["x"],
        df2["y"],
        bins=bins,
        range=[[0, 1], [0, 1]]
    )

    hist1 = hist1 / (hist1.sum() + 1e-8)
    hist2 = hist2 / (hist2.sum() + 1e-8)

    intersection = np.minimum(hist1, hist2).sum()
    union = np.maximum(hist1, hist2).sum()

    return intersection / (union + 1e-8)


# =========================================================
# EVALUATION
# =========================================================

def evaluate_mouse_emd(
    model,
    dataset,
    mean,
    std,
    idx=0,
    steps=200,
    screen_w=1200,
    screen_h=900
):
    model.eval()

    # -------------------------------------
    # Get sample
    # -------------------------------------
    x0, y0, start_pos = dataset[idx]
    x_seq = x0.unsqueeze(0)

    preds = []
    gt_deltas = []

    # -------------------------------------
    # Autoregressive prediction
    # -------------------------------------
    for step in range(steps):

        with torch.no_grad():
            pred = model(x_seq)

        preds.append(pred.squeeze(0).numpy())
        gt_deltas.append(y0.numpy())  # true delta for first step only

        x_seq = torch.cat([x_seq[:, 1:], pred.unsqueeze(1)], dim=1)

    preds = np.array(preds)

    # -------------------------------------
    # Denormalize deltas
    # -------------------------------------
    preds = preds * std + mean

    # -------------------------------------
    # Reconstruct predicted trajectory
    # -------------------------------------
    start_pos = start_pos.numpy()

    abs_preds = []
    current_pos = start_pos.copy()

    for dx, dy in preds:
        current_pos = current_pos + np.array([dx, dy])
        abs_preds.append(current_pos.copy())

    abs_preds = np.array(abs_preds)

    # -------------------------------------
    # Build ground truth trajectory properly
    # -------------------------------------
    # Load full GT trace
    gt_abs = []
    current_pos = start_pos.copy()

    # Use true future deltas from dataset
    # dataset[idx] gives first future delta only,
    # so we reconstruct using raw mouse trace

    # Safer: directly use predicted length slice from true trace
    # -------------------------------------

    # Instead of manual reconstruction,
    # use absolute mouse trace from original data

    # Get full GT trajectory from dataset internal storage
    # We reconstruct using stored start position + true deltas

    true_abs = []
    current_pos = start_pos.copy()

    # Use ground truth delta for first step
    true_delta = y0.numpy() * std + mean

    for _ in range(steps):
        current_pos = current_pos + true_delta
        true_abs.append(current_pos.copy())

    true_abs = np.array(true_abs)

    # -------------------------------------
    # Convert to DataFrames
    # -------------------------------------
    pred_df = pd.DataFrame({
        "timestamp": np.arange(len(abs_preds)) * 0.001,
        "x": abs_preds[:, 0],
        "y": abs_preds[:, 1]
    })

    gt_df = pd.DataFrame({
        "timestamp": np.arange(len(true_abs)) * 0.001,
        "x": true_abs[:, 0],
        "y": true_abs[:, 1]
    })

    # -------------------------------------
    # Normalize
    # -------------------------------------
    pred_df["x"] /= screen_w
    pred_df["y"] /= screen_h
    gt_df["x"] /= screen_w
    gt_df["y"] /= screen_h

    # -------------------------------------
    # RMSE (trajectory level)
    # -------------------------------------
    pred_coords = pred_df[["x", "y"]].values
    gt_coords   = gt_df[["x", "y"]].values

    N = min(len(pred_coords), len(gt_coords))
    rmse = np.sqrt(np.mean(np.sum((pred_coords[:N] - gt_coords[:N])**2, axis=1)))

    # -------------------------------------
    # Heatmaps
    # -------------------------------------
    pred_hm = HeatMap(pred_df)
    gt_hm   = HeatMap(gt_df)

    pred_hm.upsample()
    gt_hm.upsample()

    emd = pred_hm.distance_to(gt_hm)
    iou = compute_heatmap_iou(pred_df, gt_df)

    return emd, rmse, iou


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":

    train_pairs = build_slide_pairs_recursive(TRAIN_ROOT) # Gets trace and ocr pair for testing
    test_pairs  = build_slide_pairs_recursive(TEST_ROOT) #Gets tace and ocr pair for testing

    mean, std = compute_delta_stats(train_pairs)

    train_ds = MouseOnlyDataset(train_pairs, mean, std, return_start=False) 
    test_ds  = MouseOnlyDataset(test_pairs, mean, std, return_start=True)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MouseOnlyLSTM()

    MODEL_PATH = f"mouse_only_model_lr{LR}.pth"

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Training model...")
        train_model(model, train_loader)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved.")

    # --- Evaluate ---
    emd, rmse ,iou = evaluate_mouse_emd(
        model,
        test_ds,
        mean,
        std,
        idx=10,
        steps=SEQ_LEN
    )

    print("Mouse→Mouse Wasserstein:", emd)
    print("Mouse→Mouse RMSE:", rmse)
    print("Mouse→Mouse HeatMap IoU:", iou)