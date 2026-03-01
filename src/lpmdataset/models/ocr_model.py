from typing import Counter
import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.lpmdataset.models.shared import *
from src.lpmdataset.models.shared import evaluate_region
import matplotlib.pyplot as plt
import random
import pandas as pd
from src.lpmdataset.modalities import mouse
from src.lpmdataset.representations import heatmap
from src.lpmdataset.representations.heatmap import HeatMap




# =========================================================
# CONFIG
# =========================================================
TOP_K_BOXES = 80


# =========================================================
# OCR BOX SELECTION (Area × Confidence)
# =========================================================

def load_top_ocr_boxes(path, K=TOP_K_BOXES):

    boxes = []

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]

        if "conf" not in header:
            return None

        li = header.index("left")
        ti = header.index("top")
        wi = header.index("width")
        hi = header.index("height")
        ci = header.index("conf")

        for row in reader:
            try:
                l = float(row[li])
                t = float(row[ti])
                w = float(row[wi])
                h = float(row[hi])
                conf = float(row[ci])
            except:
                continue

            # --- Remove bad OCR ---
            if conf <= 0:
                continue

            # --- Remove tiny boxes ---
            if w < 8 or h < 8:
                continue

            area = w * h
            if area < 100:   # tiny region threshold
                continue

            # --- Score = area × confidence ---
            score = area * (conf / 100.0)
            boxes.append((score, (l, t, w, h)))

    if len(boxes) == 0:
        return None

    boxes.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in boxes]


# =========================================================
# BUILD CORNERS + REGION CENTERS
# =========================================================
def build_regions(boxes):

    corners = []
    corner_to_box = []
    centers = []

    for box_id, (l, t, w, h) in enumerate(boxes):

        pts = [
            (l, t),
            (l + w, t),
            (l, t + h),
            (l + w, t + h)
        ]

        for p in pts:
            corners.append(p)
            corner_to_box.append(box_id)

        centers.append([l + w/2, t + h/2])

    corners = np.array(corners)
    corner_to_box = np.array(corner_to_box)
    centers = np.array(centers)

    # --- Ensure fixed number of regions ---
    if len(centers) > TOP_K_BOXES:
        centers = centers[:TOP_K_BOXES]
        # keep only matching corners
        valid_mask = corner_to_box < TOP_K_BOXES
        corners = corners[valid_mask]
        corner_to_box = corner_to_box[valid_mask]

    # Pad to fixed TOP_K regions
    if len(centers) < TOP_K_BOXES:
        pad = TOP_K_BOXES - len(centers)
        centers = np.vstack([centers, np.zeros((pad, 2))])

    return corners, corner_to_box, centers


# =========================================================
# ASSIGN REGION (nearest corner → fused box)
# =========================================================
def assign_regions(mouse_pts, corners, corner_to_box):

    regions = []

    for xy in mouse_pts:
        d = ((corners - xy) ** 2).sum(axis=1)
        idx = d.argmin()
        regions.append(corner_to_box[idx])

    regions = np.array(regions)
    regions = np.clip(regions, 0, TOP_K_BOXES - 1)

    return regions


# =========================================================
# DATASET
# =========================================================
class OCRRegionDataset(Dataset):

    def __init__(self, pairs):

        self.samples = []
        skipped = 0

        for ocr_path, mouse_path in pairs:

            # --- Load OCR boxes ---
            boxes = load_top_ocr_boxes(ocr_path)
            if boxes is None:
                skipped += 1
                continue

            # --- Build regions ---
            corners, corner_to_box, centers = build_regions(boxes)

            # --- Load mouse ---
            pts, _ = load_mouse_trace(mouse_path)
            if len(pts) <= SEQ_LEN:
                continue

            # --- Assign mouse → region ---
            regions = assign_regions(pts, corners, corner_to_box)

            # --- Normalize geometry (per axis, stable) ---
            centers_norm = centers.copy()

            max_x = np.max(centers[:, 0]) + 1e-6
            max_y = np.max(centers[:, 1]) + 1e-6

            centers_norm[:, 0] /= max_x
            centers_norm[:, 1] /= max_y

            geom_feat = centers_norm.flatten()   # shape = (2K,)

            # --- Build sequences ---
            for i in range(len(regions) - SEQ_LEN):

                # Previous region one-hot (temporal signal)
                prev_r = regions[i:i+SEQ_LEN]
                prev_onehot = np.eye(TOP_K_BOXES)[prev_r]   # (seq, K)

                # Repeat geometry across sequence
                geom_repeat = np.repeat(geom_feat[None, :], SEQ_LEN, axis=0)

                # Final feature = [prev_region, geometry]
                x = np.concatenate([prev_onehot, geom_repeat], axis=1)

                y = regions[i + SEQ_LEN]

                self.samples.append((x, y, centers, ocr_path))

        print("Skipped slides (bad OCR):", skipped)
        print("Total samples:", len(self.samples))

    #  REQUIRED by PyTorch
    def __len__(self):
        return len(self.samples)

    #  REQUIRED by PyTorch
    def __getitem__(self, idx):
        x, y, centers, ocr_path = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(centers, dtype=torch.float32),
            ocr_path
)

# =========================================================
# MODEL
# =========================================================
class OCRRegionModel(nn.Module):

    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(input_size=3 * TOP_K_BOXES, hidden_size=64, batch_first=True)
        self.fc = nn.Linear(64, TOP_K_BOXES)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1])
# =========================================================
# TRAIN
# =========================================================
def train_region(model, loader):

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for ep in range(EPOCHS):
        total = 0
        for x, y, _, _ in loader:
            opt.zero_grad()
            pred = model(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            total += loss.item()

        print(f"Epoch {ep+1}/{EPOCHS} Loss={total/len(loader):.4f}")

def predict_region_sequence(model, x0, centers, steps=200):
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

def region_ids_to_coords(region_ids, centers):
    coords = [centers[r] for r in region_ids]
    xs = [c[0] for c in coords]
    ys = [c[1] for c in coords]
    return np.array(xs), np.array(ys)



def evaluate_ocr_baseline(
    model,
    dataset,
    test_pairs,
    idx=0,
    steps=300,
    screen_w=1200,
    screen_h=900
):
    """
    Computes Wasserstein (Earth Mover) distance between
    predicted OCR region trajectory and real mouse trace.
    """

    model.eval()

    # -------------------------------------------------
    # 1. Get dataset sample
    # -------------------------------------------------
    x, y, centers, ocr_path = dataset[idx]
    _, mouse_path = test_pairs[idx]

    centers = centers.numpy()

    # -------------------------------------------------
    # 2. Predict region sequence
    # -------------------------------------------------
    with torch.no_grad():
        pred_ids = predict_region_sequence(
            model,
            x,
            centers,
            steps=steps
        )

    # -------------------------------------------------
    # 3. Convert region IDs → coordinates
    # -------------------------------------------------
    px, py = region_ids_to_coords(pred_ids, centers)

    pred_coords = np.stack([px, py], axis=1)

    pred_df = pd.DataFrame({
        "timestamp": np.arange(len(px)) * 0.001,
        "x": px,
        "y": py
    })

    # -------------------------------------------------
    # 4. Load ground truth mouse trace
    # -------------------------------------------------
    gt_df = mouse.load_trace_data(mouse_path)
    gt_coords = gt_df[["x", "y"]].values

    # -------------------------------------------------
    # 5. Normalize BOTH to same coordinate space
    # -------------------------------------------------
    pred_df["x"] /= screen_w
    pred_df["y"] /= screen_h

    gt_df["x"] /= screen_w
    gt_df["y"] /= screen_h

    # -------------------------------------------------
    # 6. Add anchors to FORCE identical histogram bounds
    # (cannot modify HeatMap.py, so we do this trick)
    # -------------------------------------------------
    anchors = pd.DataFrame({
        "timestamp": [-1, -1, -1, -1],
        "x": [0.0, 0.0, 1.0, 1.0],
        "y": [0.0, 1.0, 0.0, 1.0]
    })

    pred_df = pd.concat([pred_df, anchors], ignore_index=True)
    gt_df = pd.concat([gt_df, anchors], ignore_index=True)

    # -------------------------------------------------
    # 7. Build HeatMaps
    # -------------------------------------------------
    
    pred_df = pred_df.astype({
    "timestamp": "float64",
    "x": "float64",
    "y": "float64"
})

    gt_df = gt_df.astype({
    "timestamp": "float64",
    "x": "float64",
    "y": "float64"
})
    pred_hm = HeatMap(pred_df)
    gt_hm = HeatMap(gt_df)

    #pred_hm.upsample()
    #gt_hm.upsample()

    # -------------------------------------------------
    # 8. Compute Wasserstein distance
    # -------------------------------------------------
    dist = pred_hm.distance_to(gt_hm)

# -------------------------------------------------
# Trajectory RMSE (pixel space)
# -------------------------------------------------
    pred_coords = np.stack([px, py], axis=1)
    gt_coords = gt_df[["x", "y"]].values

    N = min(len(pred_coords), len(gt_coords))

    pred_coords = pred_coords[:N]
    gt_coords = gt_coords[:N]

    rmse = np.sqrt(np.mean(np.sum((pred_coords - gt_coords) ** 2, axis=1)))
    iou = pred_hm.iou_to(gt_hm)

    return dist,rmse,iou


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    train_pairs = build_slide_pairs_recursive(TRAIN_ROOT)
    test_pairs  = build_slide_pairs_recursive(TEST_ROOT)

    train_ds = OCRRegionDataset(train_pairs)
    test_ds  = OCRRegionDataset(test_pairs)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = OCRRegionModel()
    MODEL_PATH = f"ocr_only_unimodal_model_lr{LR}.pth"

    if os.path.exists(MODEL_PATH ):
        try:
            print("Loading saved model...")
            model.load_state_dict(torch.load(MODEL_PATH ))
        except Exception as e:
            print("Error loading model:")
            print(e)
            print("Model incompatible → retraining.")
            train_region(model, train_loader)
    else:
        print("Training model...")
        train_region(model, train_loader)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved.")
# --- Evaluate ---
    acc, l2, norm_l2 = evaluate_region(
    model,
    test_loader,
    screen_width=1200   # same width as mouse eval
    #same width as mouse eval
) 
    
  

print("Train slides:", len(train_pairs))
print("Test slides:", len(test_pairs))


all_dists = []
all_rmse =[]

for i in range(20):
    d,rmse,iou = evaluate_ocr_baseline(
        model,
        test_ds,
        test_pairs,
        idx=i,
        steps=SEQ_LEN
    )
    all_dists.append(d)
    all_rmse.append(rmse)
print("Train slides:", len(train_pairs))
print("Test slides:", len(test_pairs))
print("Mean Wasserstein distance:", np.mean(all_dists))
print("Mean RMSE:", np.mean(all_rmse))
print("OCR → Region Accuracy:", acc)


