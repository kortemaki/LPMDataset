import os
import csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter

from src.lpmdataset.models.shared import *
from src.lpmdataset.models.shared import evaluate_region
from src.lpmdataset.modalities import mouse
from src.lpmdataset.representations.heatmap import HeatMap


# =========================================================
# CONFIG
# =========================================================
TOP_K_BOXES = 80
TEXT_DIM = 100


# =========================================================
# OCR BOX SELECTION
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

            if conf <= 0 or w < 8 or h < 8 or w * h < 100:
                continue

            score = (w * h) * (conf / 100.0)
            boxes.append((score, (l, t, w, h)))

    if len(boxes) == 0:
        return None

    boxes.sort(key=lambda x: x[0], reverse=True)
    return [b for _, b in boxes[:K]]


# =========================================================
# REGION BUILDING
# =========================================================
def build_regions(boxes):

    corners, corner_to_box, centers = [], [], []

    for box_id, (l, t, w, h) in enumerate(boxes):
        pts = [(l, t), (l+w, t), (l, t+h), (l+w, t+h)]
        for p in pts:
            corners.append(p)
            corner_to_box.append(box_id)

        centers.append([l + w/2, t + h/2])

    centers = np.array(centers)
    corners = np.array(corners)
    corner_to_box = np.array(corner_to_box)

    if len(centers) < TOP_K_BOXES:
        pad = TOP_K_BOXES - len(centers)
        centers = np.vstack([centers, np.zeros((pad, 2))])

    return corners, corner_to_box, centers


def assign_regions(mouse_pts, corners, corner_to_box):

    regions = []
    for xy in mouse_pts:
        d = ((corners - xy) ** 2).sum(axis=1)
        regions.append(corner_to_box[d.argmin()])

    return np.clip(np.array(regions), 0, TOP_K_BOXES - 1)


# =========================================================
# ASR UTILITIES
# =========================================================
def build_vocab(pairs, max_vocab=TEXT_DIM):

    counter = Counter()

    for _, mouse_path in pairs:
        asr_path = mouse_path.replace("_trace.csv", "_spoken.csv")
        if not os.path.exists(asr_path):
            continue

        df = pd.read_csv(asr_path)
        counter.update(df["Word"].astype(str).str.lower().tolist())

    vocab = [w for w, _ in counter.most_common(max_vocab)] #count / word> takes top 100 words.
    return {w: i for i, w in enumerate(vocab)}


def asr_to_vector(asr_path, word_to_idx):

    vec = np.zeros(len(word_to_idx))

    if not os.path.exists(asr_path):
        return vec

    df = pd.read_csv(asr_path)

    for w in df["Word"].astype(str).str.lower():
        if w in word_to_idx:
            vec[word_to_idx[w]] += 1

    if vec.sum() > 0:
        vec /= vec.sum()

    return vec


# =========================================================
# DATASET
# =========================================================
class OCRASRDataset(Dataset):

    def __init__(self, pairs, word_to_idx):

        self.samples = []

        for slide_idx, (ocr_path, mouse_path) in enumerate(pairs):

            boxes = load_top_ocr_boxes(ocr_path)
            if boxes is None:
                continue

            corners, corner_to_box, centers = build_regions(boxes)

            pts, _ = load_mouse_trace(mouse_path)
            if len(pts) <= SEQ_LEN:
                continue

            regions = assign_regions(pts, corners, corner_to_box)

            asr_path = mouse_path.replace("_trace.csv", "_spoken.csv")
            text_vec = asr_to_vector(asr_path, word_to_idx)

            centers_norm = centers.copy()
            centers_norm[:, 0] /= (centers[:, 0].max() + 1e-6)
            centers_norm[:, 1] /= (centers[:, 1].max() + 1e-6)
            geom_feat = centers_norm.flatten()

            for i in range(len(regions) - SEQ_LEN):

                prev_onehot = np.eye(TOP_K_BOXES)[regions[i:i+SEQ_LEN]]
                geom_repeat = np.repeat(geom_feat[None, :], SEQ_LEN, axis=0)
                text_repeat = np.repeat(text_vec[None, :], SEQ_LEN, axis=0)

                x = np.concatenate([prev_onehot, geom_repeat, text_repeat], axis=1)
                y = regions[i + SEQ_LEN]

                # 🔥 Store slide index properly
                self.samples.append((x, y, centers, slide_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y, centers, slide_idx = self.samples[idx]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(centers, dtype=torch.float32),
            slide_idx
        )

# =========================================================
# MODEL
# =========================================================
class OCRASRModel(nn.Module):

    def __init__(self):
        super().__init__()
        input_dim = 3 * TOP_K_BOXES + TEXT_DIM
        self.lstm = nn.LSTM(input_dim, 64, batch_first=True)
        self.fc = nn.Linear(64, TOP_K_BOXES)

    def forward(self, x):
        o, _ = self.lstm(x)
        return self.fc(o[:, -1])


# =========================================================
# TRAIN
# =========================================================
def train_region(model, loader):

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(EPOCHS):

        total_loss = 0

        for x, y, _, _ in loader:

            optimizer.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} Loss={total_loss/len(loader):.4f}")

def compute_heatmap_iou(df1, df2, bins=16):
    """
    Computes spatial IoU between two coordinate DataFrames.
    Coordinates must be normalized to [0,1].
    """

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
def evaluate_multimodal(
    model,
    dataset,
    test_pairs,
    idx=0,
    steps=SEQ_LEN,
    screen_w=1200,
    screen_h=900
):

    model.eval()

    # 🔥 Now dataset gives slide_idx
    x, _, centers, slide_idx = dataset[idx]

    # Correct slide-level matching
    _, mouse_path = test_pairs[slide_idx]

    seq = x.unsqueeze(0)
    preds = []

    with torch.no_grad():
        for _ in range(steps):

            r = model(seq).argmax(dim=1).item()
            preds.append(r)

            onehot = torch.zeros(1, 1, TOP_K_BOXES)
            onehot[0, 0, r] = 1

            geom = seq[:, -1, TOP_K_BOXES:3*TOP_K_BOXES].unsqueeze(1)
            text = seq[:, -1, 3*TOP_K_BOXES:].unsqueeze(1)

            next_input = torch.cat([onehot, geom, text], dim=2)
            seq = torch.cat([seq[:, 1:], next_input], dim=1)

    # Convert regions → coordinates
    pred_coords = np.array([centers.numpy()[r] for r in preds])
    pred_df = pd.DataFrame(pred_coords, columns=["x", "y"])

    # Load correct GT
    gt_df = mouse.load_trace_data(mouse_path)

    # Normalize
    pred_df["x"] /= screen_w
    pred_df["y"] /= screen_h
    gt_df["x"] /= screen_w
    gt_df["y"] /= screen_h

    # ---------- RMSE ----------
    N = min(len(pred_df), len(gt_df))
    rmse = np.sqrt(np.mean(np.sum(
        (pred_df[["x","y"]].values[:N] -
         gt_df[["x","y"]].values[:N]) ** 2, axis=1)))

    # ---------- IoU ----------
    iou = compute_heatmap_iou(pred_df, gt_df, bins=16)

    # ---------- Wasserstein ----------
    anchors = pd.DataFrame({"x":[0,0,1,1],"y":[0,1,0,1]})
    pred_df_emd = pd.concat([pred_df, anchors], ignore_index=True)
    gt_df_emd   = pd.concat([gt_df, anchors], ignore_index=True)

    emd = HeatMap(pred_df_emd).distance_to(HeatMap(gt_df_emd))

    return emd, rmse, iou


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    # If you have multiple train folders
    TRAIN_ROOTS = [
        "mlpdataset/data_oct/anat-1/AnatomyPhysiology/01",
        "mlpdataset/data_oct/anat-1/AnatomyPhysiology/02",
        "mlpdataset/data_oct/anat-1/AnatomyPhysiology/03",
        "mlpdataset/data_oct/anat-1/AnatomyPhysiology/04",
        "mlpdataset/data_oct/anat-1/AnatomyPhysiology/05",
    ]

    TEST_ROOTS = [
        "mlpdataset/data_oct/anat-2",       
    ]

    # -------- Build train pairs --------
    train_pairs = []
    for root in TRAIN_ROOTS:
        train_pairs.extend(build_slide_pairs_recursive(root))

    # -------- Build test pairs --------
    test_pairs = []
    for root in TEST_ROOTS:
        test_pairs.extend(build_slide_pairs_recursive(root))

    print("Train slides:", len(train_pairs))
    print("Test slides:", len(test_pairs))

    # -------- Build vocab --------
    word_to_idx = build_vocab(train_pairs)

    # -------- Build datasets --------
    train_ds = OCRASRDataset(train_pairs, word_to_idx)
    test_ds  = OCRASRDataset(test_pairs, word_to_idx)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = OCRASRModel()

    if not os.path.exists("ocr_asr_model.pth"):
        train_region(model, train_loader)
        torch.save(model.state_dict(), "ocr_asr_model.pth")
    else:
        model.load_state_dict(torch.load("ocr_asr_model.pth"))

    acc, _, _ = evaluate_region(
        model,
        DataLoader(test_ds, batch_size=BATCH_SIZE),
        1200
    )

    all_emd = []
    all_rmse = []
    all_iou = []

    for idx in range(20):   # first 50 samples
        emd, rmse, iou = evaluate_multimodal(
            model,
            test_ds,
            test_pairs,
            idx=idx
        )

        all_emd.append(emd)
        all_rmse.append(rmse)
        all_iou.append(iou)
    

    print("OCR+ASR Accuracy:", acc)
    print("OCR+ASR Wasserstein:", emd)
    print("OCR+ASR RMSE:", rmse)
    print("OCR+ASR IoU:", iou)
    print("OCR+ASR IoU: {:.6e}".format(iou))