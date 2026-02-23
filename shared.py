import os
import numpy as np
import csv
from glob import glob
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# ---------------- CONFIG ----------------
SEQ_LEN = 20
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
SEED = 42

#DATA_ROOT = "mlpdataset/data_oct/anat-1"

#TRAIN_PATTERNS = ["0*", "1*","20","21","22","23"]
#TEST_PATTERNS = ["24","25","26","27","28","29"]

TRAIN_ROOT = "mlpdataset/data_oct/psy-1"
TEST_ROOT  = "mlpdataset/data_oct/ml-1"

# If you want all folders inside each:
TRAIN_PATTERNS = ["*"]
TEST_PATTERNS  = ["*"]


np.random.seed(SEED)
torch.manual_seed(SEED)

# ---------------- ALIGN FILES ----------------
from glob import glob
import os


from glob import glob
import os


def build_slide_pairs_recursive(root):
    """
    Works for ANY dataset layout depth.
    Finds all slide_*_trace.csv recursively and matches OCR.
    """

    trace_files = glob(os.path.join(root, "**", "slide_*_trace.csv"), recursive=True)

    print(f"{root} → Found {len(trace_files)} trace files")
    
    pairs = []

    for trace_path in trace_files:
        ocr_path = trace_path.replace("_trace.csv", "_ocr.csv")

        if os.path.exists(ocr_path):
            pairs.append((ocr_path, trace_path))
        else:
            print("⚠ Missing OCR:", trace_path)

    print("Aligned pairs:", len(pairs))
    return pairs

# ---------------- LOAD MOUSE ----------------
def load_mouse_trace(path):
    xs, ys = [], []

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]
        ci = header.index("coord")

        for row in reader:
            coord = row[ci].strip().strip("()")
            if not coord:
                continue
            try:
                x_str, y_str = coord.split(",")
                xs.append(float(x_str))
                ys.append(float(y_str))
            except:
                continue

    pts = np.column_stack([xs, ys])
    if len(pts) < 2:
        return np.zeros((0,2)), np.zeros((0,2))

    deltas = pts[1:] - pts[:-1]
    return pts[:-1], deltas


# ---------------- LOAD OCR FEATURE ----------------
def load_ocr_feature(path):
    xs, ys = [], []

    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        header = [h.strip().lower() for h in next(reader)]

        if not all(k in header for k in ["left","top","width","height"]):
            return np.zeros(2)

        li = header.index("left")
        ti = header.index("top")
        wi = header.index("width")
        hi = header.index("height")

        for row in reader:
            try:
                l = float(row[li])
                t = float(row[ti])
                w = float(row[wi])
                h = float(row[hi])
                xs.append(l + w/2)
                ys.append(t + h/2)
            except:
                continue

    if len(xs) == 0:
        return np.zeros(2)

    return np.array([np.mean(xs), np.mean(ys)])


# ---------------- NORMALIZATION ----------------
def compute_delta_stats(pairs):
    all_d = []
    for _, m in pairs:
        _, d = load_mouse_trace(m)
        if len(d) > 0:
            all_d.append(d)

    all_d = np.concatenate(all_d, axis=0)
    return all_d.mean(0), all_d.std(0) + 1e-6


# ---------------- TRAINING ----------------
def train_model(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.MSELoss()

    for ep in range(EPOCHS):
        total = 0
        for x,y in loader:
            optimizer.zero_grad()
            pred = model(x)
            loss = loss_fn(pred,y)
            loss.backward()
            optimizer.step()
            total += loss.item()

        print(f"Epoch {ep+1}/{EPOCHS} Loss={total/len(loader):.4f}")


# ---------------- EVALUATION ----------------
#Mouse -> Mouse
def evaluate_mouse(model, loader, mean, std, screen_width=1200):

    model.eval()
    preds, trues = [], []

    with torch.no_grad():
        for x, y in loader:
            p = model(x)
            p = p.numpy() * std + mean
            t = y.numpy() * std + mean
            preds.append(p)
            trues.append(t)

    preds = np.concatenate(preds)
    trues = np.concatenate(trues)

    # --- L2 (RMSE) ---
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)

    # --- Normalized RMSE ---
    nrmse = rmse / screen_width

    # --- Spatial Accuracy @2% screen width ---
    threshold = 0.02 * screen_width
    dists = np.sqrt(((preds - trues) ** 2).sum(axis=1))
    acc2pct = np.mean(dists <= threshold)

    print("\n=== Mouse → Mouse ===")
    print("RMSE (px):", rmse)
    print("Normalized RMSE:", nrmse)
    print("Spatial Acc@2% width:", acc2pct)

    return rmse, nrmse, acc2pct

#  OCR->Mouse
def evaluate_region(model, loader, screen_width=1200):

    model.eval()
    correct = 0
    total = 0
    l2_dists = []

    with torch.no_grad():
        for x, y, centers,_ in loader:

            logits = model(x)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += len(y)

            for i in range(len(y)):
                true_c = centers[i][y[i]].numpy()
                pred_c = centers[i][pred[i]].numpy()

                l2 = np.sqrt(((true_c - pred_c) ** 2).sum())
                l2_dists.append(l2)

    acc = correct / total
    mean_l2 = np.mean(l2_dists)

    # --- Normalized L2 ---
    norm_l2 = mean_l2 / screen_width

    print("\n=== OCR → Region ===")
    print("Region Accuracy:", acc)
    print("Mean L2 Distance (px):", mean_l2)
    print("Normalized L2:", norm_l2)

    return acc, mean_l2, norm_l2