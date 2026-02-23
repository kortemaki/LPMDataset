from typing import Counter

import numpy as np
import csv
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from shared import *
from shared import evaluate_region
import matplotlib.pyplot as plt
import random


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

    # 🔴 REQUIRED by PyTorch
    def __len__(self):
        return len(self.samples)

    # 🔴 REQUIRED by PyTorch
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


# =========================================================
# 1. TRAIN vs TEST PERFORMANCE
# =========================================================
def check_train_vs_test(model, train_loader, test_loader):
    print("\n--- Sanity Check: Train vs Test ---")

    train_acc, _, _ = evaluate_region(model, train_loader, screen_width=1200)
    test_acc, _, _  = evaluate_region(model, test_loader, screen_width=1200)

    print(f"Train Acc: {train_acc:.4f}")
    print(f"Test  Acc: {test_acc:.4f}")

    if train_acc - test_acc > 0.15:
        print("⚠ Possible overfitting")
    elif train_acc > 0.98:
        print("⚠ Suspiciously high train accuracy → check leakage")
    else:
        print("✔ Looks normal")


# =========================================================
# 2. REGION DISTRIBUTION (CLASS IMBALANCE)
# =========================================================
def check_region_distribution(dataset, name="Dataset"):
    print(f"\n--- Region Distribution: {name} ---")

    counts = Counter()
    for _, y, _ in dataset:
        counts[int(y)] += 1

    total = sum(counts.values())
    for k in sorted(counts.keys()):
        print(f"Region {k}: {counts[k]/total:.3f}")

    max_ratio = max(counts.values()) / total
    if max_ratio > 0.6:
        print("⚠ Strong class imbalance → accuracy may be inflated")
    else:
        print("✔ Distribution looks reasonable")


# =========================================================
# 3. RANDOM BASELINE
# =========================================================
def check_random_baseline(dataset, K):
    print("\n--- Random Baseline ---")

    correct = 0
    total = 0

    for _, y, _ in dataset:
        pred = random.randint(0, K-1)
        if pred == int(y):
            correct += 1
        total += 1

    acc = correct / total
    print("Random accuracy:", acc)
    print("Your model should be >> random")


# =========================================================
# 4. SHUFFLE TEMPORAL ORDER (DESTROY TEMPORAL SIGNAL)
# =========================================================
def check_shuffle_temporal(model, loader):
    print("\n--- Temporal Shuffle Test ---")

    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for x, y, _ in loader:

            # Shuffle sequence order (breaks temporal info)
            idx = torch.randperm(x.size(1))
            x_shuffled = x[:, idx, :]

            logits = model(x_shuffled)
            pred = logits.argmax(dim=1)

            correct += (pred == y).sum().item()
            total += len(y)

    acc = correct / total
    print("Accuracy after shuffle:", acc)

    if acc < 0.4:
        print("✔ Model relies on temporal signal (good)")
    else:
        print("⚠ Model not using temporal info properly")


# =========================================================
# 5. REGION TRANSITION ENTROPY
# =========================================================
def check_transition_entropy(dataset, K):
    print("\n--- Region Transition Entropy ---")

    trans = np.zeros((K, K))

    for x, y, _ in dataset:
        prev_region = np.argmax(x[-1][:K])  # last prev region
        trans[prev_region, int(y)] += 1

    probs = trans / (trans.sum(axis=1, keepdims=True) + 1e-6)
    entropy = -np.sum(probs * np.log(probs + 1e-6)) / K

    print("Transition entropy:", entropy)

    if entropy < 0.5:
        print("⚠ Transitions too predictable → task easy")
    else:
        print("✔ Reasonable transition diversity")


# =========================================================
# 6. DATASET OVERLAP CHECK
# =========================================================
def check_dataset_overlap(train_ds, test_ds):
    print("\n--- Train/Test Overlap Check ---")

    train_hashes = set()
    for x, y, _ in train_ds:
        train_hashes.add(hash(x.numpy().tobytes()))

    overlap = 0
    for x, y, _ in test_ds:
        if hash(x.numpy().tobytes()) in train_hashes:
            overlap += 1

    print("Overlapping samples:", overlap)

    if overlap > 0:
        print("⚠ Possible data leakage")
    else:
        print("✔ No direct sample overlap")


# =========================================================
# RUN ALL CHECKS
# =========================================================
def run_all_sanity_checks(model, train_ds, test_ds, train_loader, test_loader):

    print("\n================ SANITY CHECKS ================")

    check_train_vs_test(model, train_loader, test_loader)
    check_region_distribution(train_ds, "Train")
    check_region_distribution(test_ds, "Test")
    check_random_baseline(test_ds, TOP_K_BOXES)
    check_shuffle_temporal(model, test_loader)
    check_transition_entropy(test_ds, TOP_K_BOXES)
    check_dataset_overlap(train_ds, test_ds)

    print("\n================ DONE ================")


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
    MODEL_PATH = f"ocr_K{TOP_K_BOXES}_lr{LR}.pth"

    if os.path.exists(MODEL_PATH ):
        try:
            print("Loading saved model...")
            model.load_state_dict(torch.load(MODEL_PATH ))
        except:
            print("Model incompatible → retraining.")
            train_region(model, train_loader)
            torch.save(model.state_dict(), MODEL_PATH)
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
    run_all_sanity_checks(model, train_ds, test_ds, train_loader, test_loader)

    print("Train slides:", len(train_pairs))
    print("Test slides:", len(test_pairs))

    


from scipy.spatial import Voronoi
import matplotlib.pyplot as plt
import numpy as np


def visualize_voronoi_prediction(model, dataset, idx=0):

    model.eval()
    x, y, centers, ocr_path = dataset[idx]

    with torch.no_grad():
        logits = model(x.unsqueeze(0))
        pred = logits.argmax(dim=1).item()

    true_region = y.item()
    centers = centers.numpy()

    # --- Rebuild Voronoi for this slide ---
    boxes = load_top_ocr_boxes(ocr_path) # careful if needed
    corners, corner_to_box, _ = build_regions(boxes)
    vor = Voronoi(corners)

    plt.figure(figsize=(8,8))

    # --- Plot Voronoi polygons ---
    for region_idx in vor.regions:
        if not region_idx or -1 in region_idx:
            continue
        polygon = [vor.vertices[i] for i in region_idx]
        polygon = np.array(polygon)
        plt.fill(polygon[:,0], polygon[:,1], alpha=0.2, edgecolor="black")

    # --- Plot region centers ---
    plt.scatter(centers[:,0], centers[:,1], c="blue", label="Region Centers")

    # --- True region ---
    plt.scatter(centers[true_region][0],
                centers[true_region][1],
                c="green", s=200, edgecolors="black",
                label="True")

    # --- Predicted region ---
    plt.scatter(centers[pred][0],
                centers[pred][1],
                c="red", s=150,
                label="Pred")

    plt.gca().invert_yaxis()
    plt.legend()
    plt.title("Voronoi Regions (Green=True, Red=Pred)")
    plt.show()
    print("Train slides:", len(train_pairs))
print("Test slides:", len(test_pairs))
#visualize_voronoi_prediction(model, test_ds, idx=10)
