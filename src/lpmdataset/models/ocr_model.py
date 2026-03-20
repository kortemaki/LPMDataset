import os, csv
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from src.lpmdataset.models.shared import *
from src.lpmdataset.modalities import mouse

# ================= CONFIG =================
TOP_K_BOXES = 80
SCREEN_W, SCREEN_H = 1200, 900
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# OCR → CENTERS
# =========================================================
def load_ocr_regions(path, K=TOP_K_BOXES):
    boxes = []
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        reader = csv.reader(f)
        try:
            header = [h.strip().lower() for h in next(reader)]
        except StopIteration:
            return None

        req = ["left","top","width","height","conf"]
        if not all(k in header for k in req):
            return None

        li, ti, wi, hi, ci = [header.index(k) for k in req]

        for row in reader:
            try:
                l,t,w,h,c = float(row[li]),float(row[ti]),float(row[wi]),float(row[hi]),float(row[ci])
            except:
                continue
            if c <= 0 or w < 8 or h < 8 or w*h < 100:
                continue
            boxes.append((w*h*(c/100.0),(l,t,w,h)))

    if not boxes:
        return None

    boxes = [b for _,b in sorted(boxes, reverse=True)[:K]]
    centers = np.array([[l+w/2, t+h/2] for (l,t,w,h) in boxes])

    if len(centers) < K:
        centers = np.vstack([centers, np.zeros((K-len(centers),2))])

    centers[:,0] /= SCREEN_W
    centers[:,1] /= SCREEN_H
    return centers


# =========================================================
# REGION ASSIGNMENT
# =========================================================
def assign_regions(mouse_pts, centers):
    d = ((mouse_pts[:,None,:] - centers[None,:,:])**2).sum(axis=2)
    return np.argmin(d, axis=1)


# =========================================================
# DATASET (LAZY)
# =========================================================
class OCRRegionDataset(Dataset):
    def __init__(self, pairs):
        self.data = []
        print("Building dataset...")

        for ocr_path, mouse_path in pairs:
            centers = load_ocr_regions(ocr_path)
            if centers is None:
                continue

            pts, _ = load_mouse_trace(mouse_path)
            if len(pts) <= SEQ_LEN:
                continue

            regions = assign_regions(pts, centers)
            self.data.append((regions, centers, ocr_path, mouse_path))

        print("Slides loaded:", len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        regions, centers, ocr_path, mouse_path = self.data[idx]

        i = np.random.randint(0, len(regions) - SEQ_LEN)
        geom = centers.flatten()

        x = np.concatenate([
            np.eye(TOP_K_BOXES)[regions[i:i+SEQ_LEN]],
            np.repeat(geom[None,:], SEQ_LEN, axis=0)
        ], axis=1)

        y = regions[i+SEQ_LEN]

        return (
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
            torch.tensor(centers, dtype=torch.float32),
            mouse_path
        )


# =========================================================
# MODEL
# =========================================================
class OCRRegionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(3*TOP_K_BOXES, 16, batch_first=True)
        self.fc = nn.Linear(16, TOP_K_BOXES)

    def forward(self, x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1])


# =========================================================
# TRAIN
# =========================================================
def train_region(model, loader):
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    model.train()

    for ep in range(EPOCHS):
        tot = 0

        for x,y,_,_ in loader:
            x = x.to(device)
            y = y.to(device)

            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()

            tot += loss.item()

        print(f"Epoch {ep+1}: {tot/len(loader):.4f}")


# =========================================================
# SAVE FULL SLIDE (STRUCTURED)
# =========================================================
def generate_full_slide_predictions_ocr(model, test_pairs):

    print("Generating OCR-only predictions...")

    model.eval()

    for (ocr_path, mouse_path) in test_pairs:

        try:
            centers = load_ocr_regions(ocr_path)
            if centers is None:
                continue

            pts, _ = load_mouse_trace(mouse_path)
            if len(pts) <= SEQ_LEN:
                continue

            regions = assign_regions(pts, centers)
            geom = centers.flatten()

            x = np.concatenate([
                np.eye(TOP_K_BOXES)[regions[:SEQ_LEN]],
                np.repeat(geom[None,:], SEQ_LEN, axis=0)
            ], axis=1)

            seq = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)

            preds = []
            steps = len(regions) - SEQ_LEN

            for step in range(steps):
                with torch.no_grad():
                    r = model(seq).argmax(1).item()

                preds.append(r)

                # teacher forcing
                next_real = regions[SEQ_LEN + step]

                onehot = torch.zeros(1,1,TOP_K_BOXES).to(device)
                onehot[0,0,next_real] = 1

                geom_t = torch.tensor(geom, dtype=torch.float32).view(1,1,-1).to(device)

                next_input = torch.cat([onehot, geom_t], dim=2)
                seq = torch.cat([seq[:,1:], next_input], dim=1)

            pred_coords = np.array([centers[r] for r in preds])
            gt = mouse.load_trace_data(mouse_path)

            N = min(len(pred_coords), len(gt))

            pred_coords = pred_coords[:N]
            gt = gt.iloc[:N]

            # ===== SAME STRUCTURE =====
            parts = mouse_path.split(os.sep)
            video = parts[-2]
            order = parts[-3]
            root  = parts[-4]

            slide_name = os.path.basename(mouse_path).replace("_trace.csv","")

            out_dir = os.path.join("results","ocr_only",root,order,video)
            os.makedirs(out_dir, exist_ok=True)

            df = pd.DataFrame({
                "time": np.arange(N),
                "pred_x": pred_coords[:,0],
                "pred_y": pred_coords[:,1],
                "gold_x": gt.x.values,
                "gold_y": gt.y.values
            })

            out_path = os.path.join(out_dir, f"{slide_name}.csv")
            df.to_csv(out_path, index=False)

            print(f"Saved: {out_path}")

        except Exception as e:
            print("Skipping:", e)

    print("Done.")


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    train_pairs = build_slide_pairs_recursive(TRAIN_ROOT)
    test_pairs  = build_slide_pairs_recursive(TEST_ROOT)

    train_ds = OCRRegionDataset(train_pairs)
    test_ds  = OCRRegionDataset(test_pairs)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    model = OCRRegionModel().to(device)

    if os.path.exists("ocr_FULL_model.pth"):
        model.load_state_dict(torch.load("ocr_FULL_model.pth"))
    else:
        train_region(model, train_loader)
        torch.save(model.state_dict(),"ocr_FULL_model.pth")

    print("Training complete.")

    # ===== SAVE RESULTS =====
    generate_full_slide_predictions_ocr(model, test_pairs)