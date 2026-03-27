import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from src.lpmdataset.models.shared import *


# =========================================================
# DATASET (UNCHANGED)
# =========================================================

class MouseOnlyDataset(Dataset):
    def __init__(self, pairs, mean, std, return_meta=False):
        self.samples = []
        self.return_meta = return_meta

        for _, m in pairs:
            pts, deltas = load_mouse_trace(m)

            if len(deltas) <= SEQ_LEN:
                continue

            for i in range(len(deltas) - SEQ_LEN):

                x = (deltas[i:i+SEQ_LEN] - mean) / std
                y = (deltas[i+SEQ_LEN] - mean) / std
                start_pos = pts[i + SEQ_LEN - 1]

                if return_meta:
                    self.samples.append((x, y, start_pos, pts, i, m))
                else:
                    self.samples.append((x, y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        sample = self.samples[i]

        if self.return_meta:
            x, y, start_pos, pts, idx, path = sample
            return (
                torch.tensor(x, dtype=torch.float32),
                torch.tensor(y, dtype=torch.float32),
                torch.tensor(start_pos, dtype=torch.float32),
                pts,
                idx,
                path
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
# ✅ FIXED FULL-SLIDE PREDICTION (NO RETRAINING NEEDED)
# =========================================================

def generate_full_slide_predictions(model, test_pairs, mean, std):

    print("Generating FULL slide predictions (FIXED)...")

    model.eval()

    for pair in test_pairs:

        try:
            # -------------------------
            # SAFE unpack
            # -------------------------
            if isinstance(pair, (list, tuple)):
                mouse_path = pair[1] if len(pair) > 1 else pair[0]
            else:
                mouse_path = pair

            # -------------------------
            # LOAD + FORCE NUMPY ✅ FIX
            # -------------------------
            pts, deltas = load_mouse_trace(mouse_path)

            pts = np.array(pts)          # 🔥 FIX
            deltas = np.array(deltas)    # 🔥 FIX

            if len(deltas) <= SEQ_LEN:
                continue

            # -------------------------
            # INITIAL WINDOW
            # -------------------------
            x_seq = (deltas[:SEQ_LEN] - mean) / std
            x_seq = torch.tensor(x_seq, dtype=torch.float32).unsqueeze(0)

            start_pos = pts[SEQ_LEN - 1]

            preds = []

            # -------------------------
            # FULL LENGTH ✅ FIX
            # -------------------------
            steps = len(deltas) - SEQ_LEN

            for step in range(steps):

                with torch.no_grad():
                    pred = model(x_seq)

                pred_np = pred.squeeze(0).cpu().numpy()
                preds.append(pred_np)

                # -------------------------
                # TEACHER FORCING ✅ FIX
                # -------------------------
                next_real = (deltas[SEQ_LEN + step] - mean) / std
                next_real = torch.tensor(next_real, dtype=torch.float32).view(1, 1, 2)

                x_seq = torch.cat([x_seq[:, 1:], next_real], dim=1)

            preds = np.array(preds)

            # -------------------------
            # DENORMALIZE
            # -------------------------
            preds = preds * std + mean

            # -------------------------
            # BOUNDS FROM REAL DATA ✅ FIX
            # -------------------------
            x_min, y_min = pts.min(axis=0)
            x_max, y_max = pts.max(axis=0)

            # -------------------------
            # BUILD ABSOLUTE TRAJECTORY
            # -------------------------
            abs_preds = []
            current_pos = start_pos.copy()

            for dx, dy in preds:
                current_pos = current_pos + np.array([dx, dy])

                # ✅ SAFE CLAMP (NO NEGATIVE EXPLOSION)
                current_pos[0] = np.clip(current_pos[0], x_min, x_max)
                current_pos[1] = np.clip(current_pos[1], y_min, y_max)

                abs_preds.append(current_pos.copy())

            abs_preds = np.array(abs_preds)

            # -------------------------
            # GOLD (PERFECT ALIGNMENT)
            # -------------------------
            true_abs = pts[SEQ_LEN : SEQ_LEN + steps]

            # -------------------------
            # SAFETY MATCH LENGTH
            # -------------------------
            N = min(len(abs_preds), len(true_abs))

            abs_preds = abs_preds[:N]
            true_abs = true_abs[:N]

            # -------------------------
            # SAVE
            # -------------------------
            parts = mouse_path.split(os.sep)

            video = parts[-2]
            order = parts[-3]
            anat  = parts[-4]

            slide_name = os.path.basename(mouse_path).replace(".csv", "")

            out_dir = os.path.join(
                "results",
                "mouse_only",
                anat,
                order,
                video
            )

            os.makedirs(out_dir, exist_ok=True)

            df = pd.DataFrame({
                "time": np.arange(N),
                "pred_x": abs_preds[:, 0],
                "pred_y": abs_preds[:, 1],
                "gold_x": true_abs[:, 0],
                "gold_y": true_abs[:, 1],
            })

            out_path = os.path.join(out_dir, f"{slide_name}.csv")
            df.to_csv(out_path, index=False)

            print(f"Saved: {out_path} | Points: {N}")

        except Exception as e:
            print("Skipping:", e)

    print("Done.")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":



    train_pairs = build_slide_pairs_recursive(TRAIN_ROOT)
    test_pairs  = build_slide_pairs_recursive(TEST_ROOT)

    mean, std = compute_delta_stats(train_pairs)

    train_ds = MouseOnlyDataset(train_pairs, mean, std, return_meta=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

    model = MouseOnlyLSTM()

    MODEL_PATH = f"mouse_only_model_FULL{LR}.pth"

    if os.path.exists(MODEL_PATH):
        print("Loading saved model...")
        model.load_state_dict(torch.load(MODEL_PATH))
    else:
        print("Training model...")
        train_model(model, train_loader)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved.")

    # =====================================================
    # ✅ FIXED GENERATION
    # =====================================================

    generate_full_slide_predictions(
        model,
        test_pairs,
        mean,
        std
    )