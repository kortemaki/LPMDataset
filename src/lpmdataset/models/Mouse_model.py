import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from src.lpmdataset.models.shared import *
from src.lpmdataset.models.shared import evaluate_mouse


# -------- Dataset --------
class MouseOnlyDataset(Dataset):
    def __init__(self, pairs, mean, std):
        self.samples = []

        for _, m in pairs:
            _, d = load_mouse_trace(m)
            if len(d) <= SEQ_LEN:
                continue

            for i in range(len(d) - SEQ_LEN):
                x = (d[i:i+SEQ_LEN] - mean) / std
                y = (d[i+SEQ_LEN] - mean) / std
                self.samples.append((x,y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self,i):
        x,y = self.samples[i]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# -------- Model --------
class MouseOnlyLSTM(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = nn.LSTM(2, 64, batch_first=True)
        self.fc = nn.Linear(64, 2)

    def forward(self,x):
        o,_ = self.lstm(x)
        return self.fc(o[:,-1])

import pandas as pd
import numpy as np
from src.lpmdataset.representations.heatmap import HeatMap
from src.lpmdataset.modalities import mouse


def evaluate_mouse_emd(
    model,
    dataset,
    test_pairs,
    mean,
    std,
    idx=0,
    steps=300,
    screen_w=1200,
    screen_h=900
):
    model.eval()

    # -------------------------------------
    # Get initial window
    # -------------------------------------
    x0, _ = dataset[idx]
    x_seq = x0.unsqueeze(0)

    preds = []

    for _ in range(steps):
        with torch.no_grad():
            pred = model(x_seq)
        preds.append(pred.squeeze(0).numpy())

        # autoregressive shift
        x_seq = torch.cat([x_seq[:,1:], pred.unsqueeze(1)], dim=1)

    preds = np.array(preds)

    # -------------------------------------
    # Denormalize
    # -------------------------------------
    preds = preds * std + mean

    pred_df = pd.DataFrame({
        "timestamp": np.arange(len(preds)) * 0.001,
        "x": preds[:,0],
        "y": preds[:,1]
    })

    # -------------------------------------
    # Ground truth
    # -------------------------------------
    _, mouse_path = test_pairs[idx]
    gt_df = mouse.load_trace_data(mouse_path)

    # Normalize spatial bounds
    pred_df["x"] /= screen_w
    pred_df["y"] /= screen_h
    gt_df["x"] /= screen_w
    gt_df["y"] /= screen_h

    pred_df = pred_df.astype(float)
    gt_df   = gt_df.astype(float)

    # -------------------------------------
    # Heatmaps
    # -------------------------------------
    pred_hm = HeatMap(pred_df)
    gt_hm   = HeatMap(gt_df)

    pred_hm.upsample()
    gt_hm.upsample()

    return pred_hm.distance_to(gt_hm) , pred_hm.iou_to(gt_hm)


# -------- Run --------
if __name__ == "__main__":

    train_pairs = build_slide_pairs_recursive(TRAIN_ROOT)
    test_pairs  = build_slide_pairs_recursive(TEST_ROOT)

    mean,std = compute_delta_stats(train_pairs)

    train_ds = MouseOnlyDataset(train_pairs, mean, std)
    test_ds  = MouseOnlyDataset(test_pairs, mean, std)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_ds, batch_size=BATCH_SIZE)

    model = MouseOnlyLSTM()

    MODEL_PATH = f"mouse_only_unimodal_model_lr{LR}.pth"

    if os.path.exists(MODEL_PATH ):
        try:
            print("Loading saved model...",{MODEL_PATH})
            model.load_state_dict(torch.load(MODEL_PATH ))
        except:
            print("Model incompatible → retraining.")
            train_model(model, train_loader)
            torch.save(model.state_dict(), MODEL_PATH)
    else:
        print("Training model...")
        train_model(model, train_loader)
        torch.save(model.state_dict(), MODEL_PATH)
        print("Model saved.")


    ##########
    train_model(model, train_loader)
# --- Evaluate ---
    rmse, nrmse, acc2pct = evaluate_mouse(
    model,
    test_loader,
    mean,
    std,
    screen_width=1200   # adjust if needed
)
    emd ,iou = evaluate_mouse_emd(
    model,
    test_ds,
    test_pairs,
    mean,
    std,
    idx=10,
    steps=SEQ_LEN
)
    

    print("Mouse→Mouse Wasserstein:", emd)
    print("Mouse→Mouse RMSE:", rmse)
    print("Mouse→Mouse HeatMap IoU:", iou)
