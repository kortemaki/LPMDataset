import os
from glob import glob
import csv
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt




# ---------------------------------------------------------
# CONFIG
# ---------------------------------------------------------
DATA_FOLDER = "mlpdataset/data_oct/anat-1/AnatomyPhysiology/*/"
WINDOW_SIZE = 20
TEST_RATIO = 0.5
BATCH_SIZE = 64
EPOCHS = 20
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)
np.random.seed(SEED)

# ---------------------------------------------------------
# LOAD ONE SLIDE
# ---------------------------------------------------------
def load_mouse_data(path):
    xs, ys = [], []

    with open(path, "r") as f:
        reader = csv.reader(f)
        next(reader)  # skip header

        for row in reader:
            coord = row[2].strip('"').strip("()")
            x_str, y_str = coord.split(",")
            xs.append(float(x_str))
            ys.append(float(y_str))

    return np.array(xs), np.array(ys)

# ---------------------------------------------------------
# COMPUTE DELTAS
# ---------------------------------------------------------
def compute_deltas(xs, ys):
    dx = np.diff(xs)
    dy = np.diff(ys)
    return np.column_stack([dx, dy])

# ---------------------------------------------------------
# LOAD ALL SLIDES
# ---------------------------------------------------------
def load_all_slides(folder_path):
    pattern = os.path.join(folder_path, "slide_*_trace.csv")
    files = sorted(glob(pattern))

    slides = []

    for file in files:
        xs, ys = load_mouse_data(file)
        deltas = compute_deltas(xs, ys)

        if len(deltas) > WINDOW_SIZE:
            slides.append(deltas)

    print(f"Loaded {len(slides)} slides.")
    return slides

# ---------------------------------------------------------
# SLIDE-LEVEL SPLIT
# ---------------------------------------------------------
def split_slides(slides, test_ratio=0.25):
    indices = np.random.permutation(len(slides))
    split = int(len(slides) * (1 - test_ratio))

    train_idx = indices[:split]
    test_idx = indices[split:]

    train_slides = [slides[i] for i in train_idx]
    test_slides = [slides[i] for i in test_idx]

    print(f"Train slides: {len(train_slides)}")
    print(f"Test slides: {len(test_slides)}")

    return train_slides, test_slides

# ---------------------------------------------------------
# CREATE SLIDING WINDOWS
# ---------------------------------------------------------
def create_windows_from_slides(slides, window_size):
    X_all = []
    Y_all = []



    for deltas in slides:
        for i in range(len(deltas) - window_size):
            X_all.append(deltas[i:i+window_size])
            Y_all.append(deltas[i+window_size])
    X_all = np.array(X_all)
    Y_all = np.array(Y_all)

    return torch.tensor(X_all, dtype=torch.float32), \
           torch.tensor(Y_all, dtype=torch.float32)

# ---------------------------------------------------------
# LSTM MODEL
# ---------------------------------------------------------
class MouseLSTM(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=64, num_layers=1):
        super().__init__()

        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        out, _ = self.lstm(x)
        last_out = out[:, -1, :]
        pred = self.fc(last_out)
        return pred
    

 # ---------------------------------------------------------
# VISUALIZATION FUNCTION (must be OUTSIDE the class)
# ---------------------------------------------------------
def visualize_predictions(actual_seq, pred_seq, title="LSTM Prediction vs Actual"):
    actual_seq = actual_seq.copy()
    pred_seq = pred_seq.copy()

    plt.figure(figsize=(12, 5))

    # --- 1. dx/dy time series ---
    plt.subplot(1, 2, 1)
    plt.plot(actual_seq[:,0], label="Actual dx", alpha=0.8)
    plt.plot(pred_seq[:,0], label="Pred dx", alpha=0.8)
    plt.plot(actual_seq[:,1], label="Actual dy", alpha=0.8)
    plt.plot(pred_seq[:,1], label="Pred dy", alpha=0.8)
    plt.title("dx/dy Over Time")
    plt.legend()

    # --- 2. reconstruct absolute positions ---
    actual_pos = np.cumsum(actual_seq, axis=0)
    pred_pos = np.cumsum(pred_seq, axis=0)

    plt.subplot(1, 2, 2)
    plt.plot(actual_pos[:,0], actual_pos[:,1], label="Actual Trajectory", linewidth=2)
    plt.plot(pred_pos[:,0], pred_pos[:,1], label="Predicted Trajectory", linewidth=2)
    plt.title("Trajectory Comparison")
    plt.legend()
    plt.gca().invert_yaxis()

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
# ---------------------------------------------------------
# MAIN
# ---------------------------------------------------------
if __name__ == "__main__":

    # 1️⃣ Load slides
    slides = load_all_slides(DATA_FOLDER)

    # 2️⃣ Split slides
    train_slides, test_slides = split_slides(slides, TEST_RATIO)

    # 3️⃣ Create sliding windows
    X_train, Y_train = create_windows_from_slides(train_slides, WINDOW_SIZE)
    X_test, Y_test = create_windows_from_slides(test_slides, WINDOW_SIZE)

    print("Train windows:", X_train.shape)
    print("Test windows:", X_test.shape)

    # 4️⃣ Normalize (IMPORTANT: use train stats only)
    mean = X_train.mean(dim=(0,1))
    std = X_train.std(dim=(0,1)) + 1e-6

    X_train = (X_train - mean) / std
    Y_train = (Y_train - mean) / std

    X_test = (X_test - mean) / std
    Y_test = (Y_test - mean) / std

    # 5️⃣ DataLoader
    train_loader = DataLoader(
        TensorDataset(X_train, Y_train),
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test, Y_test),
        batch_size=BATCH_SIZE
    )

    # 6️⃣ Initialize model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MouseLSTM().to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # 7️⃣ Training loop
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)

            pred = model(xb)
            loss = criterion(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS}, Train Loss: {total_loss/len(train_loader):.6f}")

    # 8️⃣ Evaluation
    model.eval()
    total_test_loss = 0

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            loss = criterion(pred, yb)
            total_test_loss += loss.item()

    print(f"\nFinal Test MSE: {total_test_loss/len(test_loader):.6f}")

 # 8️⃣ Evaluation + Collect Predictions
model.eval()
all_preds = []
all_actual = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)
        pred = model(xb).cpu().numpy()
        all_preds.append(pred)
        all_actual.append(yb.numpy())

# Convert to arrays
preds = np.concatenate(all_preds)
actual = np.concatenate(all_actual)

# Compute test MSE
mse = np.mean((preds - actual)**2)
print(f"\nFinal Test MSE: {mse:.6f}")

# 9️⃣ Denormalize for visualization
actual_denorm = actual * std.numpy() + mean.numpy()
preds_denorm = preds * std.numpy() + mean.numpy()

# 1–2 slide worth of predictions (e.g., first 300)
visualize_predictions(actual_denorm[:300], preds_denorm[:300])
