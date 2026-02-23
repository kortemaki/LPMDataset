import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from shared import *
from shared import evaluate_mouse

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
    train_model(model, train_loader)
# --- Evaluate ---
    rmse, nrmse, acc2pct = evaluate_mouse(
    model,
    test_loader,
    mean,
    std,
    screen_width=1200   # adjust if needed
)
    print("Mouse→Mouse RMSE:", rmse)
    print("Mouse→Mouse NRMSE:", nrmse)
    print("Mouse→Mouse Accuracy:", acc2pct)