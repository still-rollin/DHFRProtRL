import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.metrics import mean_absolute_error


# -----------------------------
# Sequence encoding
# -----------------------------
AA = "ACDEFGHIKLMNPQRSTVWY"
aa2idx = {aa:i for i,aa in enumerate(AA)}

def seq_to_onehot(seq, L):
    arr = np.zeros((20, L), dtype=np.float32)
    seq = seq.strip()[:L]
    for i, aa in enumerate(seq):
        idx = aa2idx.get(aa, None)
        if idx is not None:
            arr[idx, i] = 1.0
    return arr

# -----------------------------
# Dataset class
# -----------------------------
class DHFRDataset(Dataset):
    def __init__(self, df, L):
        self.seqs = df['sequence'].values
        self.targets = df['target'].values.astype(np.float32)
        self.L = L
    def __len__(self):
        return len(self.seqs)
    def __getitem__(self, idx):
        x = seq_to_onehot(self.seqs[idx], self.L)
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y

# -----------------------------
# CNN model
# -----------------------------
class CNN1D(nn.Module):
    def __init__(self, L):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(20, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.fc(self.conv(x)).squeeze(-1)

# -----------------------------
# Metrics
# -----------------------------
def compute_metrics(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    mse = np.mean((y_true - y_pred)**2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    r2 = 1 - np.sum((y_true - y_pred)**2)/np.sum((y_true - y_true.mean())**2)
    pearson = np.corrcoef(y_true, y_pred)[0,1]
    spearman = spearmanr(y_true, y_pred).correlation
    return dict(mse=mse, rmse=rmse, mae=mae, r2=r2, pearson=pearson, spearman=spearman)

# -----------------------------
# Training loop
# -----------------------------
def train_model(model, train_loader, val_loader, device, epochs=300, lr=1e-3, wd=1e-5, patience=10):
    model.to(device)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    criterion = nn.MSELoss()
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None

    for ep in range(epochs):
        model.train()
        train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            opt.step()
            train_loss += loss.item() * xb.size(0)
        train_loss /= len(train_loader.dataset)

        # validation
        model.eval()
        val_loss = 0
        y_true, y_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = model(xb)
                val_loss += F.mse_loss(preds, yb, reduction='sum').item()
                y_true.extend(yb.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
        val_loss /= len(val_loader.dataset)
        metrics = compute_metrics(y_true, y_pred)
        print(f"Epoch {ep+1}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, R2={metrics['r2']:.3f}, Spearman={metrics['spearman']:.3f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = model.state_dict()
            patience_counter = 0
        else:
            patience_counter += 1
     

    model.load_state_dict(best_state)
    return model

# -----------------------------
# Main
# -----------------------------
def main():
    csv_path = os.path.expanduser("data/DHFR/medium.csv")  # replace with your CSV filename
    df = pd.read_csv(csv_path)
    L = df['sequence'].str.len().max()

    # Split
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

    train_ds = DHFRDataset(train_df, L)
    val_ds = DHFRDataset(val_df, L)
    test_ds = DHFRDataset(test_df, L)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNN1D(L)

    model = train_model(model, train_loader, val_loader, device)

    # Test metrics

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_metrics = compute_metrics(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    test_metrics["MAE"] = mae

    print("Test metrics:", test_metrics)

    # Save model and mapping
    os.makedirs("./model", exist_ok=True)
    torch.save({'model_state_dict': model.state_dict(), 'seq_len': L}, "./model/dhfr_oracle.pt")
    import json
    with open("./model/aa2idx.json", "w") as f:
        json.dump(aa2idx, f)

if __name__ == "__main__":
    main()
