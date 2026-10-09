import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from sklearn.model_selection import train_test_split
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error
import os
import json


class LengthMaxPool1D(nn.Module):
    def __init__(self, in_dim, out_dim, activation='relu', linear=True):
        super(LengthMaxPool1D, self).__init__()
        self.linear = linear
        if linear:
            self.fc = nn.Linear(in_dim, out_dim)
        else:
            self.out_dim = out_dim

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        else:
            self.activation = nn.Identity()

    def forward(self, x):
        x, _ = torch.max(x, dim=1)
        if self.linear:
            x = self.fc(x)
        x = self.activation(x)
        return x


class BaseCNN(nn.Module):
    def __init__(
            self,
            n_tokens: int = 20,
            kernel_size: int = 5,
            input_size: int = 256,
            dropout: float = 0.0,
            make_one_hot=True,
            activation: str = 'relu',
            linear: bool=True,
            **kwargs):
        super(BaseCNN, self).__init__()
        self.encoder = nn.Conv1d(n_tokens, input_size, kernel_size=kernel_size, padding=kernel_size//2)
        self.embedding = LengthMaxPool1D(
            linear=linear,
            in_dim=input_size,
            out_dim=input_size*2,
            activation=activation,
        )
        self.decoder = nn.Linear(input_size*2, 1)
        self.n_tokens = n_tokens
        self.dropout = nn.Dropout(dropout)
        self.input_size = input_size
        self._make_one_hot = make_one_hot

    def get_embed(self, x):
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        return x

    def forward(self, x, get_embed=False):
        if self._make_one_hot:
            x = F.one_hot(x.long(), num_classes=self.n_tokens)
        x = x.permute(0, 2, 1).float()
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        output = self.decoder(x).squeeze(1)
        if get_embed:
            return x, output
        return output


class DHFRBaseCNN(BaseCNN):
    def __init__(self, seq_len=None, **kwargs):
        # DHFR-specific defaults
        defaults = {
            'n_tokens': 20,          # 20 amino acids
            'kernel_size': 5,        # Slightly larger than original for better patterns
            'input_size': 128,       # Start with reasonable size
            'dropout': 0.2,          # Add regularization
            'make_one_hot': False,   # We'll handle encoding separately
            'activation': 'relu',
            'linear': True
        }
        defaults.update(kwargs)
        super(DHFRBaseCNN, self).__init__(**defaults)
        self.seq_len = seq_len

    def forward(self, x, get_embed=False):
        # x should already be one-hot encoded as [batch, channels, length]
        x = self.encoder(x).permute(0, 2, 1)
        x = self.dropout(x)
        x = self.embedding(x)
        output = self.decoder(x).squeeze(-1)
        if get_embed:
            return x, output
        return output


class DHFRDataset(Dataset):
    def __init__(self, df, seq_len):
        self.seqs = df['sequence'].values
        self.targets = df['target'].values.astype(np.float32)
        self.seq_len = seq_len

        # Amino acid mapping
        self.AA = "ACDEFGHIKLMNPQRSTVWY"
        self.aa2idx = {aa: i for i, aa in enumerate(self.AA)}

    def seq_to_onehot(self, seq):
        arr = np.zeros((20, self.seq_len), dtype=np.float32)
        seq = seq.strip()[:self.seq_len]
        for i, aa in enumerate(seq):
            idx = self.aa2idx.get(aa, None)
            if idx is not None:
                arr[idx, i] = 1.0
        return arr

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        x = self.seq_to_onehot(self.seqs[idx])
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return x, y


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


def train_dhfr_basecnn(model, train_loader, val_loader, device, epochs=300, lr=1e-3, wd=1e-5, patience=10):
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

        # Validation
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
            if patience_counter >= patience:
                print(f"Early stopping at epoch {ep+1}")
                break

    model.load_state_dict(best_state)
    return model


def predict_fitness_basecnn(model, sequence, seq_len, device):
    """Predict fitness for a single sequence"""
    AA = "ACDEFGHIKLMNPQRSTVWY"
    aa2idx = {aa: i for i, aa in enumerate(AA)}

    # Convert to one-hot
    arr = np.zeros((20, seq_len), dtype=np.float32)
    seq = sequence.strip()[:seq_len]
    for i, aa in enumerate(seq):
        idx = aa2idx.get(aa, None)
        if idx is not None:
            arr[idx, i] = 1.0

    x = torch.tensor(arr, dtype=torch.float32).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        pred = model(x).item()
    return pred


def main():
    csv_path = os.path.expanduser("data/DHFR/medium.csv")
    df = pd.read_csv(csv_path)
    seq_len = df['sequence'].str.len().max()

    # Split data
    train_df, test_df = train_test_split(df, test_size=0.15, random_state=42)
    train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=42)

    # Create datasets
    train_ds = DHFRDataset(train_df, seq_len)
    val_ds = DHFRDataset(val_df, seq_len)
    test_ds = DHFRDataset(test_df, seq_len)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=64)
    test_loader = DataLoader(test_ds, batch_size=64)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Initialize DHFR-specific BaseCNN
    model = DHFRBaseCNN(
        seq_len=seq_len,
        input_size=128,
        dropout=0.2,
        kernel_size=5
    )

    # Train model
    model = train_dhfr_basecnn(model, train_loader, val_loader, device)

    # Test evaluation
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            preds = model(xb)
            y_true.extend(yb.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    test_metrics = compute_metrics(y_true, y_pred)
    test_metrics["MAE"] = mean_absolute_error(y_true, y_pred)
    print("Test metrics:", test_metrics)

    # Save model
    os.makedirs("./model", exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'seq_len': seq_len,
        'model_config': {
            'input_size': 128,
            'dropout': 0.2,
            'kernel_size': 5
        }
    }, "./model/dhfr_basecnn_oracle.pt")

    # Save amino acid mapping
    AA = "ACDEFGHIKLMNPQRSTVWY"
    aa2idx = {aa: i for i, aa in enumerate(AA)}
    with open("./model/aa2idx_basecnn.json", "w") as f:
        json.dump(aa2idx, f)


if __name__ == "__main__":
    main()