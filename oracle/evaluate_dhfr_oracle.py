import torch
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from dhfr_basecnn import DHFRBaseCNN, DHFRDataset, compute_metrics
from torch.utils.data import DataLoader

# ========== CONFIG ==========
MODEL_PATH = "./model/dhfr_oracle.ckpt"
CSV_PATH = "./data/dhfr.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# =============================


# ----- Load Checkpoint -----
print(f"Loading checkpoint from: {MODEL_PATH}")
checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)

# Get model config + seq length
seq_len = checkpoint.get("seq_len", 237)
model_config = checkpoint.get("model_config", {
    "input_size": 128,
    "dropout": 0.2,
    "kernel_size": 5
})

# Recreate model
model = DHFRBaseCNN(seq_len=seq_len, **model_config)
model.load_state_dict(checkpoint["model_state_dict"])
model.to(DEVICE)
model.eval()

print("\nModel loaded successfully!")
print(f"Sequence length: {seq_len}")
print(f"Model config: {model_config}")


# ----- Load Test Data -----
print("\nLoading test data from CSV...")
df = pd.read_csv(CSV_PATH)
df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
split1 = int(0.85 * len(df))
split2 = int(0.85 * split1)
test_df = df[split1:]  # last 15%

test_ds = DHFRDataset(test_df, seq_len)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

print(f"Test samples: {len(test_ds)}")


# ----- Evaluate -----
y_true, y_pred = [], []
with torch.no_grad():
    for xb, yb in test_loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        preds = model(xb)
        y_true.extend(yb.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

metrics = compute_metrics(y_true, y_pred)
print("\nEvaluation Metrics:")
for k, v in metrics.items():
    print(f"{k:>10}: {v:.4f}")

# ----- Plot predicted vs actual fitness -----
plt.figure(figsize=(6,6))
plt.scatter(y_true, y_pred, alpha=0.6)
plt.xlabel("True Fitness")
plt.ylabel("Predicted Fitness")
plt.title("DHFR CNN Predicted vs True Fitness")
plt.grid(True)
plt.tight_layout()
plt.show()
