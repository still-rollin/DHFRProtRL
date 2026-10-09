"""Run from the repo root (CPU, about 20 minutes per split).
How well does the DHFROracle architecture generalize when trained from scratch on part of the data?
Compares with the shipped checkpoint's Pearson 0.935 on all 4,161 variants.  Two splits:
  random   : 80/20 over variants
  position : hold out 20% of positions entirely (all 19 substitutions at those positions)
"""
import sys, time, collections, numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, ".")
from net.rew import DHFROracle
torch.manual_seed(0); np.random.seed(0)
A = "ACDEFGHIKLMNPQRSTVWY"; IX = {a: i for i, a in enumerate(A)}
d = pd.read_csv("data/DHFR/medium.csv")
D = torch.tensor([[IX[c] for c in s] for s in d.sequence]); y = torch.tensor(d.target.values, dtype=torch.float32)
ref = np.array([collections.Counter(D[:, i].tolist()).most_common(1)[0][0] for i in range(D.shape[1])])
pos = np.array([int(np.where(r.numpy() != ref)[0][0]) for r in D])

def fit(tr, te, epochs=150, tag=""):
    idx = np.random.permutation(tr); nv = len(idx) // 10; va, tr = idx[:nv], idx[nv:]
    m = DHFROracle(n_tokens=20, make_one_hot=True); opt = torch.optim.Adam(m.parameters(), 3e-3); lossf = nn.MSELoss()
    best, bs, bad = 1e9, None, 0; t0 = time.time()
    for ep in range(epochs):
        m.train(); perm = np.random.permutation(tr)
        for i in range(0, len(perm), 64):
            b = perm[i:i + 64]; opt.zero_grad(); l = lossf(m(D[b]), y[b]); l.backward(); opt.step()
        m.eval()
        with torch.no_grad(): vl = lossf(m(D[va]), y[va]).item()
        if vl < best: best, bs, bad = vl, {k: v.clone() for k, v in m.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= 25: break
    m.load_state_dict(bs); m.eval()
    with torch.no_grad(): ptr = m(D[tr]).numpy(); pte = m(D[te]).numpy()
    print(f"[{tag}] epochs={ep + 1} time={time.time() - t0:.0f}s  TRAIN pearson={pearsonr(ptr, y[tr].numpy())[0]:.3f}  HELD-OUT pearson={pearsonr(pte, y[te].numpy())[0]:.3f} spearman={spearmanr(pte, y[te].numpy())[0]:.3f}", flush=True)

n = len(y); perm = np.random.permutation(n); te = perm[: n // 5]; tr = perm[n // 5:]
fit(tr, te, tag="random 80/20")
P = np.random.permutation(219); hold = set(P[:44].tolist()); mask = np.array([p in hold for p in pos])
fit(np.where(~mask)[0], np.where(mask)[0], tag="position-held-out")
print("shipped checkpoint, all 4,161 variants: pearson 0.935", flush=True)
