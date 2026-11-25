"""make_splits.py — Deterministic 10 x 5 repeated stratified splits.

Each repetition r uses seed = 1000 + r. For every repetition, each sample
gets exactly one fold assignment; folds are stratified on y (394 positive /
210 negative per fold). Splits are saved to splits/rep{r}.csv with columns
sample_id, fold (0..4), y. These files make every fold assignment in the
study reproducible.

Usage : python make_splits.py
"""
import os, numpy as np, pandas as pd

N_REPS, N_FOLDS = 10, 5
df = pd.read_csv("data/processed.csv")
y = df["y"].values
os.makedirs("splits", exist_ok=True)

for rep in range(N_REPS):
    rng = np.random.default_rng(1000 + rep)
    fold = np.empty(len(y), dtype=int)
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        fold[idx] = np.arange(len(idx)) % N_FOLDS
    pd.DataFrame({"sample_id": np.arange(len(y)), "fold": fold, "y": y}
                 ).to_csv(f"splits/rep{rep}.csv", index=False)

    # sanity: per-fold class counts identical across folds (stratified)
    chk = pd.Series(fold).groupby([fold, y]).size().unstack()
    assert chk.nunique().max() == 1, f"rep{rep} not stratified"
print("saved splits/rep0.csv ... rep9.csv; seeds = 1000..1009")
