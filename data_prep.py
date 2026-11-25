"""data_prep.py — Build the leakage-free input matrix from the Mendeley dataset.

Audit rule (Appendix Table A1): keep only the 7 independent predictors
{Gender, Age, University, Department, Academic Year, Current CGPA, Scholarship};
exclude all scale items, subscale scores, category labels, and the two
derived variables. One-hot encode -> 69 dimensions.

Input : Mendeley CSV (3,020 x 41), downloaded from
        https://doi.org/10.17632/vwxrgx78x4.1
Output: data/processed.csv  (columns: y, feat_0 ... feat_68)
Usage : python data_prep.py <path_to_mendeley_csv>
"""
import sys, pandas as pd

KEEP = ["Gender", "Age", "University", "Department",
        "Academic Year", "Current CGPA", "Scholarship"]
LABEL = "Mental Health Level"

df = pd.read_csv(sys.argv[1])
assert len(df) == 3020 and df.shape[1] == 41, "unexpected raw shape"

y = (df[LABEL] != "Low").astype(int)          # Medium+Severe -> 1 (at risk)
X = pd.get_dummies(df[KEEP], columns=[c for c in KEEP if c != "Age"])
X["Age"] = df["Age"].astype(float)
X["Age"] = (X["Age"] - X["Age"].min()) / (X["Age"].max() - X["Age"].min())  # min-max
X = X[sorted(X.columns)]
assert X.shape[1] == 69, f"expected 69 dims, got {X.shape[1]}"

out = pd.concat([y.rename("y"), X.reset_index(drop=True)], axis=1)
out.to_csv("data/processed.csv", index=False)
print("saved data/processed.csv", out.shape,
      "| positives:", int(y.sum()), f"({y.mean():.4f})")
