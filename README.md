# Student Mental-Health Risk Prediction from Academic and Demographic Data: A Label-Leakage-Aware CNN Evaluation

## Contents

| File | Description |
| `code/` | Training and evaluation code for all nine models (AM-CNN, PM-CNN, standard CNN, Transformer, MLP, LR, RF, XGBoost, LightGBM), the ablation configurations, and the label-leakage case study 
| `seeds/` | Random seeds used for data splitting, training and all stochastic procedures 
| `splits/` | Exact train/validation/test partition files of the 10 x 5 repeated stratified cross-validation (one file per repetition) 
| `Table_S1_per_fold_results.csv` | Per-fold prediction counts (TP/FN/FP/TN) and metrics (ACC/P/R/F1/AUC) for all nine models across the 50 folds

## Datasets

- Main dataset: Turzo S S, Islam K M A, Naz M, et al. Mental health dataset of private university students in Bangladesh under academic pressure (Version 1). Mendeley Data, 2026. https://doi.org/10.17632/vwxrgx78x4.1
- Case-study dataset: Salamat A. Mental health of students dataset. Kaggle. https://www.kaggle.com/datasets/aminasalamat/mental-health-of-students-dataset/data

Both datasets are publicly available and de-identified; no new data were collected.
