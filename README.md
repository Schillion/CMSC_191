# Advertisement Response Prediction using a Multilayer Perceptron

CMSC 191 — Neural Computing

## Overview

This project predicts whether a customer will respond to a shop advertisement using a multilayer feed-forward neural network (MLP). The model is trained on customer demographic and behavioral data from a marketing campaign dataset with 2,240 records.

The main research question is: given a customer's profile (age, income, spending habits, campaign history), can a neural network identify who is likely to respond to a new advertisement?

## Requirements

Python 3.8 or higher. Install dependencies with:

```
pip install numpy pandas matplotlib scikit-learn joblib
```

## Setup

Place the dataset file in the same folder as the script:

```
CMSC 191/
    advertisement_response_nn.py
    Predicting the Result of a Shop's Advertisement.csv
```

## How to Run

```
python advertisement_response_nn.py
```

Each run creates a new timestamped folder under `outputs/` so previous results are never overwritten.

## Dataset

- 2,240 customer records, 29 original columns
- Target column: `Response` (1 = responded to advertisement, 0 = did not)
- Class distribution: ~85% non-responders, ~15% responders (imbalanced)

## Methodology

1. **Feature engineering** — derives Customer_Tenure_Days, Age, Total_Spending, Total_Purchases, Total_Accepted_Campaigns, Total_Children, Spending_to_Income_Ratio, Purchases_per_Recency from the raw columns
2. **Stratified split** — 60% train / 20% validation / 20% test, preserving class ratios
3. **Oversampling** — minority class (Response=1) is upsampled only in the training set to avoid data leakage
4. **Preprocessing** — median imputation and standard scaling for numerical features; mode imputation and one-hot encoding for categorical features
5. **MLP training** — two hidden layers (32, 16), ReLU activation, Adam optimizer, early stopping
6. **Threshold tuning** — classification threshold swept from 0.10 to 0.90 on the validation set; best threshold chosen by F1-score
7. **Final evaluation** — test set evaluated once with the tuned threshold

## Model Architecture

| Parameter | Value |
|---|---|
| Hidden layers | (32, 16) |
| Activation | ReLU |
| Optimizer | Adam |
| Alpha (L2 penalty) | 0.001 |
| Learning rate | 0.001 |
| Early stopping | Enabled (patience = 40) |
| Max iterations | 1000 |

## Results

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC |
|---|---|---|---|---|---|
| Majority class baseline | — | — | — | 0.0 | 0.50 |
| MLP (threshold = 0.50) | — | — | — | — | — |
| MLP (tuned threshold = 0.89) | 0.8725 | 0.5818 | 0.4848 | 0.5289 | 0.8859 |

Baseline and default-threshold results are computed at runtime and saved to `model_comparison.csv`.

## Output Files

Each run produces the following inside `outputs/<timestamp>/`:

| File | Description |
|---|---|
| `summary.txt` | Full run summary including all metrics |
| `model_metrics.csv` | Final model performance metrics |
| `model_comparison.csv` | Majority baseline vs default MLP vs tuned MLP |
| `threshold_tuning_results.csv` | Precision, recall, F1 at each threshold (0.10–0.90) |
| `test_predictions.csv` | Per-sample predictions on the test set |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC |
| `precision_recall_curve.png` | Precision-recall curve with AUC |
| `feature_importance.csv` | Permutation-based feature importance scores |
| `feature_importance.png` | Top 10 features bar chart |
| `advertisement_response_mlp_model.pkl` | Saved trained model |
