# Predicting Customer Advertisement Response Using a Multilayer Feed-Forward Neural Network

CMSC 191 — Special Topics in Computer Science / Introduction to Neural Computing

---

## Overview

This repository contains the implementation for a research project that predicts whether a customer will respond to a shop advertisement campaign. The model used is a multilayer feed-forward neural network (MLP) trained on customer demographic and behavioral data.

The dataset is publicly available and contains 2,240 customer records with 29 original columns. The target variable is `Response` (1 = responded, 0 = did not respond), with a class distribution of roughly 85% non-responders and 15% responders.

---

## Repository Structure

```
CMSC 191/
├── advertisement_response_nn.py           # Main script
├── Predicting the Result of a Shop's Advertisement.csv   # Dataset (place here)
├── README.md
└── outputs/
    └── YYYYMMDD_HHMMSS/                   # One folder per run
        ├── summary.txt
        ├── model_metrics.csv
        ├── model_comparison.csv
        ├── threshold_tuning_results.csv
        ├── test_predictions.csv
        ├── confusion_matrix.png
        ├── roc_curve.png
        ├── precision_recall_curve.png
        ├── feature_importance.csv
        ├── feature_importance.png
        └── advertisement_response_mlp_model.pkl
```

---

## Requirements

Python 3.8 or higher. Install dependencies:

```
pip install numpy pandas matplotlib scikit-learn joblib
```

---

## How to Run (Windows PowerShell)

Place the dataset file in the same folder as the script, then run:

```powershell
cd "C:\Users\carlt\Desktop\CMSC 191"
python .\advertisement_response_nn.py
```

Each run creates a new timestamped subfolder under `outputs/` so previous results are not overwritten.

---

## Methodology

**Feature engineering**

Eight features are derived from the raw columns before modeling:

| Feature | Description |
|---|---|
| `Customer_Tenure_Days` | Days since the customer's earliest recorded enrollment date |
| `Age` | Estimated from `Year_Birth` and enrollment year |
| `Total_Spending` | Sum of spending across all product categories |
| `Total_Purchases` | Sum of purchases across all channels |
| `Total_Accepted_Campaigns` | Number of previous campaigns the customer accepted |
| `Total_Children` | Sum of `Kidhome` and `Teenhome` |
| `Spending_to_Income_Ratio` | `Total_Spending / (Income + 1)` |
| `Purchases_per_Recency` | `Total_Purchases / (Recency + 1)` |

**Preprocessing**

Numerical features are median-imputed and standard-scaled. Categorical features (`Education`, `Marital_Status`) are mode-imputed and one-hot encoded. Both steps are done inside a scikit-learn pipeline to prevent any leakage between splits.

**Train / validation / test split**

The dataset is split 60% train / 20% validation / 20% test using stratified sampling to preserve the class ratio in each subset.

**Handling class imbalance**

The minority class (Response = 1) is oversampled using random replication, but only in the training set. The validation and test sets keep the original class distribution. Oversampling before the split would allow augmented copies of validation and test samples to appear in training, which inflates performance metrics — this is avoided here.

**Model**

A multilayer perceptron with two hidden layers (32 units, 16 units), ReLU activation, Adam optimizer, and early stopping. The full configuration is listed below.

| Parameter | Value |
|---|---|
| Hidden layers | (32, 16) |
| Activation | ReLU |
| Optimizer | Adam |
| L2 penalty (alpha) | 0.001 |
| Learning rate | 0.001 |
| Max iterations | 1000 |
| Early stopping | Enabled |
| Early stopping patience | 40 iterations |
| Validation fraction (internal) | 20% |

**Threshold tuning**

Because the dataset is imbalanced, the default classification threshold of 0.50 may not give the best balance between precision and recall for the minority response class. The threshold is swept from 0.10 to 0.90 on the validation set and the value that maximizes the validation F1-score is selected. The test set is evaluated only once, using this threshold.

---

## Results

**Final model — test set (threshold = 0.89)**

| Metric | Value |
|---|---|
| Accuracy | 0.8725 |
| Precision | 0.5818 |
| Recall | 0.4848 |
| F1-score | 0.5289 |
| ROC-AUC | 0.8859 |
| Mean 5-fold CV F1 | 0.5057 |

**Confusion matrix**

```
                Predicted No  Predicted Yes
Actual No             358            23
Actual Yes             34            32
```

Comparison against a majority-class baseline and a default-threshold MLP (0.50) is computed at runtime and saved to `model_comparison.csv`.

---

## How to Interpret the Results

- **ROC-AUC of 0.8859** means the model ranks a randomly chosen responder above a randomly chosen non-responder about 89% of the time. This is the strongest result and reflects that the model's probability scores are well-calibrated.
- **Precision of 0.5818** means that when the model predicts a customer will respond, it is correct about 58% of the time.
- **Recall of 0.4848** means the model correctly identifies about 48% of actual responders. Roughly half of true responders are missed.
- **F1-score of 0.5289** reflects the tradeoff between precision and recall. It is moderate, which is expected for an imbalanced binary classification problem.
- **Threshold = 0.89** is high, which means the model is conservative — it only flags a customer as a likely responder when it is fairly confident. Lowering the threshold would catch more responders but increase false positives.
- The **confusion matrix** shows 32 true positives and 34 false negatives out of 66 actual responders in the test set, and 23 false positives out of 381 non-responders.

---

## Limitations

- **Class imbalance.** Even with oversampling, recall for the minority class is moderate. Alternative strategies such as SMOTE or cost-sensitive learning were not explored.
- **No hyperparameter search.** The model architecture and training parameters were chosen manually. A grid search or random search might find a better configuration.
- **Oversampling method.** Random replication of the minority class does not add new information — it only duplicates existing samples. More sophisticated augmentation methods could help.
- **Single dataset.** The model has not been evaluated on data from a different campaign or time period, so its generalizability is unknown.
- **Threshold selection.** The threshold was tuned to maximize F1-score on the validation set, which reflects an equal weighting of precision and recall. A different objective (such as maximizing recall to reduce missed customers) would produce a different threshold.
- **Interpretability.** MLP models are not directly interpretable. Feature importances here are estimated using permutation importance, which is an approximation.

---

## Output Files

Each run writes to `outputs/YYYYMMDD_HHMMSS/`:

| File | Description |
|---|---|
| `summary.txt` | Full run summary: data sizes, model config, all metrics |
| `model_metrics.csv` | Final test metrics for the tuned MLP |
| `model_comparison.csv` | Majority baseline, default MLP, and tuned MLP side by side |
| `threshold_tuning_results.csv` | Precision, recall, F1, TP/FP/FN/TN at each threshold |
| `test_predictions.csv` | Per-sample predicted probability and predicted label for test set |
| `confusion_matrix.png` | Confusion matrix heatmap |
| `roc_curve.png` | ROC curve with AUC |
| `precision_recall_curve.png` | Precision-recall curve with AUC |
| `feature_importance.csv` | Permutation importance scores for each input feature |
| `feature_importance.png` | Bar chart of the top 10 features |
| `advertisement_response_mlp_model.pkl` | Serialized trained model (joblib) |
