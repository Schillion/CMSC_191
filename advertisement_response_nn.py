import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


warnings.filterwarnings("ignore", category=ConvergenceWarning)

# A single seed controls all randomness so the script produces the same
# results on every run — important for a research paper.
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

BASE_DIR = Path(__file__).resolve().parent

# Each run writes to its own timestamped folder so previous results are
# never overwritten. Makes it easy to compare runs.
RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = BASE_DIR / "outputs" / RUN_TIMESTAMP
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Output folder:", OUTPUT_DIR)


# --- Load dataset ---

def find_dataset(base_dir):
    for filename in [
        "Predicting the Result of a Shop's Advertisement.csv",
        "Predicting the Result of a Shop's Advertisement.xlsx",
        "Predicting the Result of a Shop's Advertisement.xls",
    ]:
        path = base_dir / filename
        if path.exists():
            return path

    # Fallback: pick the first spreadsheet in the folder
    candidates = (
        list(base_dir.glob("*.csv"))
        + list(base_dir.glob("*.xlsx"))
        + list(base_dir.glob("*.xls"))
    )
    if candidates:
        return candidates[0]

    raise FileNotFoundError(
        "Dataset not found. Place the CSV file in the same folder as this script."
    )


def load_dataset(path):
    # The dataset is sometimes tab-separated; try that first and fall back to
    # comma-separated if only one column comes out.
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(path)
        return df
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    raise ValueError("Unsupported file format.")


dataset_path = find_dataset(BASE_DIR)
df = load_dataset(dataset_path)

if "Response" not in df.columns:
    raise ValueError("Column 'Response' not found. Check that the dataset is correct.")

print("Dataset loaded:", dataset_path.name)
print("Shape:", df.shape)
print("\nTarget distribution:")
print(df["Response"].value_counts())
print("\nTarget distribution (%):")
print(df["Response"].value_counts(normalize=True).mul(100).round(2))


# --- Feature engineering ---
# Raw columns like Year_Birth and Dt_Customer are less informative than derived
# features. Aggregates like Total_Spending capture overall behavior in one number,
# which helps the MLP learn faster with fewer parameters.

data = df.copy()

if "Dt_Customer" in data.columns:
    data["Dt_Customer"] = pd.to_datetime(data["Dt_Customer"], dayfirst=True, errors="coerce")
    # Days since earliest recorded enrollment; higher = longer relationship with shop.
    data["Customer_Tenure_Days"] = (data["Dt_Customer"].max() - data["Dt_Customer"]).dt.days

if "Year_Birth" in data.columns and "Dt_Customer" in data.columns:
    data["Age"] = data["Dt_Customer"].dt.year - data["Year_Birth"]
elif "Year_Birth" in data.columns:
    data["Age"] = 2026 - data["Year_Birth"]

if "Age" in data.columns:
    # Drop rows with unrealistic ages — likely data entry errors.
    data = data[(data["Age"] >= 18) & (data["Age"] <= 100)]

spending_cols  = ["MntWines", "MntFruits", "MntMeatProducts", "MntFishProducts", "MntSweetProducts", "MntGoldProds"]
purchase_cols  = ["NumDealsPurchases", "NumWebPurchases", "NumCatalogPurchases", "NumStorePurchases"]
campaign_cols  = ["AcceptedCmp1", "AcceptedCmp2", "AcceptedCmp3", "AcceptedCmp4", "AcceptedCmp5"]

if all(c in data.columns for c in spending_cols):
    data["Total_Spending"] = data[spending_cols].sum(axis=1)

if all(c in data.columns for c in purchase_cols):
    data["Total_Purchases"] = data[purchase_cols].sum(axis=1)

if all(c in data.columns for c in campaign_cols):
    data["Total_Accepted_Campaigns"] = data[campaign_cols].sum(axis=1)

if "Kidhome" in data.columns and "Teenhome" in data.columns:
    data["Total_Children"] = data["Kidhome"] + data["Teenhome"]

if "Income" in data.columns and "Total_Spending" in data.columns:
    # +1 prevents division by zero for customers with no recorded income.
    data["Spending_to_Income_Ratio"] = data["Total_Spending"] / (data["Income"] + 1)

if "Recency" in data.columns and "Total_Purchases" in data.columns:
    data["Purchases_per_Recency"] = data["Total_Purchases"] / (data["Recency"] + 1)


# --- Prepare features and target ---

TARGET = "Response"
drop_cols = ["ID", "Dt_Customer", "Year_Birth", "Z_CostContact", "Z_Revenue", TARGET]
drop_cols = [c for c in drop_cols if c in data.columns]

X = data.drop(columns=drop_cols)
y = data[TARGET].astype(int)

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features   = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nTotal features:", X.shape[1])
print("Categorical:", categorical_features)
print("Numerical:", len(numerical_features), "columns")


# --- Train / validation / test split ---
# Stratified split preserves the original class ratio in every subset so the
# minority class (Response=1) is represented in all three sets.
# 60% train | 20% validation | 20% test

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=RANDOM_SEED, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_SEED, stratify=y_temp
)

print(f"\nSplit — Train: {X_train.shape[0]} | Val: {X_val.shape[0]} | Test: {X_test.shape[0]}")


# --- Oversample minority class in training set only ---
# Oversampling before the split would leak augmented copies of validation/test
# samples into training and inflate performance metrics. We balance only the
# training data after the split is done.

def oversample_minority(X_input, y_input):
    df_tmp = X_input.copy()
    df_tmp[TARGET] = y_input.values

    majority  = df_tmp[df_tmp[TARGET] == 0]
    minority  = df_tmp[df_tmp[TARGET] == 1]

    minority_up = minority.sample(n=len(majority), replace=True, random_state=RANDOM_SEED)
    balanced    = pd.concat([majority, minority_up]).sample(frac=1, random_state=RANDOM_SEED)

    return balanced.drop(columns=[TARGET]), balanced[TARGET].astype(int)


X_train_bal, y_train_bal = oversample_minority(X_train, y_train)

print("\nOriginal training distribution:")
print(y_train.value_counts())
print("\nBalanced training distribution:")
print(y_train_bal.value_counts())


# --- Preprocessing pipeline ---
# Numerical: median imputation is more robust to outliers than mean imputation.
#            StandardScaler normalizes the range so no single feature dominates.
# Categorical: mode imputation, then one-hot encoding.
# Using a pipeline ensures the same transformations are applied consistently
# to train, validation, and test data without manual re-fitting.

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot",  OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", num_pipeline, numerical_features),
    ("cat", cat_pipeline, categorical_features),
])


# --- MLP model ---
# Two hidden layers (32, 16) with ReLU activation. Smaller second layer acts
# as a bottleneck that forces the model to learn compact representations.
# Early stopping monitors a 20% internal validation split and stops training
# when improvement stalls for 40 consecutive iterations — prevents overfitting.

mlp_params = dict(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.20,
    n_iter_no_change=40,
    random_state=RANDOM_SEED,
)

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   MLPClassifier(**mlp_params)),
])

model.fit(X_train_bal, y_train_bal)
print(f"\nTraining complete. Iterations used: {model.named_steps['classifier'].n_iter_}")


# --- Baseline models ---
# Including baselines in the paper shows how much the MLP actually contributes.
# Majority-class baseline: always predicts the most common class (Response=0).
# Default MLP (threshold=0.50): standard predictions without any threshold tuning.

majority_clf = DummyClassifier(strategy="most_frequent", random_state=RANDOM_SEED)
# Fit on unbalanced training data so the classifier learns the real majority class.
majority_clf.fit(X_train, y_train)

test_probabilities = model.predict_proba(X_test)[:, 1]
majority_preds     = majority_clf.predict(X_test)
default_preds      = model.predict(X_test)  # uses default 0.50 threshold


def compute_metrics(y_true, y_pred, y_prob=None):
    return {
        "Accuracy":  round(accuracy_score(y_true, y_pred), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
        "F1-score":  round(f1_score(y_true, y_pred, zero_division=0), 4),
        "ROC-AUC":   round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else "N/A",
    }


majority_metrics = compute_metrics(y_test, majority_preds)
default_metrics  = compute_metrics(y_test, default_preds, test_probabilities)

print("\nMajority baseline metrics:", majority_metrics)
print("MLP (threshold=0.50):", default_metrics)


# --- Threshold tuning on validation set ---
# For imbalanced data, the default 0.50 threshold often predicts too few
# positives. We sweep thresholds from 0.10 to 0.90 and pick the one with
# the highest F1-score on the validation set — not on the test set, to avoid
# optimistic bias.

val_probs = model.predict_proba(X_val)[:, 1]
threshold_rows = []

# np.linspace avoids floating-point drift that can cause np.arange to miss 0.90.
for t in np.linspace(0.10, 0.90, 81):
    preds   = (val_probs >= t).astype(int)
    cm_val  = confusion_matrix(y_val, preds, labels=[0, 1])
    tn, fp, fn, tp = cm_val.ravel()

    threshold_rows.append({
        "Threshold": round(t, 2),
        "Accuracy":  round(accuracy_score(y_val, preds), 4),
        "Precision": round(precision_score(y_val, preds, zero_division=0), 4),
        "Recall":    round(recall_score(y_val, preds, zero_division=0), 4),
        "F1-score":  round(f1_score(y_val, preds, zero_division=0), 4),
        "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn),
    })

threshold_df  = pd.DataFrame(threshold_rows)
best_row      = threshold_df.loc[threshold_df["F1-score"].idxmax()]
best_threshold = float(best_row["Threshold"])

threshold_df.to_csv(OUTPUT_DIR / "threshold_tuning_results.csv", index=False)

print(f"\nBest threshold: {best_threshold}  (validation F1 = {best_row['F1-score']})")


# --- Final test evaluation ---
# Test set is touched only once, here, with the threshold selected on validation.
# This is the number reported in the paper.

test_preds_tuned = (test_probabilities >= best_threshold).astype(int)
tuned_metrics    = compute_metrics(y_test, test_preds_tuned, test_probabilities)
cm               = confusion_matrix(y_test, test_preds_tuned, labels=[0, 1])

pr_precision, pr_recall, _ = precision_recall_curve(y_test, test_probabilities)
pr_auc = round(auc(pr_recall, pr_precision), 4)

print("\n================ FINAL TEST RESULTS ================")
print(f"Threshold:  {best_threshold}")
for k, v in tuned_metrics.items():
    print(f"{k}: {v}")
print(f"PR-AUC:    {pr_auc}")
print("\nConfusion Matrix:\n", cm)
print("\nClassification Report:")
print(classification_report(y_test, test_preds_tuned, zero_division=0))


# --- Baseline comparison table ---
# All three models compared side by side — useful for the results section.

comparison = pd.DataFrame([
    {"Model": "Majority Baseline",                 **majority_metrics, "PR-AUC": "N/A"},
    {"Model": "MLP (threshold=0.50)",              **default_metrics,  "PR-AUC": "N/A"},
    {"Model": f"MLP (threshold={best_threshold})", **tuned_metrics,    "PR-AUC": pr_auc},
])
comparison.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)

print("\nModel Comparison:")
print(comparison.to_string(index=False))


# --- 5-fold cross-validation ---
# CV gives a less noisy estimate of generalization performance by averaging
# over multiple train/val splits. We run it on the original (unbalanced)
# training split to stay realistic.

cv_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier",   MLPClassifier(**mlp_params)),
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
cv_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="f1")

print(f"\n5-fold CV F1-scores: {np.round(cv_scores, 4)}")
print(f"Mean CV F1: {round(cv_scores.mean(), 4)}")


# --- Save metrics and predictions ---

metrics_rows = [
    ("Model",                          f"MLP hidden layers (32, 16)"),
    ("Threshold",                      best_threshold),
    ("Accuracy",                       tuned_metrics["Accuracy"]),
    ("Precision",                      tuned_metrics["Precision"]),
    ("Recall",                         tuned_metrics["Recall"]),
    ("F1-score",                       tuned_metrics["F1-score"]),
    ("ROC-AUC",                        tuned_metrics["ROC-AUC"]),
    ("PR-AUC",                         pr_auc),
    ("Mean 5-Fold CV F1",              round(cv_scores.mean(), 4)),
    ("Training Iterations",            model.named_steps["classifier"].n_iter_),
    ("Train Non-Responders (original)",int(y_train.value_counts().get(0, 0))),
    ("Train Responders (original)",    int(y_train.value_counts().get(1, 0))),
    ("Train Non-Responders (balanced)",int(y_train_bal.value_counts().get(0, 0))),
    ("Train Responders (balanced)",    int(y_train_bal.value_counts().get(1, 0))),
]

pd.DataFrame(metrics_rows, columns=["Metric", "Value"]).to_csv(
    OUTPUT_DIR / "model_metrics.csv", index=False
)

pred_output = X_test.copy()
pred_output["Actual_Response"]       = y_test.values
pred_output["Predicted_Probability"] = test_probabilities
pred_output["Predicted_Response"]    = test_preds_tuned
pred_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

joblib.dump(model, OUTPUT_DIR / "advertisement_response_mlp_model.pkl")


# --- Confusion matrix plot ---

fig, ax = plt.subplots(figsize=(5, 4))
ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_xticks([0, 1]); ax.set_xticklabels(["No Response", "Response"])
ax.set_yticks([0, 1]); ax.set_yticklabels(["No Response", "Response"])

halfway = cm.max() / 2 if cm.max() > 0 else 1
for i in range(2):
    for j in range(2):
        ax.text(j, i, cm[i, j], ha="center", va="center",
                color="white" if cm[i, j] > halfway else "black")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
plt.close()


# --- ROC curve ---

fpr, tpr, _ = roc_curve(y_test, test_probabilities)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"MLP  ROC-AUC = {tuned_metrics['ROC-AUC']}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random classifier")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300)
plt.close()


# --- Precision-Recall curve ---
# PR curve is more informative than ROC for imbalanced datasets because it
# focuses on the minority class (Response=1) without being swayed by the
# large number of true negatives.

plt.figure(figsize=(6, 5))
plt.plot(pr_recall, pr_precision, label=f"MLP  PR-AUC = {pr_auc}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=300)
plt.close()


# --- Feature importance ---
# Permutation importance works with any sklearn pipeline: it shuffles one
# feature at a time on the test set and measures how much the F1-score drops.
# A larger drop means the feature matters more.

try:
    imp = permutation_importance(
        model, X_test, y_test,
        n_repeats=10, random_state=RANDOM_SEED, scoring="f1"
    )

    feat_imp = pd.DataFrame({
        "Feature":         X_test.columns,
        "Importance_Mean": imp.importances_mean,
        "Importance_Std":  imp.importances_std,
    }).sort_values("Importance_Mean", ascending=False)

    feat_imp.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top10 = feat_imp.head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(top10["Feature"][::-1], top10["Importance_Mean"][::-1])
    plt.title("Top 10 Feature Importances (Permutation)")
    plt.xlabel("Mean Decrease in F1-score when feature is shuffled")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
    plt.close()

    print("\nTop 10 features by permutation importance:")
    print(top10[["Feature", "Importance_Mean"]].to_string(index=False))

except Exception as e:
    print("\nFeature importance skipped:", e)


# --- Save summary ---

summary = f"""
Neural Network Advertisement Response Prediction
Run timestamp: {RUN_TIMESTAMP}
Random seed:   {RANDOM_SEED}

Dataset: {dataset_path.name}
Original shape: {df.shape}
After cleaning and feature engineering: {data.shape}

Target distribution (full dataset):
{data[TARGET].value_counts().to_string()}

Split sizes:
  Train:      {X_train.shape[0]}
  Validation: {X_val.shape[0]}
  Test:       {X_test.shape[0]}

Training class distribution after oversampling:
{y_train_bal.value_counts().to_string()}

Model: MLPClassifier
  Hidden layers:    (32, 16)
  Activation:       relu
  Optimizer:        adam
  Alpha:            {mlp_params['alpha']}
  Learning rate:    {mlp_params['learning_rate_init']}
  Early stopping:   enabled (patience=40)
  Max iterations:   {mlp_params['max_iter']}
  Iterations used:  {model.named_steps['classifier'].n_iter_}

Baseline Comparison:
{comparison.to_string(index=False)}

Final Model Results (threshold = {best_threshold}):
  Accuracy:       {tuned_metrics['Accuracy']}
  Precision:      {tuned_metrics['Precision']}
  Recall:         {tuned_metrics['Recall']}
  F1-score:       {tuned_metrics['F1-score']}
  ROC-AUC:        {tuned_metrics['ROC-AUC']}
  PR-AUC:         {pr_auc}
  Mean 5-Fold CV F1: {round(cv_scores.mean(), 4)}

Confusion Matrix:
{cm}

Output files:
{OUTPUT_DIR}
"""

with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as f:
    f.write(summary)


print("\nAll outputs saved to:", OUTPUT_DIR)
print("\nFiles generated:")
for fname in sorted(OUTPUT_DIR.iterdir()):
    print(" -", fname.name)
