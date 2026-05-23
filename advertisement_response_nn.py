from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
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


BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "outputs"
OUTPUT_DIR.mkdir(exist_ok=True)


# Load dataset
def find_dataset(base_dir):
    for filename in [
        "Predicting the Result of a Shop's Advertisement.csv",
        "Predicting the Result of a Shop's Advertisement.xlsx",
        "Predicting the Result of a Shop's Advertisement.xls",
    ]:
        path = base_dir / filename
        if path.exists():
            return path

    possible_files = (
        list(base_dir.glob("*.csv"))
        + list(base_dir.glob("*.xlsx"))
        + list(base_dir.glob("*.xls"))
    )

    if possible_files:
        return possible_files[0]

    raise FileNotFoundError(
        "No CSV/XLSX dataset found. Put the dataset in the same folder as this script."
    )


def load_dataset(path):
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(path)
        return df

    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)

    raise ValueError("Unsupported dataset file type.")


dataset_path = find_dataset(BASE_DIR)
df = load_dataset(dataset_path)

if "Response" not in df.columns:
    raise ValueError("Target column 'Response' was not found.")

print("Dataset loaded:", dataset_path.name)
print("Dataset shape:", df.shape)
print("\nTarget distribution:")
print(df["Response"].value_counts())
print("\nTarget distribution percentage:")
print(df["Response"].value_counts(normalize=True).round(4))


# Feature engineering
data = df.copy()

if "Dt_Customer" in data.columns:
    data["Dt_Customer"] = pd.to_datetime(
        data["Dt_Customer"],
        dayfirst=True,
        errors="coerce",
    )
    data["Customer_Tenure_Days"] = (
        data["Dt_Customer"].max() - data["Dt_Customer"]
    ).dt.days

if "Year_Birth" in data.columns and "Dt_Customer" in data.columns:
    data["Age"] = data["Dt_Customer"].dt.year - data["Year_Birth"]
elif "Year_Birth" in data.columns:
    data["Age"] = 2026 - data["Year_Birth"]

if "Age" in data.columns:
    data = data[(data["Age"] >= 18) & (data["Age"] <= 100)]

spending_columns = [
    "MntWines",
    "MntFruits",
    "MntMeatProducts",
    "MntFishProducts",
    "MntSweetProducts",
    "MntGoldProds",
]

purchase_columns = [
    "NumDealsPurchases",
    "NumWebPurchases",
    "NumCatalogPurchases",
    "NumStorePurchases",
]

campaign_columns = [
    "AcceptedCmp1",
    "AcceptedCmp2",
    "AcceptedCmp3",
    "AcceptedCmp4",
    "AcceptedCmp5",
]

if all(col in data.columns for col in spending_columns):
    data["Total_Spending"] = data[spending_columns].sum(axis=1)

if all(col in data.columns for col in purchase_columns):
    data["Total_Purchases"] = data[purchase_columns].sum(axis=1)

if all(col in data.columns for col in campaign_columns):
    data["Total_Accepted_Campaigns"] = data[campaign_columns].sum(axis=1)

if "Kidhome" in data.columns and "Teenhome" in data.columns:
    data["Total_Children"] = data["Kidhome"] + data["Teenhome"]

if "Income" in data.columns and "Total_Spending" in data.columns:
    data["Spending_to_Income_Ratio"] = data["Total_Spending"] / (data["Income"] + 1)

if "Recency" in data.columns and "Total_Purchases" in data.columns:
    data["Purchases_per_Recency"] = data["Total_Purchases"] / (data["Recency"] + 1)


# Prepare features and target
target = "Response"

drop_columns = ["ID", "Dt_Customer", "Year_Birth", "Z_CostContact", "Z_Revenue", target]
drop_columns = [col for col in drop_columns if col in data.columns]

X = data.drop(columns=drop_columns)
y = data[target].astype(int)

categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features = X.select_dtypes(exclude=["object"]).columns.tolist()

print("\nNumber of input features before encoding:", X.shape[1])
print("Categorical features:", categorical_features)
print("Numerical features:", numerical_features)


# Train/validation/test split
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print("\nSplit sizes:")
print("Training:", X_train.shape[0])
print("Validation:", X_val.shape[0])
print("Testing:", X_test.shape[0])


# Oversample minority class in training set
def oversample_minority_class(X_input, y_input):
    train_data = X_input.copy()
    train_data[target] = y_input.values

    majority = train_data[train_data[target] == 0]
    minority = train_data[train_data[target] == 1]

    minority_oversampled = minority.sample(n=len(majority), replace=True, random_state=42)

    balanced_data = pd.concat([majority, minority_oversampled], axis=0)
    balanced_data = balanced_data.sample(frac=1, random_state=42)

    X_balanced = balanced_data.drop(columns=[target])
    y_balanced = balanced_data[target].astype(int)

    return X_balanced, y_balanced


X_train_balanced, y_train_balanced = oversample_minority_class(X_train, y_train)

print("\nOriginal training distribution:")
print(y_train.value_counts())
print("\nBalanced training distribution:")
print(y_train_balanced.value_counts())


# Build preprocessing pipeline
numerical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numerical_pipeline, numerical_features),
        ("cat", categorical_pipeline, categorical_features),
    ]
)


# Define and train the MLP model
mlp = MLPClassifier(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    solver="adam",
    alpha=0.001,
    learning_rate_init=0.001,
    max_iter=1000,
    early_stopping=True,
    validation_fraction=0.20,
    n_iter_no_change=40,
    random_state=42,
)

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", mlp),
    ]
)

model.fit(X_train_balanced, y_train_balanced)

print("\nTraining completed.")
print("Training iterations used:", model.named_steps["classifier"].n_iter_)


# Tune classification threshold on validation set
val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.10, 0.91, 0.01)
threshold_results = []

for threshold in thresholds:
    val_predictions = (val_probabilities >= threshold).astype(int)
    threshold_results.append(
        {
            "Threshold": round(threshold, 2),
            "Accuracy": accuracy_score(y_val, val_predictions),
            "Precision": precision_score(y_val, val_predictions, zero_division=0),
            "Recall": recall_score(y_val, val_predictions, zero_division=0),
            "F1-score": f1_score(y_val, val_predictions, zero_division=0),
        }
    )

threshold_df = pd.DataFrame(threshold_results)
best_row = threshold_df.loc[threshold_df["F1-score"].idxmax()]
best_threshold = float(best_row["Threshold"])

threshold_df.to_csv(OUTPUT_DIR / "threshold_tuning_results.csv", index=False)

print("\nBest threshold based on validation F1-score:")
print(best_row)


# Evaluate on test set
test_probabilities = model.predict_proba(X_test)[:, 1]
test_predictions = (test_probabilities >= best_threshold).astype(int)

accuracy = accuracy_score(y_test, test_predictions)
precision = precision_score(y_test, test_predictions, zero_division=0)
recall = recall_score(y_test, test_predictions, zero_division=0)
f1 = f1_score(y_test, test_predictions, zero_division=0)
roc_auc = roc_auc_score(y_test, test_probabilities)
cm = confusion_matrix(y_test, test_predictions)

print("\n================ FINAL TEST RESULTS ================")
print("Model: MLP with hidden layers (32, 16)")
print("Threshold:", round(best_threshold, 2))
print("Accuracy:", round(accuracy, 4))
print("Precision:", round(precision, 4))
print("Recall:", round(recall, 4))
print("F1-score:", round(f1, 4))
print("ROC-AUC:", round(roc_auc, 4))

print("\nConfusion Matrix:")
print(cm)

print("\nClassification Report:")
print(classification_report(y_test, test_predictions, zero_division=0))


# Cross-validation estimate
cv_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "classifier",
            MLPClassifier(
                hidden_layer_sizes=(32, 16),
                activation="relu",
                solver="adam",
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=1000,
                early_stopping=True,
                validation_fraction=0.20,
                n_iter_no_change=40,
                random_state=42,
            ),
        ),
    ]
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_f1_scores = cross_val_score(cv_model, X_train, y_train, cv=cv, scoring="f1")

print("\n5-fold cross-validation F1-scores:")
print(np.round(cv_f1_scores, 4))
print("Mean CV F1-score:", round(cv_f1_scores.mean(), 4))


# Save metrics and predictions
metrics = pd.DataFrame(
    {
        "Metric": [
            "Model",
            "Threshold",
            "Accuracy",
            "Precision",
            "Recall",
            "F1-score",
            "ROC-AUC",
            "Mean 5-Fold CV F1-score",
            "Training Iterations",
            "Original Training Non-Responders",
            "Original Training Responders",
            "Balanced Training Non-Responders",
            "Balanced Training Responders",
        ],
        "Value": [
            "MLP hidden layers (32, 16)",
            round(best_threshold, 2),
            round(accuracy, 4),
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            round(roc_auc, 4),
            round(cv_f1_scores.mean(), 4),
            model.named_steps["classifier"].n_iter_,
            int(y_train.value_counts().get(0, 0)),
            int(y_train.value_counts().get(1, 0)),
            int(y_train_balanced.value_counts().get(0, 0)),
            int(y_train_balanced.value_counts().get(1, 0)),
        ],
    }
)

metrics.to_csv(OUTPUT_DIR / "model_metrics.csv", index=False)

predictions_output = X_test.copy()
predictions_output["Actual_Response"] = y_test.values
predictions_output["Predicted_Probability"] = test_probabilities
predictions_output["Predicted_Response"] = test_predictions
predictions_output.to_csv(OUTPUT_DIR / "test_predictions.csv", index=False)

joblib.dump(model, OUTPUT_DIR / "advertisement_response_mlp_model.pkl")


# Confusion matrix plot
plt.figure(figsize=(5, 4))
plt.imshow(cm)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.xticks([0, 1], ["No Response", "Response"])
plt.yticks([0, 1], ["No Response", "Response"])

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, cm[i, j], ha="center", va="center")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / "confusion_matrix.png", dpi=300)
plt.close()


# ROC curve
fpr, tpr, _ = roc_curve(y_test, test_probabilities)

plt.figure(figsize=(6, 5))
plt.plot(fpr, tpr, label=f"ROC-AUC = {roc_auc:.4f}")
plt.plot([0, 1], [0, 1], linestyle="--", label="Random Classifier")
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "roc_curve.png", dpi=300)
plt.close()


# Precision-Recall curve
pr_precision, pr_recall, _ = precision_recall_curve(y_test, test_probabilities)
pr_auc = auc(pr_recall, pr_precision)

plt.figure(figsize=(6, 5))
plt.plot(pr_recall, pr_precision, label=f"PR-AUC = {pr_auc:.4f}")
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "precision_recall_curve.png", dpi=300)
plt.close()


# Feature importance via permutation
try:
    importance = permutation_importance(
        model,
        X_test,
        y_test,
        n_repeats=10,
        random_state=42,
        scoring="f1",
    )

    feature_importance = pd.DataFrame(
        {
            "Feature": X_test.columns,
            "Importance_Mean": importance.importances_mean,
            "Importance_Std": importance.importances_std,
        }
    ).sort_values(by="Importance_Mean", ascending=False)

    feature_importance.to_csv(OUTPUT_DIR / "feature_importance.csv", index=False)

    top_features = feature_importance.head(10)

    plt.figure(figsize=(8, 5))
    plt.barh(
        top_features["Feature"][::-1],
        top_features["Importance_Mean"][::-1],
    )
    plt.title("Top 10 Feature Importances")
    plt.xlabel("Permutation Importance")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "feature_importance.png", dpi=300)
    plt.close()

except Exception as error:
    print("\nFeature importance skipped:")
    print(error)


# Save summary
summary_text = f"""
Neural Network Advertisement Response Prediction Summary

Dataset file: {dataset_path.name}
Original dataset shape: {df.shape}
Dataset shape after cleaning: {data.shape}

Target Distribution:
{data[target].value_counts().to_string()}

Original Training Distribution:
{y_train.value_counts().to_string()}

Balanced Training Distribution:
{y_train_balanced.value_counts().to_string()}

Final Model:
- Type: Multilayer Feed-Forward Neural Network / MLP
- Hidden layers: (32, 16)
- Activation: ReLU
- Optimizer: Adam
- Alpha: 0.001
- Learning rate: 0.001
- Early stopping: enabled
- Training imbalance handling: minority-class oversampling on training set only
- Threshold selected using validation F1-score

Final Test Results:
- Threshold: {best_threshold:.2f}
- Accuracy: {accuracy:.4f}
- Precision: {precision:.4f}
- Recall: {recall:.4f}
- F1-score: {f1:.4f}
- ROC-AUC: {roc_auc:.4f}
- PR-AUC: {pr_auc:.4f}
- Mean 5-fold CV F1-score: {cv_f1_scores.mean():.4f}
- Training iterations: {model.named_steps["classifier"].n_iter_}

Confusion Matrix:
{cm}

Generated Output Files:
- model_metrics.csv
- threshold_tuning_results.csv
- test_predictions.csv
- advertisement_response_mlp_model.pkl
- confusion_matrix.png
- roc_curve.png
- precision_recall_curve.png
- feature_importance.csv
- feature_importance.png
- summary.txt
"""

with open(OUTPUT_DIR / "summary.txt", "w", encoding="utf-8") as file:
    file.write(summary_text)

print("\nAll outputs saved in:", OUTPUT_DIR)
print("\nGenerated files:")
for file in sorted(OUTPUT_DIR.iterdir()):
    print("-", file.name)
