# =============================================================
# STEP 3 - ML MODEL TRAINING: TF-IDF + LOGISTIC REGRESSION
# CCS3356 NLP | Group 13 - Fake News Detection
# Member 1: Thisumi Tanisha (CIT-24-01-0473)
# Run 01_data_cleaning.py FIRST - this reads processed_data.csv
# =============================================================

# %% 1. Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)

# %% 2. Load cleaned data
data_path = r"C:\Users\thisu\Documents\NLP_Group_13\dataa\processed_data.csv"
df = pd.read_csv(data_path)

# IMPORTANT: fillna BEFORE astype(str) - this is the fix for the NaN error
df["final_text"] = df["final_text"].fillna("").astype(str)

print("Any NaN left in final_text?", df["final_text"].isna().sum())

X = df["final_text"]
y = df["label"]

# Where to save plots and model artifacts
reports_dir = r"C:\Users\thisu\Documents\NLP_Group_13\Reports\ML"
models_dir = r"C:\Users\thisu\Documents\NLP_Group_13\Models"
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

# %% 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Train size:", X_train.shape[0], "| Test size:", X_test.shape[0])

# %% 4. TF-IDF vectorization (feature engineering)
tfidf = TfidfVectorizer(max_features=10000, ngram_range=(1, 2))
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)
print("TF-IDF train matrix shape:", X_train_tfidf.shape)

# %% 5. Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42)
log_reg.fit(X_train_tfidf, y_train)

# %% 6. Predict
y_pred = log_reg.predict(X_test_tfidf)
y_proba = log_reg.predict_proba(X_test_tfidf)[:, 1]

# %% 7. Evaluation metrics
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)

print(f"Accuracy : {acc:.4f}")
print(f"Precision: {prec:.4f}")
print(f"Recall   : {rec:.4f}")
print(f"F1 Score : {f1:.4f}")
print(f"ROC-AUC  : {auc:.4f}")
print("\nClassification Report:\n",
      classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# %% 8. Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Logistic Regression - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "ml_confusion_matrix.png"))
plt.show()

# %% 9. ROC curve plot
fpr, tpr, _ = roc_curve(y_test, y_proba)
plt.figure(figsize=(5, 4))
plt.plot(fpr, tpr, label=f"AUC = {auc:.3f}")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Logistic Regression")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "ml_roc_curve.png"))
plt.show()

# %% 10. Save model + vectorizer for the final app / integration step
joblib.dump(log_reg, os.path.join(models_dir, "logistic_regression_model.pkl"))
joblib.dump(tfidf, os.path.join(models_dir, "tfidf_vectorizer.pkl"))
print("Saved model and vectorizer to:", models_dir)