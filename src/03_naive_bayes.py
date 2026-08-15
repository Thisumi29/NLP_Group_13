import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

# =========================================================
# Load preprocessed data (output of 01_pipeline_preprocessing.py)
# =========================================================
PROCESSED_DIR = r"C:\Users\imant\Documents\NLP_Group_13\dataa\processed"
INPUT_PATH = os.path.join(PROCESSED_DIR, "final_preprocessed.csv")

if not os.path.exists(INPUT_PATH):
    raise FileNotFoundError(
        f"Could not find: {INPUT_PATH}\n"
        f"Run 01_pipeline_preprocessing.py first to create final_preprocessed.csv."
    )

df = pd.read_csv(INPUT_PATH)
print("Loaded preprocessed dataset:", df.shape)

# =========================================================
# STEP 6: Train/Test Split + TF-IDF Feature Extraction
# =========================================================
print("\n\n=== STEP 6 ===")

X = df["final_text"]
y = df["label"]

# 80/20 split, stratified so both sets keep the same class balance
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

# ---------------------------------------------------------
# TF-IDF Vectorizer
#   max_features: caps vocabulary size to the top N most
#                 informative words (keeps things fast + avoids noise)
#   ngram_range=(1,2): includes both single words AND word pairs
#                 (e.g. "fake news" as one feature, not just "fake","news")
# ---------------------------------------------------------
tfidf = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))

X_train_tfidf = tfidf.fit_transform(X_train)   # fit ONLY on training data
X_test_tfidf = tfidf.transform(X_test)         # transform test data using same vocab

print("\nTF-IDF training matrix shape:", X_train_tfidf.shape)
print("TF-IDF test matrix shape:", X_test_tfidf.shape)

# ---------------------------------------------------------
# Save the vectorizer and split data for reuse in model training
# ---------------------------------------------------------
MODELS_DIR = r"C:\Users\imant\Documents\NLP_Group_13\Models"

print("Checking path:", MODELS_DIR)
print("Exists:", os.path.exists(MODELS_DIR))
print("Is directory:", os.path.isdir(MODELS_DIR))
print("Is file:", os.path.isfile(MODELS_DIR))
print("Parent folder exists:", os.path.exists(os.path.dirname(MODELS_DIR)))
print("Parent folder contents:", os.listdir(os.path.dirname(MODELS_DIR)))

try:
    os.makedirs(MODELS_DIR, exist_ok=True)
except Exception as e:
    print("makedirs failed with:", type(e).__name__, "-", e)

with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), "wb") as f:
    pickle.dump(tfidf, f)

with open(os.path.join(MODELS_DIR, "tfidf_train_test_split.pkl"), "wb") as f:
    pickle.dump((X_train_tfidf, X_test_tfidf, y_train, y_test), f)

print(f"\nTF-IDF vectorizer saved to: {MODELS_DIR}\\tfidf_vectorizer.pkl")
print(f"Train/test TF-IDF data saved to: {MODELS_DIR}\\tfidf_train_test_split.pkl")
print("\nFiles now inside Models folder:", os.listdir(MODELS_DIR))

# =========================================================
# STEP 7: Train and Evaluate Naive Bayes
# =========================================================
print("\n\n=== STEP 7: Naive Bayes ===")

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

y_pred = nb_model.predict(X_test_tfidf)
y_pred_proba = nb_model.predict_proba(X_test_tfidf)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"Accuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")
print(f"ROC-AUC:   {roc_auc:.2%}")

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred))

report_dict = classification_report(y_test, y_pred, target_names=["Fake", "Real"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

for col in ["precision", "recall", "f1-score"]:
    report_df[col] = report_df[col] * 100

print(report_df.round(2))
