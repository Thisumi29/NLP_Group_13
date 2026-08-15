# =============================================================
# STEP 4 - DL MODEL TRAINING: LSTM
# CCS3356 NLP | Group 13 - Fake News Detection
# Member 1: Thisumi Tanisha (CIT-24-01-0473)
# Run 01_data_cleaning.py FIRST - this reads processed_data.csv
# =============================================================

# %% 1. Imports
import os
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

# %% 2. Load cleaned data
data_path = r"C:\Users\thisu\Documents\NLP_Group_13\dataa\processed_data.csv"
df = pd.read_csv(data_path)
df["final_text"] = df["final_text"].fillna("").astype(str)

# Where to save plots and model artifacts
reports_dir = r"C:\Users\thisu\Documents\NLP_Group_13\Reports\DL"
models_dir = r"C:\Users\thisu\Documents\NLP_Group_13\Models"
os.makedirs(reports_dir, exist_ok=True)
os.makedirs(models_dir, exist_ok=True)

X = df["final_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# %% 3. Tokenize & pad sequences (LSTM needs integer sequences, not TF-IDF vectors)
VOCAB_SIZE = 10000
MAX_LEN = 200

tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding="post", truncating="post")
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding="post", truncating="post")

# %% 4. Build the LSTM model
model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=128, input_length=MAX_LEN),
    LSTM(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation="relu"),
    Dropout(0.2),
    Dense(1, activation="sigmoid")
])

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
model.summary()

# %% 5. Train
early_stop = EarlyStopping(monitor="val_loss", patience=2, restore_best_weights=True)

history = model.fit(
    X_train_pad, y_train,
    validation_split=0.1,
    epochs=8,
    batch_size=64,
    callbacks=[early_stop]
)

# %% 6. Predict & evaluate
y_proba = model.predict(X_test_pad).ravel()
y_pred = (y_proba >= 0.5).astype(int)

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

# %% 7. Training curves
plt.figure(figsize=(10, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="train")
plt.plot(history.history["val_accuracy"], label="val")
plt.title("LSTM Accuracy")
plt.xlabel("Epoch")
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="train")
plt.plot(history.history["val_loss"], label="val")
plt.title("LSTM Loss")
plt.xlabel("Epoch")
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "lstm_training_curves.png"))
plt.show()

# %% 8. Save model + tokenizer
model.save(os.path.join(models_dir, "lstm_model.h5"))

import pickle
with open(os.path.join(models_dir, "lstm_tokenizer.pkl"), "wb") as f:
    pickle.dump(tokenizer, f)

print("Saved LSTM model and tokenizer to:", models_dir)

# %% 9. ML vs DL comparison note (for your report Section 4)
print("""
For your report's model comparison table, put the Logistic Regression metrics
(from ml_logistic_regression.py) next to these LSTM metrics side by side,
and justify which one you'd recommend for the final app based on accuracy,
F1-score, and training/inference cost.
""")

# %% 6b. Confusion Matrix (required per assignment brief 7.6)
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("LSTM - Confusion Matrix")
plt.tight_layout()
plt.savefig(os.path.join(reports_dir, "lstm_confusion_matrix.png"))
plt.show()