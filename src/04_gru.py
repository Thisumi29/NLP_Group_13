import pandas as pd
import os
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report
)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, GRU, Dense, Dropout

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

# ---------------------------------------------------------
# Recreate the SAME train/test split used in 03_naive_bayes.py
# (same random_state, test_size, and stratify => identical split
# on the same data, so results stay comparable across models)
# ---------------------------------------------------------
X = df["final_text"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Train size:", X_train.shape[0])
print("Test size:", X_test.shape[0])

MODELS_DIR = r"C:\Users\imant\Documents\NLP_Group_13\Models"
os.makedirs(MODELS_DIR, exist_ok=True)

# =========================================================
# STEP 8: Tokenization + Padding for GRU
# =========================================================
print("\n\n=== STEP 8: Sequence Preparation for GRU ===")

VOCAB_SIZE = 10000
MAX_LEN = 300   # based on your word count stats (75th percentile ~290 words)

keras_tokenizer = Tokenizer(num_words=VOCAB_SIZE, oov_token="<OOV>")
keras_tokenizer.fit_on_texts(X_train)   # fit ONLY on training text

X_train_seq = keras_tokenizer.texts_to_sequences(X_train)
X_test_seq = keras_tokenizer.texts_to_sequences(X_test)

X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN, padding='post', truncating='post')
X_test_pad = pad_sequences(X_test_seq, maxlen=MAX_LEN, padding='post', truncating='post')

print("Padded training shape:", X_train_pad.shape)
print("Padded test shape:", X_test_pad.shape)

# Save tokenizer for later reuse (e.g. in your web app)
with open(os.path.join(MODELS_DIR, "keras_tokenizer.pkl"), "wb") as f:
    pickle.dump(keras_tokenizer, f)

print(f"Keras tokenizer saved to: {MODELS_DIR}\\keras_tokenizer.pkl")


# =========================================================
# STEP 9: Train and Evaluate GRU
# =========================================================
print("\n\n=== STEP 9: GRU ===")

EMBEDDING_DIM = 100

gru_model = Sequential([
    Embedding(input_dim=VOCAB_SIZE, output_dim=EMBEDDING_DIM, input_length=MAX_LEN),
    GRU(64, return_sequences=False),
    Dropout(0.3),
    Dense(32, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
gru_model.summary()

history = gru_model.fit(
    X_train_pad, y_train,
    validation_split=0.1,
    epochs=5,
    batch_size=64,
    verbose=1
)

# ---------------------------------------------------------
# Evaluate on test set
# ---------------------------------------------------------
y_pred_proba_gru = gru_model.predict(X_test_pad).flatten()
y_pred_gru = (y_pred_proba_gru >= 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred_gru)
precision = precision_score(y_test, y_pred_gru)
recall = recall_score(y_test, y_pred_gru)
f1 = f1_score(y_test, y_pred_gru)
roc_auc = roc_auc_score(y_test, y_pred_proba_gru)

print(f"\nAccuracy:  {accuracy:.2%}")
print(f"Precision: {precision:.2%}")
print(f"Recall:    {recall:.2%}")
print(f"F1 Score:  {f1:.2%}")
print(f"ROC-AUC:   {roc_auc:.2%}")

print("\n--- Confusion Matrix ---")
print(confusion_matrix(y_test, y_pred_gru))

print("\n--- Classification Report ---")
report_dict = classification_report(y_test, y_pred_gru, target_names=["Fake", "Real"], output_dict=True)
report_df = pd.DataFrame(report_dict).transpose()

for col in ["precision", "recall", "f1-score"]:
    report_df[col] = report_df[col] * 100

print(report_df.round(2))

# ---------------------------------------------------------
# Save the trained GRU model
# ---------------------------------------------------------
gru_model.save(os.path.join(MODELS_DIR, "gru_model.keras"))
print(f"\nGRU model saved to: {MODELS_DIR}\\gru_model.keras")