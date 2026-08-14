from pathlib import Path
import pickle

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
)
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    Bidirectional,
    Dense,
    Dropout,
    Embedding,
    LSTM,
)
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer

from feature_engineering import load_and_split_data


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"

# Bi-LSTM settings
VOCAB_SIZE = 20000
MAX_LEN = 300
EMBEDDING_DIM = 64
BATCH_SIZE = 128
EPOCHS = 5

# Reproducible results
np.random.seed(42)
tf.random.set_seed(42)


def train_bilstm():
    """Train, evaluate and save the Bi-LSTM model."""

    # Load train and test text
    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
    ) = load_and_split_data()

    X_train_text = X_train_text.astype(str)
    X_test_text = X_test_text.astype(str)

    # Convert words into integer sequences
    tokenizer = Tokenizer(
        num_words=VOCAB_SIZE,
        oov_token="<OOV>",
    )

    tokenizer.fit_on_texts(X_train_text)

    X_train_sequences = tokenizer.texts_to_sequences(
        X_train_text
    )

    X_test_sequences = tokenizer.texts_to_sequences(
        X_test_text
    )

    # Make all sequences the same length
    X_train_padded = pad_sequences(
        X_train_sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    X_test_padded = pad_sequences(
        X_test_sequences,
        maxlen=MAX_LEN,
        padding="post",
        truncating="post",
    )

    print("\nTraining sequence shape:", X_train_padded.shape)
    print("Testing sequence shape:", X_test_padded.shape)

    # Build Bi-LSTM model
    bilstm_model = Sequential(
        [
            Embedding(
                input_dim=VOCAB_SIZE,
                output_dim=EMBEDDING_DIM,
            ),
            Bidirectional(
                LSTM(64)
            ),
            Dropout(0.5),
            Dense(
                32,
                activation="relu",
            ),
            Dropout(0.3),
            Dense(
                1,
                activation="sigmoid",
            ),
        ]
    )

    bilstm_model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    bilstm_model.build(
        input_shape=(None, MAX_LEN)
    )

    print("\nBi-LSTM Model Summary:")
    bilstm_model.summary()

    early_stopping = EarlyStopping(
        monitor="val_loss",
        patience=2,
        restore_best_weights=True,
    )

    print("\nTraining Bi-LSTM model...")

    bilstm_model.fit(
        X_train_padded,
        y_train,
        validation_split=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stopping],
        verbose=1,
    )

    # Evaluate model
    test_loss, test_accuracy = bilstm_model.evaluate(
        X_test_padded,
        y_test,
        verbose=0,
    )

    print(f"\nBi-LSTM Test Loss: {test_loss:.4f}")
    print(
        f"Bi-LSTM Accuracy: "
        f"{test_accuracy * 100:.2f}%"
    )

    # Predictions
    prediction_probabilities = bilstm_model.predict(
        X_test_padded,
        verbose=0,
    )

    y_pred = (
        prediction_probabilities >= 0.5
    ).astype(int).ravel()

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            y_pred,
        )
    )

    # Save model and tokenizer
    SAVED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        SAVED_MODELS_DIR / "bilstm_model.keras"
    )

    tokenizer_path = (
        SAVED_MODELS_DIR / "bilstm_tokenizer.pkl"
    )

    bilstm_model.save(model_path)

    with open(tokenizer_path, "wb") as file:
        pickle.dump(tokenizer, file)

    print("\nBi-LSTM model saved to:", model_path)
    print("Tokenizer saved to:", tokenizer_path)
    print("\nBi-LSTM training completed successfully.")

    return bilstm_model, tokenizer, test_accuracy


if __name__ == "__main__":
    train_bilstm()