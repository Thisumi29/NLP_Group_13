from pathlib import Path

import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from feature_engineering import (
    load_and_split_data,
    create_tfidf_features,
)


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
SAVED_MODELS_DIR = PROJECT_DIR / "saved_models"


def train_random_forest():
    """Train, evaluate and save the Random Forest model."""

    # Load train and test data
    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
    ) = load_and_split_data()

    # Create TF-IDF features
    (
        X_train_tfidf,
        X_test_tfidf,
        tfidf_vectorizer,
    ) = create_tfidf_features(
        X_train_text,
        X_test_text,
    )

    print("\nTraining Random Forest model...")

    # Create and train model
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
    )

    rf_model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test_tfidf)

    # Evaluate model
    accuracy = accuracy_score(y_test, y_pred)

    print(
        f"\nRandom Forest Accuracy: "
        f"{accuracy * 100:.2f}%"
    )

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Save model and vectorizer
    SAVED_MODELS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_path = (
        SAVED_MODELS_DIR / "random_forest_model.pkl"
    )

    vectorizer_path = (
        SAVED_MODELS_DIR / "tfidf_vectorizer.pkl"
    )

    joblib.dump(rf_model, model_path)
    joblib.dump(tfidf_vectorizer, vectorizer_path)

    print("\nRandom Forest model saved to:", model_path)
    print("TF-IDF vectorizer saved to:", vectorizer_path)
    print("\nRandom Forest training completed successfully.")

    return rf_model, tfidf_vectorizer, accuracy


if __name__ == "__main__":
    train_random_forest()