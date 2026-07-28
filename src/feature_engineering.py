from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "dataa" / "processed"

INPUT_PATH = PROCESSED_DIR / "final_preprocessed.csv"


def load_and_split_data(
    test_size: float = 0.2,
    random_state: int = 42,
):
    """Load the processed dataset and create train/test splits."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "final_preprocessed.csv was not found. "
            "Run src/data_preprocessing.py first."
        )

    df = pd.read_csv(INPUT_PATH)

    # Remove any unusable rows
    df = df.dropna(subset=["processed_content", "label"])
    df = df[df["processed_content"].astype(str).str.strip() != ""]
    df = df.reset_index(drop=True)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    X_train_text, X_test_text, y_train, y_test = train_test_split(
        df["processed_content"].astype(str),
        df["label"].astype(int),
        test_size=test_size,
        random_state=random_state,
        stratify=df["label"],
    )

    print("\nTrain size:", X_train_text.shape[0])
    print("Test size:", X_test_text.shape[0])

    return X_train_text, X_test_text, y_train, y_test


def create_tfidf_features(
    X_train_text,
    X_test_text,
    max_features: int = 5000,
):
    """Convert text data into TF-IDF feature matrices."""

    tfidf_vectorizer = TfidfVectorizer(
        max_features=max_features
    )

    X_train_tfidf = tfidf_vectorizer.fit_transform(
        X_train_text
    )

    X_test_tfidf = tfidf_vectorizer.transform(
        X_test_text
    )

    print(
        "\nTF-IDF training matrix shape:",
        X_train_tfidf.shape,
    )

    print(
        "TF-IDF testing matrix shape:",
        X_test_tfidf.shape,
    )

    print("\nSample feature names:")
    print(tfidf_vectorizer.get_feature_names_out()[:20])

    return (
        X_train_tfidf,
        X_test_tfidf,
        tfidf_vectorizer,
    )


if __name__ == "__main__":
    (
        X_train_text,
        X_test_text,
        y_train,
        y_test,
    ) = load_and_split_data()

    (
        X_train_tfidf,
        X_test_tfidf,
        tfidf_vectorizer,
    ) = create_tfidf_features(
        X_train_text,
        X_test_text,
    )

    print("\nFeature engineering completed successfully.")