from pathlib import Path
import re

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "dataa" / "processed"

INPUT_PATH = PROCESSED_DIR / "merged_labeled.csv"


def download_nltk_resources() -> None:
    """Download the NLTK resources required for preprocessing."""
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)


def clean_text(text: str) -> str:
    """Perform basic text cleaning."""
    text = str(text).lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\S+", " ", text)

    # Remove HTML tags
    text = re.sub(r"<.*?>", " ", text)

    # Keep English letters and spaces only
    text = re.sub(r"[^a-z\s]", " ", text)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


def preprocess_text(
    text: str,
    stop_words: set[str],
    lemmatizer: WordNetLemmatizer,
) -> str:
    """Remove stopwords and lemmatize words."""
    tokens = text.split()

    tokens = [
        lemmatizer.lemmatize(word)
        for word in tokens
        if word not in stop_words and len(word) > 1
    ]

    return " ".join(tokens)


def preprocess_dataset() -> pd.DataFrame:
    """Clean and preprocess the merged fake-news dataset."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "merged_labeled.csv was not found. "
            "Run src/data_loading.py first."
        )

    download_nltk_resources()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    # Load merged dataset
    df = pd.read_csv(INPUT_PATH)

    print("Dataset shape before preprocessing:", df.shape)
    print("\nMissing values before cleaning:")
    print(df.isnull().sum())

    # Combine title and text into one content column
    df["content"] = (
        df["title"].fillna("").astype(str)
        + " "
        + df["text"].fillna("").astype(str)
    ).str.strip()

    combined_path = PROCESSED_DIR / "combined_title_text.csv"
    df.to_csv(combined_path, index=False)
    print("\nCombined title and text saved to:", combined_path)

    # Remove empty content and missing labels
    df = df[df["content"] != ""]
    df = df[df["label"].notna()]

    # Remove duplicates
    duplicate_count = df.duplicated(subset=["content"]).sum()
    print("Duplicate rows removed:", duplicate_count)

    df = df.drop_duplicates(subset=["content"], keep="first")
    df = df.reset_index(drop=True)

    cleaned_path = PROCESSED_DIR / "cleaned_data.csv"
    df.to_csv(cleaned_path, index=False)
    print("Cleaned dataset saved to:", cleaned_path)

    # Text cleaning
    df["cleaned_content"] = df["content"].apply(clean_text)

    # Stopword removal and lemmatization
    df["processed_content"] = df["cleaned_content"].apply(
        lambda text: preprocess_text(
            text,
            stop_words,
            lemmatizer,
        )
    )

    # Remove rows that became empty after preprocessing
    df = df[df["processed_content"] != ""]
    df = df.reset_index(drop=True)

    final_path = PROCESSED_DIR / "final_preprocessed.csv"
    df.to_csv(final_path, index=False)

    print("\nFinal preprocessed dataset saved to:", final_path)
    print("Final dataset shape:", df.shape)
    print("\nMissing values after preprocessing:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    preprocess_dataset()