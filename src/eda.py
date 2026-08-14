from pathlib import Path
from collections import Counter

import pandas as pd
import matplotlib.pyplot as plt


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
PROCESSED_DIR = PROJECT_DIR / "dataa" / "processed"

INPUT_PATH = PROCESSED_DIR / "final_preprocessed.csv"


def run_eda() -> None:
    """Generate and save EDA visualizations."""

    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            "final_preprocessed.csv was not found. "
            "Run src/data_preprocessing.py first."
        )

    # Load final preprocessed dataset
    df = pd.read_csv(INPUT_PATH)

    print("Dataset loaded successfully.")
    print("Dataset shape:", df.shape)

    # -------------------------------------------------
    # 1. Class distribution
    # -------------------------------------------------
    print("\nClass distribution:")
    print(df["label"].value_counts())

    plt.figure(figsize=(6, 4))
    df["label"].value_counts().sort_index().plot(kind="bar")

    plt.title("Class Distribution (0 = Fake, 1 = Real)")
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()

    class_path = PROJECT_DIR / "class_distribution.png"
    plt.savefig(class_path)
    plt.close()

    print("Class distribution saved to:", class_path)

    # -------------------------------------------------
    # 2. Subject distribution
    # -------------------------------------------------
    if "subject" in df.columns:
        print("\nSubject distribution:")
        print(df["subject"].value_counts())

        plt.figure(figsize=(10, 6))
        df["subject"].value_counts().plot(kind="bar")

        plt.title("News Subject Distribution")
        plt.xlabel("Subject")
        plt.ylabel("Count")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()

        subject_path = PROJECT_DIR / "subject_distribution.png"
        plt.savefig(subject_path)
        plt.close()

        print("Subject distribution saved to:", subject_path)

    # -------------------------------------------------
    # 3. Top words
    # -------------------------------------------------
    all_words = " ".join(
        df["processed_content"].fillna("").astype(str)
    ).split()

    word_counts = Counter(all_words)
    top_words = word_counts.most_common(20)

    top_words_df = pd.DataFrame(
        top_words,
        columns=["word", "count"],
    )

    print("\nTop 20 words:")
    print(top_words_df)

    plt.figure(figsize=(10, 6))
    plt.barh(
        top_words_df["word"][::-1],
        top_words_df["count"][::-1],
    )

    plt.title("Top 20 Most Frequent Words")
    plt.xlabel("Frequency")
    plt.ylabel("Word")
    plt.tight_layout()

    top_words_path = PROJECT_DIR / "top_words.png"
    plt.savefig(top_words_path)
    plt.close()

    print("Top words chart saved to:", top_words_path)

    # -------------------------------------------------
    # 4. Word count distribution
    # -------------------------------------------------
    df["word_count"] = (
        df["processed_content"]
        .fillna("")
        .astype(str)
        .str.split()
        .str.len()
    )

    print("\nWord count statistics:")
    print(df["word_count"].describe())

    plt.figure(figsize=(8, 5))
    plt.hist(df["word_count"], bins=50)

    plt.title("Word Count Distribution")
    plt.xlabel("Number of Words")
    plt.ylabel("Frequency")
    plt.tight_layout()

    word_distribution_path = (
        PROJECT_DIR / "word_count_distribution.png"
    )

    plt.savefig(word_distribution_path)
    plt.close()

    print(
        "Word count distribution saved to:",
        word_distribution_path,
    )

    # -------------------------------------------------
    # 5. Average word count by class
    # -------------------------------------------------
    average_word_count = (
        df.groupby("label")["word_count"]
        .mean()
        .sort_index()
    )

    print("\nAverage word count by class:")
    print(average_word_count)

    plt.figure(figsize=(6, 4))
    average_word_count.plot(kind="bar")

    plt.title("Average Word Count by Class")
    plt.xlabel("Label (0 = Fake, 1 = Real)")
    plt.ylabel("Average Word Count")
    plt.xticks(rotation=0)
    plt.tight_layout()

    word_class_path = PROJECT_DIR / "word_count_by_class.png"
    plt.savefig(word_class_path)
    plt.close()

    print("Word count by class saved to:", word_class_path)
    print("\nEDA completed successfully.")


if __name__ == "__main__":
    run_eda()