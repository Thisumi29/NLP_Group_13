from pathlib import Path

import pandas as pd


# Project folders
PROJECT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_DIR / "dataa" / "raw"
PROCESSED_DIR = PROJECT_DIR / "dataa" / "processed"


def load_and_merge_data() -> pd.DataFrame:
    """Load Fake.csv and True.csv, add labels, and merge them."""

    fake_path = RAW_DIR / "Fake.csv"
    true_path = RAW_DIR / "True.csv"

    print("Fake.csv exists:", fake_path.exists())
    print("True.csv exists:", true_path.exists())

    if not fake_path.exists() or not true_path.exists():
        raise FileNotFoundError(
            "Fake.csv or True.csv was not found inside dataa/raw."
        )

    # Load raw datasets
    fake_df = pd.read_csv(fake_path)
    true_df = pd.read_csv(true_path)

    print("Fake dataset shape:", fake_df.shape)
    print("True dataset shape:", true_df.shape)

    # Add labels: 0 = Fake, 1 = Real
    fake_df["label"] = 0
    true_df["label"] = 1

    # Merge and shuffle both datasets
    df = pd.concat([fake_df, true_df], axis=0)
    df = df.sample(frac=1, random_state=42)
    df = df.reset_index(drop=True)

    print("Merged dataset shape:", df.shape)
    print("Label counts:")
    print(df["label"].value_counts())

    # Save merged and labelled dataset
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    output_path = PROCESSED_DIR / "merged_labeled.csv"
    df.to_csv(output_path, index=False)

    print("Merged dataset saved to:", output_path)

    return df


if __name__ == "__main__":
    load_and_merge_data()