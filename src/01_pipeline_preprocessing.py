import pandas as pd
import os
import re

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# =========================================================
# STEP 1: Load and Inspect
# =========================================================
RAW_DIR = r"C:\Users\imant\Documents\NLP_Group_13\dataa\raw"
FAKE_PATH = os.path.join(RAW_DIR, "Fake.csv")
TRUE_PATH = os.path.join(RAW_DIR, "True.csv")

if not os.path.isdir(RAW_DIR):
    raise FileNotFoundError(
        f"Could not find the folder: {RAW_DIR}\n"
        f"Double-check that 'dataa\\raw' actually exists at that exact "
        f"location, or edit RAW_DIR above to match your real path."
    )

print(f"Looking inside: {RAW_DIR}")
print(f"Files found there: {os.listdir(RAW_DIR)}\n")

for path in [FAKE_PATH, TRUE_PATH]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find: {path}")

fake_df = pd.read_csv(FAKE_PATH)
true_df = pd.read_csv(TRUE_PATH)

print("Fake news file loaded:", fake_df.shape)
print("Real news file loaded:", true_df.shape)

inspect_fake = fake_df.copy()
inspect_true = true_df.copy()
inspect_fake["label"] = 0   
inspect_true["label"] = 1   
inspect_df = pd.concat([inspect_fake, inspect_true], axis=0, ignore_index=True)
inspect_df = inspect_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nCombined dataset shape:", inspect_df.shape)
print("\n--- First 5 rows ---")
print(inspect_df.head())
print("\n--- Missing values per column ---")
print(inspect_df.isnull().sum())
print("\n--- Duplicate rows (full row duplicates) ---")
print(inspect_df.duplicated().sum())
if "text" in inspect_df.columns:
    print("\n--- Duplicate articles (based on 'text' column) ---")
    print(inspect_df.duplicated(subset=["text"]).sum())
print("\n--- Class distribution ---")
print(inspect_df["label"].value_counts())

text_col = "text" if "text" in inspect_df.columns else inspect_df.columns[0]
inspect_df["text_length"] = inspect_df[text_col].astype(str).apply(len)
print(f"\n--- Text length stats (column used: '{text_col}') ---")
print(inspect_df["text_length"].describe())
print("\n--- Number of empty/near-empty articles (<20 chars) ---")
print((inspect_df["text_length"] < 20).sum())


# =========================================================
# STEP 2: Merge and Label
# =========================================================
PROCESSED_DIR = r"C:\Users\imant\Documents\NLP_Group_13\dataa\processed"
STEP2_OUTPUT = os.path.join(PROCESSED_DIR, "merged_labeled.csv")

fake_df["label"] = 0   
true_df["label"] = 1   
fake_df["label_name"] = "Fake"
true_df["label_name"] = "Real"
fake_df["source_file"] = "Fake.csv"
true_df["source_file"] = "True.csv"

merged_df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
merged_df = merged_df.sample(frac=1, random_state=42).reset_index(drop=True)

print("\n\n=== STEP 2 ===")
print("Merged dataset shape:", merged_df.shape)
print("\n--- Label distribution ---")
print(merged_df["label_name"].value_counts())

os.makedirs(PROCESSED_DIR, exist_ok=True)
merged_df.to_csv(STEP2_OUTPUT, index=False)
print(f"\nMerged & labeled dataset saved to: {STEP2_OUTPUT}")


# =========================================================
# STEP 3: Combine Title and Text
# =========================================================
STEP3_OUTPUT = os.path.join(PROCESSED_DIR, "combined_title_text.csv")

df = merged_df.copy()
df["title"] = df["title"].fillna("")
df["text"] = df["text"].fillna("")

print("\n\n=== STEP 3 ===")
print("Missing titles:", (df["title"] == "").sum())
print("Missing texts:", (df["text"] == "").sum())

df["full_text"] = df["title"].str.strip() + ". " + df["text"].str.strip()

print("\n--- Example combined row ---")
print(df[["title", "text", "full_text"]].iloc[0])

df["full_text_length"] = df["full_text"].str.len()
print("\n--- full_text length stats ---")
print(df["full_text_length"].describe())

df.to_csv(STEP3_OUTPUT, index=False)
print(f"\nSaved combined dataset to: {STEP3_OUTPUT}")


# =========================================================
# STEP 4: Text Cleaning
# =========================================================
STEP4_OUTPUT = os.path.join(PROCESSED_DIR, "cleaned_data.csv")

print("\n\n=== STEP 4 ===")

before = df.shape[0]
df = df.drop_duplicates(subset=["text"]).reset_index(drop=True)
print(f"Removed {before - df.shape[0]} duplicate articles (by original 'text'). "
      f"New shape: {df.shape}")

before = df.shape[0]
df = df[df["text"].str.len() >= 20].reset_index(drop=True)
print(f"Removed {before - df.shape[0]} near-empty articles. New shape: {df.shape}")

def clean_text(text):
    text = str(text)
    text = text.lower()
    # Remove Reuters-style bylines (label-leak in this dataset)
    text = re.sub(r'^.*?\(reuters\)\s*-\s*', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # Remove Twitter handles / hashtag symbol
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove punctuation/special characters
    text = re.sub(r'[^a-z\s]', ' ', text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text

print("\nCleaning text... (this may take a minute on 44k+ rows)")
df["clean_text"] = df["full_text"].apply(clean_text)

before = df.shape[0]
df = df[df["clean_text"].str.strip().str.len() > 0].reset_index(drop=True)
print(f"Removed {before - df.shape[0]} rows that were empty after cleaning.")

print("\n--- Before vs after example ---")
print("BEFORE:", df["full_text"].iloc[0][:200])
print("\nAFTER :", df["clean_text"].iloc[0][:200])

df["clean_text_length"] = df["clean_text"].str.len()
print("\n--- clean_text length stats ---")
print(df["clean_text_length"].describe())

print("\n--- Final class distribution ---")
print(df["label_name"].value_counts())

df.to_csv(STEP4_OUTPUT, index=False)
print(f"\nCleaned dataset saved to: {STEP4_OUTPUT}")
print(f"Final shape: {df.shape}")
print(f"Columns: {list(df.columns)}")

# =========================================================
# STEP 5: Tokenization, Stopword Removal & Lemmatization
# =========================================================
STEP5_OUTPUT = os.path.join(PROCESSED_DIR, "final_preprocessed.csv")

print("\n\n=== STEP 5 ===")

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def preprocess_text(text):
    # Tokenize into words
    tokens = word_tokenize(text)

    # Remove stopwords and very short tokens (1-2 letters, usually noise)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]

    # Lemmatize (convert words to their base form, e.g. "running" -> "run")
    tokens = [lemmatizer.lemmatize(t) for t in tokens]

    return " ".join(tokens)

print("Tokenizing, removing stopwords, and lemmatizing... "
      "(this will take a few minutes on 44k+ rows)")
df["final_text"] = df["clean_text"].apply(preprocess_text)

# ---------------------------------------------------------
# Remove rows that became empty after this step
# ---------------------------------------------------------
before = df.shape[0]
df = df[df["final_text"].str.strip().str.len() > 0].reset_index(drop=True)
print(f"Removed {before - df.shape[0]} rows that were empty after preprocessing.")

# ---------------------------------------------------------
# Sanity check
# ---------------------------------------------------------
print("\n--- Before vs after example ---")
print("CLEAN TEXT:", df["clean_text"].iloc[0][:200])
print("\nFINAL TEXT:", df["final_text"].iloc[0][:200])

df["final_text_length"] = df["final_text"].str.split().apply(len)
print("\n--- Word count stats (final_text) ---")
print(df["final_text_length"].describe())

print("\n--- Final class distribution ---")
print(df["label_name"].value_counts())

# ---------------------------------------------------------
# Save final preprocessed dataset
# ---------------------------------------------------------
df.to_csv(STEP5_OUTPUT, index=False)
print(f"\nFinal preprocessed dataset saved to: {STEP5_OUTPUT}")
print(f"Final shape: {df.shape}")
print(f"Columns: {list(df.columns)}")