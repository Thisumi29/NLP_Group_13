# =============================================================
# STEP 1 - DATA CLEANING & PREPROCESSING
# CCS3356 NLP | Group 13 - Fake News Detection
# Member 1: Thisumi Tanisha (CIT-24-01-0473)
# =============================================================

# %% 1. Imports
import re
import pandas as pd

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# %% 2. Load the dataset
# Dataset: "Fake and Real News Dataset" (Kaggle)
# https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset
#
# Loading from local folder: C:\Users\thisu\Documents\NLP_Group_13\dataa\raw
fake_df = pd.read_csv(r"C:\Users\thisu\Documents\NLP_Group_13\dataa\raw\Fake.csv")
true_df = pd.read_csv(r"C:\Users\thisu\Documents\NLP_Group_13\dataa\raw\True.csv")

# Label the classes: 0 = Fake, 1 = Real
fake_df["label"] = 0
true_df["label"] = 1

df = pd.concat([fake_df, true_df], axis=0).reset_index(drop=True)

# Combine title + text into one field for richer context
df["content"] = df["title"].astype(str) + " " + df["text"].astype(str)

print("Raw shape:", df.shape)
print(df["label"].value_counts())

# %% 3. Handle missing values
print("\nMissing values before:\n", df.isnull().sum())
df = df.dropna(subset=["content"]).reset_index(drop=True)

# %% 4. Text cleaning function
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)     # remove URLs
    text = re.sub(r"<.*?>", " ", text)                # remove HTML tags
    text = re.sub(r"\S+@\S+", " ", text)              # remove emails
    text = re.sub(r"[^a-z\s]", " ", text)             # remove digits/punctuation/special chars
    text = re.sub(r"\s+", " ", text).strip()           # collapse whitespace
    return text

df["clean_text"] = df["content"].apply(clean_text)

# %% 5. Remove duplicate articles
before = len(df)
df = df.drop_duplicates(subset="clean_text").reset_index(drop=True)
print(f"\nRemoved {before - len(df)} duplicate rows")
print("Shape after dedup:", df.shape)

# %% 6. Tokenization
df["tokens"] = df["clean_text"].apply(word_tokenize)

# %% 7. Stop-word removal
stop_words = set(stopwords.words("english"))
df["tokens_no_stop"] = df["tokens"].apply(
    lambda tokens: [t for t in tokens if t not in stop_words and len(t) > 2]
)

# %% 8. Lemmatization
lemmatizer = WordNetLemmatizer()
df["tokens_lemmatized"] = df["tokens_no_stop"].apply(
    lambda tokens: [lemmatizer.lemmatize(t) for t in tokens]
)

# Rejoin into a clean string (used later for TF-IDF and Keras tokenizer)
df["final_text"] = df["tokens_lemmatized"].apply(lambda tokens: " ".join(tokens))

print("\nBefore/after example:")
print("RAW  :", df["content"].iloc[0][:200])
print("CLEAN:", df["final_text"].iloc[0][:200])

# %% 9. Save cleaned data for the next steps
df.to_csv("processed_data.csv", index=False)
print("\nSaved processed_data.csv - shape:", df.shape)

# %% 9. Save cleaned data for the next steps
output_path = r"C:\Users\thisu\Documents\NLP_Group_13\dataa\processed_data.csv"
df.to_csv(output_path, index=False)
print(f"\nSaved processed data to: {output_path}")
print("Shape:", df.shape)