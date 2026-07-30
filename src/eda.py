# =============================================================
# STEP 2 - EXPLORATORY DATA ANALYSIS (EDA)
# CCS3356 NLP | Group 13 - Fake News Detection
# Member 1: Thisumi Tanisha (CIT-24-01-0473)
# Run 01_data_cleaning.py FIRST - this reads processed_data.csv
# =============================================================

# %% 1. Imports
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

sns.set_style("whitegrid")

# %% 2. Load cleaned data
data_path = r"C:\Users\thisu\Documents\NLP_Group_13\dataa\processed_data.csv"
df = pd.read_csv(data_path)
df["final_text"] = df["final_text"].fillna("").astype(str)
print(df.shape)
df.head()

# Where to save the plots
output_dir = r"C:\Users\thisu\Documents\NLP_Group_13\Reports\EDA"
os.makedirs(output_dir, exist_ok=True)

# %% 3. Class distribution
plt.figure(figsize=(5, 4))
sns.countplot(x="label", data=df)
plt.xticks([0, 1], ["Fake", "Real"])
plt.title("Class Distribution: Fake vs Real News")
plt.xlabel("Class")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_class_distribution.png"))
plt.show()

print(df["label"].value_counts(normalize=True) * 100)

# %% 4. Article length distribution (word count)
df["word_count"] = df["final_text"].apply(lambda x: len(str(x).split()))

plt.figure(figsize=(8, 4))
sns.histplot(data=df, x="word_count", hue="label", bins=50, kde=True, element="step")
plt.title("Article Length Distribution (after cleaning)")
plt.xlabel("Word Count")
plt.xlim(0, 1000)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_word_count_distribution.png"))
plt.show()

print(df.groupby("label")["word_count"].describe())

# %% 5. Most frequent words - Fake vs Real news
def top_words(text_series, n=20):
    all_words = " ".join(text_series).split()
    return Counter(all_words).most_common(n)

fake_top = top_words(df[df["label"] == 0]["final_text"])
real_top = top_words(df[df["label"] == 1]["final_text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fake_words, fake_counts = zip(*fake_top)
axes[0].barh(fake_words[::-1], fake_counts[::-1], color="crimson")
axes[0].set_title("Top 20 Words - Fake News")

real_words, real_counts = zip(*real_top)
axes[1].barh(real_words[::-1], real_counts[::-1], color="steelblue")
axes[1].set_title("Top 20 Words - Real News")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_top_words.png"))
plt.show()

# %% 6. Word clouds
from wordcloud import WordCloud

fake_text = " ".join(df[df["label"] == 0]["final_text"])
real_text = " ".join(df[df["label"] == 1]["final_text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

wc_fake = WordCloud(width=600, height=400, background_color="white").generate(fake_text)
axes[0].imshow(wc_fake, interpolation="bilinear")
axes[0].axis("off")
axes[0].set_title("Fake News Word Cloud")

wc_real = WordCloud(width=600, height=400, background_color="white").generate(real_text)
axes[1].imshow(wc_real, interpolation="bilinear")
axes[1].axis("off")
axes[1].set_title("Real News Word Cloud")

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "eda_wordclouds.png"))
plt.show()

# %% 7. Subject / category distribution (if the 'subject' column exists)
if "subject" in df.columns:
    plt.figure(figsize=(9, 5))
    sns.countplot(y="subject", data=df, hue="label",
                  order=df["subject"].value_counts().index)
    plt.title("News Subject/Category by Class")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "eda_subject_distribution.png"))
    plt.show()

# %% 8. Quick written insights (fill these in for your report after viewing the plots)
print("""
INSIGHTS TO WRITE UP IN YOUR REPORT:
- Is the dataset balanced or imbalanced between Fake/Real? (see class_distribution)
- Do fake articles tend to be shorter/longer than real ones? (see word_count plot)
- Which words dominate each class - do they suggest sensational/politically
  charged language in fake news vs neutral/formal language in real news?
- Are certain subjects/categories skewed heavily toward one class? This is a
  potential source of bias (also relevant to your Ethics section, Q13).
""")

print(df["final_text"].isna().sum())