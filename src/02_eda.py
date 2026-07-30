import pandas as pd
import os

import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

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

# =========================================================
# STEP 5.5: Exploratory Data Analysis (EDA)
# =========================================================
print("\n\n=== STEP 5.5: Exploratory Data Analysis ===")

EDA_DIR = r"C:\Users\imant\Documents\NLP_Group_13\Reports\EDA"
os.makedirs(EDA_DIR, exist_ok=True)

# ---------------------------------------------------------
# 1. Class Distribution
# ---------------------------------------------------------
plt.figure(figsize=(6, 5))
sns.countplot(data=df, x="label_name", palette=["#e74c3c", "#2ecc71"])
plt.title("Class Distribution: Real vs Fake News")
plt.xlabel("Class")
plt.ylabel("Number of Articles")
for i, count in enumerate(df["label_name"].value_counts()):
    plt.text(i, count + 200, str(count), ha='center', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "class_distribution.png"), dpi=150)
plt.close()
print("Saved: class_distribution.png")

class_counts = df["label_name"].value_counts()
print("\nClass counts:\n", class_counts)
print(f"Class balance: {class_counts.min() / class_counts.max():.2%} (closer to 100% = more balanced)")

# ---------------------------------------------------------
# 2. Text Length Distribution (word count, by class)
# ---------------------------------------------------------
plt.figure(figsize=(8, 5))
sns.histplot(data=df, x="final_text_length", hue="label_name", bins=50, kde=True, alpha=0.5)
plt.title("Article Length Distribution (Word Count) by Class")
plt.xlabel("Number of Words")
plt.ylabel("Number of Articles")
plt.xlim(0, df["final_text_length"].quantile(0.99))  # trim extreme outliers for readability
plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "text_length_distribution.png"), dpi=150)
plt.close()
print("Saved: text_length_distribution.png")

print("\nWord count stats by class:")
print(df.groupby("label_name")["final_text_length"].describe())

# ---------------------------------------------------------
# 3. Word Frequency Analysis (top 20 words per class)
# ---------------------------------------------------------
def get_top_words(text_series, n=20):
    all_words = " ".join(text_series).split()
    return Counter(all_words).most_common(n)

fake_words = get_top_words(df[df["label_name"] == "Fake"]["final_text"])
real_words = get_top_words(df[df["label_name"] == "Real"]["final_text"])

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

fake_df_plot = pd.DataFrame(fake_words, columns=["word", "count"])
sns.barplot(data=fake_df_plot, y="word", x="count", ax=axes[0], color="#e74c3c")
axes[0].set_title("Top 20 Words — Fake News")

real_df_plot = pd.DataFrame(real_words, columns=["word", "count"])
sns.barplot(data=real_df_plot, y="word", x="count", ax=axes[1], color="#2ecc71")
axes[1].set_title("Top 20 Words — Real News")

plt.tight_layout()
plt.savefig(os.path.join(EDA_DIR, "top_words_by_class.png"), dpi=150)
plt.close()
print("Saved: top_words_by_class.png")

print("\nTop 10 Fake news words:", fake_words[:10])
print("Top 10 Real news words:", real_words[:10])

# ---------------------------------------------------------
# 4. Word Clouds (visual, good for report/video)
# ---------------------------------------------------------
try:
    from wordcloud import WordCloud

    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    fake_text = " ".join(df[df["label_name"] == "Fake"]["final_text"])
    wc_fake = WordCloud(width=600, height=400, background_color="white",
                         colormap="Reds").generate(fake_text)
    axes[0].imshow(wc_fake, interpolation="bilinear")
    axes[0].set_title("Fake News Word Cloud")
    axes[0].axis("off")

    real_text = " ".join(df[df["label_name"] == "Real"]["final_text"])
    wc_real = WordCloud(width=600, height=400, background_color="white",
                         colormap="Greens").generate(real_text)
    axes[1].imshow(wc_real, interpolation="bilinear")
    axes[1].set_title("Real News Word Cloud")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "wordclouds.png"), dpi=150)
    plt.close()
    print("Saved: wordclouds.png")
except ImportError:
    print("wordcloud library not installed — skipping word clouds. "
          "Run 'pip install wordcloud' to enable this.")

# ---------------------------------------------------------
# 5. Subject / Category Distribution (dataset-specific column)
# ---------------------------------------------------------
if "subject" in df.columns:
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, y="subject", hue="label_name",
                   order=df["subject"].value_counts().index)
    plt.title("Article Subject Distribution by Class")
    plt.xlabel("Number of Articles")
    plt.ylabel("Subject")
    plt.tight_layout()
    plt.savefig(os.path.join(EDA_DIR, "subject_distribution.png"), dpi=150)
    plt.close()
    print("Saved: subject_distribution.png")

    print("\nSubject breakdown by class:")
    print(pd.crosstab(df["subject"], df["label_name"]))

# ---------------------------------------------------------
# EDA Insights Summary
# ---------------------------------------------------------
print("\n--- EDA Insights ---")
print(f"1. Dataset has {len(df)} articles after cleaning, "
      f"{class_counts.get('Real', 0)} Real and {class_counts.get('Fake', 0)} Fake "
      f"({class_counts.min()/class_counts.max():.1%} balance ratio).")
print(f"2. Average article length: {df['final_text_length'].mean():.0f} words "
      f"(Fake: {df[df['label_name']=='Fake']['final_text_length'].mean():.0f}, "
      f"Real: {df[df['label_name']=='Real']['final_text_length'].mean():.0f}).")
if "subject" in df.columns:
    print("3. Subject categories appear strongly correlated with class label — "
          "this is a known dataset leakage risk worth noting in the ethics section, "
          "since a model could learn to classify based on subject/source patterns "
          "rather than genuine linguistic differences between real and fake news.")
print(f"4. All EDA visualizations saved to: {EDA_DIR}")