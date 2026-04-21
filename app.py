import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
import time

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics.pairwise import cosine_similarity

# Download required resources
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

# -------------------------------
# Text Cleaning Function
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""

    text = text.lower()
    text = re.sub(r'http\S+\s*', '', text)
    text = re.sub(r'RT|cc', '', text)
    text = re.sub(r'#\S+', '', text)
    text = re.sub(r'@\S+', '', text)
    text = re.sub(r'[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()

    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()

    return " ".join([
        lemmatizer.lemmatize(w)
        for w in text.split()
        if w not in stop_words
    ])

# -------------------------------
# Main Function
# -------------------------------
def run_smart_screening():

    print("\n" + "="*80)
    print(" " * 20 + "SMART RESUME SCREENING SYSTEM")
    print("="*80)

    try:
        df = pd.read_csv("Resume.csv", on_bad_lines='skip', encoding='latin1')
        df.dropna(subset=['Resume_str', 'Category'], inplace=True)
    except Exception as e:
        print(f"CRITICAL ERROR: {e}")
        return

    # -------------------------------
    # STEP 1: Visualization
    # -------------------------------
    print("\n[STEP 1] DATASET INSIGHTS & VISUALIZATION...")

    plt.figure(figsize=(12, 8))
    sns.countplot(
        y=df['Category'],
        order=df['Category'].value_counts().index,
        palette='magma'
    )

    plt.title("RESUME CATEGORY DISTRIBUTION", fontsize=14, fontweight='bold')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()

    # -------------------------------
    # STEP 2: Model Training
    # -------------------------------
    print("\n[STEP 2] AI ENGINE INITIALIZATION...")

    df['cleaned'] = df['Resume_str'].apply(clean_text)

    tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X = tfidf.fit_transform(df['cleaned'])
    y = df['Category']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    rf_classifier = RandomForestClassifier(n_estimators=150, random_state=42)
    rf_classifier.fit(X_train, y_train)

    predictions = rf_classifier.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print(f"> AI Model Accuracy: {accuracy * 100:.2f}%")

    # -------------------------------
    # STEP 3: Job Matching
    # -------------------------------
    job_description = """
    Looking for a Senior Data Scientist or Software Engineer.
    Must have expertise in Python, Machine Learning, Deep Learning,
    Natural Language Processing (NLP), and Big Data Analytics.
    Experience with SQL, Pandas, and Scikit-learn is required.
    """

    print("\n[STEP 3] PROCESSING JOB MATCHING...")
    time.sleep(1)

    job_vec = tfidf.transform([clean_text(job_description)])
    match_scores = cosine_similarity(X, job_vec).flatten()

    df['Match_Confidence'] = (match_scores * 100).round(2)

    qualified_candidates = df[df['Match_Confidence'] >= 20] \
        .sort_values(by='Match_Confidence', ascending=False)

    print("\n" + "="*80)
    print(f"{'RANK':<6} | {'CATEGORY':<35} | {'MATCH %':<10}")
    print("="*80)

    top_results = qualified_candidates.head(10)

    for i, row in enumerate(top_results.itertuples(), 1):
        print(f"#{i:<5} | {row.Category:<35} | {row.Match_Confidence:>8}%")

    print("="*80)
    print(f"POOL SIZE: {len(df)} | MATCHED: {len(qualified_candidates)}")

    # -------------------------------
    # STEP 4: Dashboard
    # -------------------------------
    print("\n[STEP 4] GENERATING DASHBOARD...")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))

    sns.barplot(
        x=top_results['Match_Confidence'],
        y=[f"ID {idx}" for idx in top_results.index],
        ax=ax1,
        palette="rocket"
    )

    ax1.set_title("TOP 10 CANDIDATES")
    ax1.set_xlabel("Match Confidence (%)")

    sector_counts = qualified_candidates['Category'].value_counts().head(5)

    ax2.pie(
        sector_counts,
        labels=sector_counts.index,
        autopct='%1.1f%%',
        startangle=140,
        colors=sns.color_palette("viridis", len(sector_counts)),
        explode=[0.07]*len(sector_counts)
    )

    ax2.set_title("TOP MATCHED SECTORS")

    plt.tight_layout()
    plt.show()


# -------------------------------
# Run
# -------------------------------
if __name__ == "__main__":
    run_smart_screening()
