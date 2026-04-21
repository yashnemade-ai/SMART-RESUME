import streamlit as st
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics.pairwise import cosine_similarity

st.set_page_config(page_title="Smart Resume Screener", layout="wide")

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

def clean_resume(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

st.title("🎯 Smart Resume Screening & Job Matching")
st.markdown("---")

uploaded_file = st.sidebar.file_uploader("Upload Resume Dataset (CSV)", type=["csv"])
job_description = st.text_area("Enter Job Description:", height=150, placeholder="Looking for a Python Developer...")

if uploaded_file and job_description:
    df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='latin1')
    df.dropna(subset=['Resume_str', 'Category'], inplace=True)
    
    with st.spinner('AI Engine Training...'):
        df['cleaned'] = df['Resume_str'].apply(clean_resume)
        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
        X = tfidf.fit_transform(df['cleaned'])
        
        job_vec = tfidf.transform([clean_resume(job_description)])
        scores = cosine_similarity(X, job_vec).flatten()
        df['Match_Score'] = (scores * 100).round(2)
        
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("📊 Dataset Overview")
        fig1, ax1 = plt.subplots()
        sns.countplot(y=df['Category'], order=df['Category'].value_counts().index[:10], palette='viridis', ax=ax1)
        st.pyplot(fig1)

    matches = df[df['Match_Score'] >= 15].sort_values(by='Match_Score', ascending=False).head(10)

    with col2:
        st.subheader("🏆 Top Candidate Matches")
        if not matches.empty:
            st.dataframe(matches[['Category', 'Match_Score']], use_container_width=True)
        else:
            st.warning("No matches found above 15%")

    if not matches.empty:
        st.markdown("---")
        st.subheader("📈 Match Analysis")
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sns.barplot(x=matches['Match_Score'], y=[f"ID {i}" for i in matches.index], palette="magma", ax=ax2)
        st.pyplot(fig2)

else:
    st.info("Please upload a CSV file and enter a job description to begin.")
