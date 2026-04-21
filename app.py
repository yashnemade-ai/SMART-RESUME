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
from sklearn.metrics.pairwise import cosine_similarity

# --- Page Config ---
st.set_page_config(page_title="AI Resume Screener", page_icon="🎯", layout="wide")

# --- NLTK Setup ---
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
    nltk.download('wordnet')

download_nltk()

def clean_text(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r'http\S+\s*', ' ', text)
    text = re.sub(r'[^a-zA-Z]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

# --- Sidebar UI ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/942/942799.png", width=100)
st.sidebar.title("AI Recruiter Control")
st.sidebar.info("Upload your resume database and find the best candidates using AI.")

# --- Main UI ---
st.title("🎯 Smart Resume Screening & Job Matching")
st.markdown("### Powered by Advanced Big Data Analytics")

uploaded_file = st.sidebar.file_uploader("Upload Resume Dataset (CSV)", type=["csv"])
job_description = st.text_area("📋 Paste Job Description Here:", height=150, placeholder="e.g. Looking for a Data Scientist with Python and ML skills...")

if uploaded_file and job_description:
    # Processing Data
    df = pd.read_csv(uploaded_file, on_bad_lines='skip', encoding='latin1')
    df.dropna(subset=['Resume_str', 'Category'], inplace=True)
    
    with st.status("AI Engine at work...", expanded=True) as status:
        st.write("Cleaning resumes...")
        df['cleaned'] = df['Resume_str'].apply(clean_text)
        
        st.write("Extracting features (TF-IDF)...")
        tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1,2))
        X = tfidf.fit_transform(df['cleaned'])
        
        st.write("Calculating similarity scores...")
        job_vec = tfidf.transform([clean_text(job_description)])
        scores = cosine_similarity(X, job_vec).flatten()
        df['Match_Score'] = (scores * 100).round(2)
        status.update(label="Analysis Complete!", state="complete", expanded=False)

    # Filter top candidates
    matches = df[df['Match_Score'] >= 15].sort_values(by='Match_Score', ascending=False)

    # --- Dashboard Layout ---
    tab1, tab2, tab3 = st.tabs(["🏆 Top Candidates", "📊 Analytics", "📄 Raw Data"])

    with tab1:
        st.subheader("Top Recommended Profiles")
        if not matches.empty:
            # Displaying a clean table
            st.dataframe(
                matches[['Category', 'Match_Score']].head(10).style.background_gradient(cmap='Greens'),
                use_container_width=True
            )
        else:
            st.error("No resumes match the required skills (Threshold < 15%)")

    with tab2:
        col1, col2 = st.columns(2)
        with col1:
            st.write("#### Match Strength Distribution")
            fig, ax = plt.subplots()
            sns.barplot(x=matches['Match_Score'].head(10), y=[f"C-{i+1}" for i in range(min(10, len(matches)))], palette="rocket", ax=ax)
            st.pyplot(fig)
        
        with col2:
            st.write("#### Top Matched Sectors")
            fig2, ax2 = plt.subplots()
            matches['Category'].value_counts().head(5).plot.pie(autopct='%1.1f%%', ax=ax2, colors=sns.color_palette("Set2"))
            st.pyplot(fig2)

    with tab3:
        st.write("Full Processed Dataset View")
        st.dataframe(df[['Category', 'Resume_str', 'Match_Score']])

else:
    st.warning("👈 Please upload a CSV file and enter a job description in the sidebar.")
    st.image("https://raw.githubusercontent.com/andymeneely/git-data-science/master/images/ds-workflow.png", use_column_width=True)
