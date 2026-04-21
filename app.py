from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z]', ' ', text)
    words = text.split()
    return " ".join([lemmatizer.lemmatize(w) for w in words if w not in stop_words])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400
    
    file = request.files['file']
    job_desc = request.form.get('job_description', '')

    if file.filename == '' or not job_desc:
        return jsonify({"error": "Missing file or job description"}), 400

    try:
        # User ki uploaded CSV load karna
        df = pd.read_csv(file, on_bad_lines='skip', encoding='latin1')
        
        # Column detect karna (Resume_str ya Resume)
        possible_cols = ['Resume_str', 'Resume', 'resume_text', 'Resume_Text']
        res_col = next((c for c in possible_cols if c in df.columns), None)
        
        if not res_col:
            return jsonify({"error": f"CSV must have one of these columns: {possible_cols}"}), 400

        df = df.dropna(subset=[res_col])
        
        # NLP Processing
        df['cleaned'] = df[res_col].apply(clean_text)
        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
        tfidf_matrix = tfidf.fit_transform(df['cleaned'])
        
        # Job Matching
        job_clean = clean_text(job_desc)
        job_vec = tfidf.transform([job_clean])
        scores = cosine_similarity(tfidf_matrix, job_vec).flatten()
        
        df['Match_Score'] = (scores * 100).round(2)
        
        # Ranking Logic
        top_matches = df.sort_values(by='Match_Score', ascending=False).head(15)
        
        # Agar 'Category' column hai toh wo bhi bhejenge
        cat_col = 'Category' if 'Category' in df.columns else None
        
        results = []
        for i, row in enumerate(top_matches.itertuples(), 1):
            results.append({
                "rank": i,
                "category": getattr(row, cat_col) if cat_col else f"Candidate {row.Index}",
                "score": row.Match_Score
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
