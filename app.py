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

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_data(text):
    text = str(text).lower()
    text = re.sub(r'[^a-z]', ' ', text)
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or not request.form.get('job_description'):
        return jsonify({"error": "Data Missing"}), 400
    
    try:
        df = pd.read_csv(request.files['file'], on_bad_lines='skip', encoding='latin1').head(500)
        jd = request.form.get('job_description')

        res_col = next((c for c in ['Resume_str', 'Resume', 'resume_text'] if c in df.columns), None)
        name_col = 'Name' if 'Name' in df.columns else None
        cat_col = 'Category' if 'Category' in df.columns else None

        if not res_col:
            return jsonify({"error": "Resume content column not found"}), 400

        df = df.dropna(subset=[res_col])
        cleaned_resumes = [clean_data(r) for r in df[res_col]]
        
        tfidf = TfidfVectorizer(max_features=2000, ngram_range=(1,2))
        matrix = tfidf.fit_transform(cleaned_resumes)
        
        job_vec = tfidf.transform([clean_data(jd)])
        scores = cosine_similarity(matrix, job_vec).flatten()
        df['Match'] = (scores * 100).round(2)
        
        ranked = df.sort_values(by='Match', ascending=False).head(10)
        
        results = []
        for i, row in enumerate(ranked.itertuples(), 1):
            results.append({
                "rank": i,
                "name": getattr(row, name_col) if name_col else f"Candidate {row.Index}",
                "category": getattr(row, cat_col) if cat_col else "N/A",
                "score": float(row.Match)
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
