from flask import Flask, render_template, request, jsonify
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os

app = Flask(__name__)

nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
CLEAN_PATTERN = re.compile(r'[^a-z\s]')

def fast_clean(text):
    if not text or pd.isna(text): return ""
    text = str(text).lower()
    text = CLEAN_PATTERN.sub(' ', text)

    return " ".join([STEMMER.stem(w) for w in text.split() if w not in STOPWORDS and len(w) > 2])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or not request.form.get('job_description'):
        return jsonify({"error": "Missing file or description"}), 400
    
    file = request.files['file']
    jd = request.form.get('job_description')

    try:
        # Optimization: Sirf headers read karke column detect karna
        header_df = pd.read_csv(file, nrows=0, encoding='latin1')
        file.seek(0)
        
        all_cols = [str(c).strip().lower() for c in header_df.columns]
        col_map = {str(c).strip().lower(): c for c in header_df.columns}
        
        name_key = next((k for k in all_cols if 'name' in k), None)
        res_key = next((k for k in all_cols if 'resume' in k or 'text' in k), None)
        cat_key = next((k for k in all_cols if 'category' in k or 'role' in k), None)

        if not res_key:
            return jsonify({"error": "Resume text column not found"}), 400


        target_cols = [col_map[k] for k in [name_key, res_key, cat_key] if k]
        df = pd.read_csv(file, usecols=target_cols, on_bad_lines='skip', encoding='latin1').head(5000)
        

        df.columns = [str(c).strip().lower() for c in df.columns]

        cleaned_resumes = [fast_clean(r) for r in df[res_key]]
        tfidf = TfidfVectorizer(max_features=1500)
        matrix = tfidf.fit_transform(cleaned_resumes)
        job_vec = tfidf.transform([fast_clean(jd)])
        
        scores = cosine_similarity(matrix, job_vec).flatten()
        df['match_score'] = (scores * 100).round(2)
        

        filtered_df = df[df['match_score'] > 0].sort_values(by='match_score', ascending=False).head(15)
        
        if filtered_df.empty:
            return jsonify({"results": [], "message": "No matching candidates found above 0%"})

        results = []
        for i, row in enumerate(filtered_df.to_dict(orient='records'), 1):
            results.append({
                "rank": i,
                "name": str(row.get(name_key, f"Candidate {i}")),
                "category": str(row.get(cat_key, "General Profile")).title(),
                "score": float(row['match_score'])
            })

        del df, matrix, cleaned_resumes, filtered_df
        
        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
