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

# Fast Setup
nltk.download('stopwords', quiet=True)
STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()
CLEAN_PATTERN = re.compile(r'[^a-z\s]')

def fast_clean(text):
    text = str(text).lower()
    text = CLEAN_PATTERN.sub(' ', text)
    words = [STEMMER.stem(w) for w in text.split() if w not in STOPWORDS and len(w) > 2]
    return " ".join(words)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or not request.form.get('job_description'):
        return jsonify({"error": "Missing input"}), 400
    
    file = request.files['file']
    jd = request.form.get('job_description')

    try:
        # Load CSV
        df = pd.read_csv(file, on_bad_lines='skip', encoding='latin1').head(500)
        
        # --- SABSE ZAROORI FIX: HEADERS CLEANING ---
        # Column names se spaces hatana aur sabko lowercase banana
        df.columns = [str(c).strip().lower() for c in df.columns]

        # Name column detect karne ka foolproof tareeka
        name_col = next((c for c in df.columns if 'name' in c), None)
        res_col = next((c for c in df.columns if 'resume' in c or 'text' in c), None)
        cat_col = next((c for c in df.columns if 'category' in c or 'role' in c), None)

        if not res_col:
            return jsonify({"error": "Resume text column not found"}), 400

        # Fast Analysis
        resumes = df[res_col].fillna("").astype(str).tolist()
        cleaned_resumes = [fast_clean(r) for r in resumes]
        
        tfidf = TfidfVectorizer(max_features=1000, stop_words='english')
        matrix = tfidf.fit_transform(cleaned_resumes)
        job_vec = tfidf.transform([fast_clean(jd)])
        
        scores = cosine_similarity(matrix, job_vec).flatten()
        df['match_score'] = (scores * 100).round(2)
        
        top_df = df.sort_values(by='match_score', ascending=False).head(10)
        
        # --- DATA EXTRACTION FIX ---
        results = []
        # to_dict('records') se column names perfectly access hote hain
        records = top_df.to_dict(orient='records')
        
        for i, row in enumerate(records, 1):
            # Agar name_col mila toh uska data lo, warna Candidate ID dikhao
            candidate_real_name = str(row.get(name_col)) if name_col and pd.notna(row.get(name_col)) else f"Candidate {i}"
            
            results.append({
                "rank": i,
                "name": candidate_real_name,
                "category": str(row.get(cat_col, "General Profile")).title(),
                "score": float(row['match_score'])
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=int(os.environ.get("PORT", 5000)))
