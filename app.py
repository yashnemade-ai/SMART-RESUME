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

# NLTK resources setup
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_data(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^a-z]', ' ', text)
    tokens = text.split()
    return " ".join([lemmatizer.lemmatize(t) for t in tokens if t not in stop_words])

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or not request.form.get('job_description'):
        return jsonify({"error": "CSV file and Job Description required"}), 400
    
    file = request.files['file']
    jd = request.form.get('job_description')

    try:
        # Optimization: head(500) for Render's limited RAM
        df = pd.read_csv(file, on_bad_lines='skip', encoding='latin1').head(500)

        # --- STEP 1: FIX COLUMN NAMES (IMPORTANT) ---
        # Sabhi column names se spaces hatana aur lowercase karna
        df.columns = df.columns.str.strip().str.lower()

        # Dynamic Column Detection
        # Ab hum lowercase mein check karenge
        name_col = next((c for c in df.columns if c in ['name', 'candidate name', 'candidate_name', 'names']), None)
        res_col = next((c for c in df.columns if c in ['resume_str', 'resume', 'resume_text', 'text', 'content']), None)
        cat_col = next((c for c in df.columns if c in ['category', 'job role', 'role', 'department']), None)

        if not res_col:
            return jsonify({"error": "Resume content column not found in CSV. Use 'Resume_str' or 'Resume'."}), 400

        # --- STEP 2: CLEAN DATA ---
        df = df.dropna(subset=[res_col])
        cleaned_resumes = [clean_data(str(r)) for r in df[res_col]]
        
        # --- STEP 3: ML PROCESSING ---
        # sublinear_tf=True helps in better keyword importance
        tfidf = TfidfVectorizer(max_features=2500, ngram_range=(1,2), sublinear_tf=True)
        matrix = tfidf.fit_transform(cleaned_resumes)
        
        job_vec = tfidf.transform([clean_data(jd)])
        scores = cosine_similarity(matrix, job_vec).flatten()
        
        df['match_score'] = (scores * 100).round(2)
        
        # Sort by Match Score
        ranked_df = df.sort_values(by='match_score', ascending=False).head(10)
        
        # --- STEP 4: FORMAT RESULTS ---
        results = []
        for i, row in enumerate(ranked_df.to_dict(orient='records'), 1):
            results.append({
                "rank": i,
                # Safe access using normalized column names
                "name": str(row.get(name_col, f"Candidate {i-1}")),
                "category": str(row.get(cat_col, "General")),
                "score": float(row.get('match_score', 0))
            })

        return jsonify({"results": results})

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": f"Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
