from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import re
import nltk
import os
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- Big Data NLP Initialization ---
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)

def perform_nlp_cleaning(text):
    if not isinstance(text, str): return ""
    # Standardizing for High-Dimensional Vector Space
    text = text.lower()
    text = re.sub(r'http\S+\s*', '', text)
    text = re.sub(r'RT|cc|#\S+|@\S+|[^a-zA-Z ]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    stop_words = set(stopwords.words("english"))
    lemmatizer = WordNetLemmatizer()
    # Feature Normalization via Lemmatization
    return " ".join([lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words])

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files or not request.form.get('job_description'):
        return jsonify({"error": "Data stream incomplete"}), 400
    
    file = request.files['file']
    job_description = request.form.get('job_description')

    try:
        # Optimization for Large Datasets (Big Data constraint)
        # Reading only first 2500 records to maintain speed on Render
        df = pd.read_csv(file, on_bad_lines='skip', encoding='latin1').head(2500)
        
        # Header Normalization (Handling BOM and spaces)
        df.columns = df.columns.str.strip().str.lower()
        
        # Automated Column Detection
        res_col = next((c for c in df.columns if 'resume' in c or 'text' in c), None)
        name_col = next((c for c in df.columns if 'name' in c), None)
        cat_col = next((c for c in df.columns if 'category' in c or 'role' in c), None)

        if not res_col:
            return jsonify({"error": "Resume Feature Column not detected"}), 400

        df = df.dropna(subset=[res_col])
        
        # [BIG DATA STEP]: Feature Extraction via TF-IDF Vectorization
        tfidf = TfidfVectorizer(max_features=3000, ngram_range=(1, 2), sublinear_tf=True)
        
        # Processing Data Stream
        cleaned_resumes = [perform_nlp_cleaning(str(r)) for r in df[res_col]]
        vector_matrix = tfidf.fit_transform(cleaned_resumes)
        
        # [BIG DATA STEP]: Similarity Measure using Cosine Distance
        job_query_vec = tfidf.transform([perform_nlp_cleaning(job_description)])
        similarity_scores = cosine_similarity(vector_matrix, job_query_vec).flatten()
        
        df['match_confidence'] = (similarity_scores * 100).round(2)
        
        # Filtering Zero-Impact Nodes
        ranked_results = df[df['match_confidence'] > 0].sort_values(by='match_confidence', ascending=False).head(15)
        
        if ranked_results.empty:
            return jsonify({"results": [], "message": "No relevant matches in current vector space"})

        # Constructing Output Payload
        results = []
        for i, row in enumerate(ranked_results.to_dict(orient='records'), 1):
            results.append({
                "rank": i,
                "name": str(row.get(name_col, f"Node-{i}")),
                "category": str(row.get(cat_col, "General Profile")).title(),
                "score": float(row.get('match_confidence', 0))
            })

        return jsonify({"results": results})

    except Exception as e:
        return jsonify({"error": f"Algorithm Error: {str(e)}"}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)
