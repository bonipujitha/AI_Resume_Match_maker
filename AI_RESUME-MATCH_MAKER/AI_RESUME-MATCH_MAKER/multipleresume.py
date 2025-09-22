import streamlit as st
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os
import requests
import re

# Download NLTK data
nltk.download('punkt', quiet=True)
nltk.download('stopwords', quiet=True)



# Helper functions
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ''
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    return ' '.join(token for token in word_tokenize(text.lower()) if token.isalnum() and token not in stop_words)

def request_openai_score(prompt, model_name='gpt-3.5-turbo'):
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Content-Type': 'application/json', 'Authorization': f'Bearer {OPENAI_API_KEY}'},
        json={'model': model_name, 'messages': [{"role": "user", "content": prompt}], 'max_tokens': 50, 'temperature': 0.5}
    )
    if response.status_code == 200:
        content = response.json()['choices'][0]['message']['content'].strip()
        match = re.search(r'\b(\d+(\.\d+)?)\b', content)
        return float(match.group(1)) if match else None
    return None

def calculate_score(candidate_resume, job_description, score_type):
    prompt = f"""Given the resume: {candidate_resume}
    The job description: {job_description}
    Calculate how well the candidate's {score_type} matches the job requirements. 
    Provide a single numerical score between 0 and 1, where 0 means no match and 1 means a perfect match. 
    Only return the numerical score without any explanation."""
    return request_openai_score(prompt)

def calculate_experience_score(candidate_resume, job_description):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform([candidate_resume, job_description])
    return float(cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0])

# Set page config
st.set_page_config(page_title="AI Resume Matchmaker", layout="wide")

# Custom CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

    body {
        font-family: 'Roboto', sans-serif;
        color: #333333;
        background-color: #E3F2FD;
    }

    .stApp {
        background-image: url('https://img.freepik.com/free-vector/abstract-technology-particle-background_52683-25766.jpg?w=1380&t=st=1689119118~exp=1689119718~hmac=22ea9a8c71b0bba7678f6e9c4106e1f780d4c83600268849a7b64af3a93b9343');
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
    }

    .main {
        background-color: rgba(255, 255, 255, 0.85);
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.2);
        backdrop-filter: blur(4px);
    }

    h1, h2, h3, h4 {
        color: #1565C0;
        text-align: center;
        font-weight: 700;
    }

    h1 {
        font-size: 3rem;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }

    h3 {
        font-size: 1.5rem;
        margin-bottom: 1.5rem;
    }

    .stFileUploader > div {
        border: 2px dashed #1565C0;
        border-radius: 10px;
        background-color: rgba(21, 101, 192, 0.1);
        transition: all 0.3s ease;
    }

    .stFileUploader > div:hover {
        background-color: rgba(21, 101, 192, 0.2);
        box-shadow: 0 0 15px rgba(21, 101, 192, 0.5);
    }

    .stButton > button {
        background-color: #1565C0;
        color: white;
        font-weight: 700;
        padding: 0.75rem 2.5rem;
        border-radius: 30px;
        border: none;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stButton > button:hover {
        background-color: #0D47A1;
        box-shadow: 0 0 20px rgba(21, 101, 192, 0.7);
    }

    .result-card {
        background-color: rgba(255, 255, 255, 0.9);
        border-radius: 10px;
        padding: 0.75rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(21, 101, 192, 0.3);
        max-width: 600px;
        margin-left: auto;
        margin-right: auto;
    }

    .result-card h4 {
        color: #1565C0;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
    }

    .score-item {
        display: inline-block;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
        padding: 0.25rem 0.5rem;
        border-radius: 5px;
        font-weight: bold;
        font-size: 0.9rem;
    }

    .overall-match {
        font-size: 1.1rem;
        font-weight: bold;
        color: #1565C0;
        margin-top: 0.25rem;
        margin-bottom: 0.5rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# Title and Subtitle
st.markdown("<h1>AI Resume Matchmaker 🤖</h1>", unsafe_allow_html=True)

# File uploader for resume
st.markdown("<h3>Upload Candidate Resume</h3>", unsafe_allow_html=True)
uploaded_resume = st.file_uploader("", type="pdf", key="resume")

# Path to the job descriptions folder
jds_folder = 'D:\\infosys\\job_descriptions'  # Replace with the path to your folder containing job descriptions

if uploaded_resume:
    if st.button('Analyze Matches'):
        resume_text = preprocess_text(read_pdf(uploaded_resume))
        jd_results = []

        # Loop through job descriptions in the folder
        for jd_file in os.listdir(jds_folder):
            if jd_file.endswith(".pdf"):
                jd_path = os.path.join(jds_folder, jd_file)
                job_desc_text = preprocess_text(read_pdf(jd_path))
                
                educational_score = calculate_score(resume_text, job_desc_text, 'educational background')
                skills_score = calculate_score(resume_text, job_desc_text, 'skills')
                experience_score = calculate_experience_score(resume_text, job_desc_text)
                
                if all(score is not None for score in [educational_score, skills_score, experience_score]):
                    final_score = 0.4 * educational_score + 0.3 * experience_score + 0.3 * skills_score
                    overall_score = (educational_score + skills_score + experience_score) / 3
                    jd_results.append({
                        'job_title': jd_file,
                        'score': round(final_score * 100, 2),
                        'overall_score': round(overall_score * 100, 2),
                        'educational_score': round(educational_score * 100, 2),
                        'experience_score': round(experience_score * 100, 2),
                        'skills_score': round(skills_score * 100, 2)
                    })
        
        # Sort results by score in descending order
        jd_results.sort(key=lambda x: x['score'], reverse=True)

        # Print top 5 results
        st.markdown("<h3>Top Most Job Matches</h3>", unsafe_allow_html=True) 
        top_5_results = [result for result in jd_results if 50 <= result['overall_score'] <= 100][:5]
        for result in top_5_results:
           st.markdown(f"""
    <div class="result-card">
        <h4 style="color: #000000;">{result['job_title']}</h4>
        <div class="overall-match" style="color: #33A1C9;">Overall Match: {result['overall_score']}%</div>
        <div class="score-item" style="background-color: rgba(51, 161, 201, 0.2); color: #33A1C9;">
        <div class="score-item" style="background-color: rgba(0, 128, 0, 0.2); color: #008000;">
            Education: {result['educational_score']}%
        </div>
        <div class="score-item" style="background-color: rgba(255, 165, 0, 0.2); color: #FFA500;">
            Experience: {result['experience_score']}%
        </div>
        <div class="score-item" style="background-color: rgba(138, 43, 226, 0.2); color: #8A2BE2;">
            Skills: {result['skills_score']}%
        </div>
    </div>
    """, unsafe_allow_html=True)