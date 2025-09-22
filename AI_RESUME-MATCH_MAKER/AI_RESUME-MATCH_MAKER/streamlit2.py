import streamlit as st
import PyPDF2
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
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

# Update the custom CSS section with this new code:
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;700&display=swap');

body {
    font-family: 'Roboto', sans-serif;
    color: #E0E0E0;
}

.stApp {
    background-image: url('https://img.freepik.com/free-vector/abstract-technology-particle-background_52683-25766.jpg');
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

h1, h2, h3 {
    color: #4a0e78;
    text-align: center;
    font-weight: 700;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
}

h1 {
    font-size: 3.5rem;
    margin-bottom: 2rem;
    background: linear-gradient(45deg, #4a0e78, #6a0dad);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

h3 {
    font-size: 1.8rem;
    margin-bottom: 1.5rem;
    color: #6a0dad;
}

.stFileUploader {
    width: 100%;
    margin: 1rem 0;
}

.stFileUploader > div {
    height: 200px;
    border: 2px dashed #4a0e78;
    border-radius: 15px;
    background-color: rgba(77, 182, 172, 0.1);
    transition: all 0.3s ease;
}

.stFileUploader > div:hover {
    background-color: rgba(77, 182, 172, 0.2);
    box-shadow: 0 0 15px rgba(77, 182, 172, 0.5);
}

.stFileUploader > div > div {
    color: #4DB6AC;
}

.stButton > button {
    background: linear-gradient(45deg,#4a0e78, #6a0dad );
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
    background: linear-gradient(45deg, #26A69A, #00897B);
    box-shadow: 0 0 20px rgba(77, 182, 172, 0.7);
    transform: translateY(-2px);
}

.stProgress > div > div {
    background-color: #4DB6AC;
}

.stMetric {
    background-color: rgba(38, 166, 154, 0.1);
    border-radius: 15px;
    padding: 1.5rem;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    border: 1px solid rgba(77, 182, 172, 0.3);
}

.stMetric label {
    color:#4a0e78;
    font-weight: 700;
    font-size: 1.2rem;
}

.stMetric .metric-value {
    font-size: 2.5rem;
    font-weight: 700;
    color: #6a0dad;
    text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
}

</style>
""", unsafe_allow_html=True)

# Update the title and add a subtitle
st.markdown("<h1>AI Resume Matchmaker</h1>", unsafe_allow_html=True)
st.markdown("<h3>Powered by Advanced AI Technology</h3>", unsafe_allow_html=True)
st.markdown("<br><br><br>", unsafe_allow_html=True) 
# File uploaders
col1, col2 = st.columns(2)

with col1:
    st.markdown("<h3>Upload Resume</h3>", unsafe_allow_html=True)
    uploaded_resume = st.file_uploader("", type="pdf", key="resume")

with col2:
    st.markdown("<h3>Upload Job Description</h3>", unsafe_allow_html=True)
    uploaded_jd = st.file_uploader("", type="pdf", key="jd")

if uploaded_resume and uploaded_jd:
    if st.button('Analyze Match'):
        with st.spinner('Analyzing...'):
            resume_text = preprocess_text(read_pdf(uploaded_resume))
            job_desc_text = preprocess_text(read_pdf(uploaded_jd))
            
            education_score = calculate_score(resume_text, job_desc_text, 'educational background')
            skills_score = calculate_score(resume_text, job_desc_text, 'skills')
            experience_score = calculate_experience_score(resume_text, job_desc_text)
            
            if all(score is not None for score in [education_score, skills_score, experience_score]):
                overall_match = 0.4 * education_score + 0.3 * experience_score + 0.3 * skills_score
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric('Work Experience', f"{experience_score:.2f}")
                with col2:
                    st.metric('Education', f"{education_score:.2f}")
                with col3:
                    st.metric('Skills', f"{skills_score:.2f}")
                
                st.markdown(f"<h2>Overall Match: {overall_match:.2f}</h2>", unsafe_allow_html=True)
                
                st.progress(overall_match)
                st.markdown(f"<h3>{int(overall_match * 100)}% Match</h3>", unsafe_allow_html=True)
            else:
                st.error("Failed to calculate scores. Please try again.")