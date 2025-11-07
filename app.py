import io
import re
import base64
from pathlib import Path
from typing import List, Tuple, Dict

import streamlit as st
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from rapidfuzz import fuzz, process
from unidecode import unidecode

# ✅ Fix: Auto install/load spaCy model
import spacy
import subprocess
import sys

def load_spacy_model():
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        subprocess.run([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        return spacy.load("en_core_web_sm")

nlp = load_spacy_model()

# ---------------------------
# Skill Ontology
# ---------------------------
SKILL_SETS = {
    "Programming": ["python", "java", "javascript", "c++", "typescript", "c#", "go", "rust", "r", "sql"],
    "Data & ML": ["pandas", "numpy", "scikit-learn", "tensorflow", "pytorch", "keras", "opencv", "nlp", "power bi", "tableau"],
    "Cloud & DevOps": ["aws", "azure", "gcp", "docker", "kubernetes", "linux", "git", "jenkins"],
    "Web & APIs": ["react", "node", "flask", "django", "fastapi", "rest api"],
    "Soft Skills": ["communication", "leadership", "teamwork", "problem solving", "presentation"]
}

ALL_SKILLS = sorted({s.lower() for v in SKILL_SETS.values() for s in v})

@st.cache_resource
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = get_model()

# ---------------------------
# Resume & JD Extraction
# ---------------------------
def read_pdf(file): return pdf_extract_text(file)
def read_docx(file): return "\n".join([p.text for p in Document(file).paragraphs])
def read_txt(file): return file.read().decode(errors="ignore")

def extract_text(uploaded):
    suffix = Path(uploaded.name).suffix.lower()
    file_bytes = uploaded.read()
    buffer = io.BytesIO(file_bytes)

    if suffix == ".pdf":
        text = read_pdf(buffer)
    elif suffix in [".docx", ".doc"]:
        text = read_docx(buffer)
    else:
        text = read_txt(buffer)

    return clean_text(text)

def clean_text(text: str) -> str:
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# ---------------------------
# Skill Extraction
# ---------------------------
def extract_skills(text, skills):
    found = set()
    text_lower = text.lower()
    for s in skills:
        if re.search(rf"\b{re.escape(s)}\b", text_lower):
            found.add(s)
        else:
            best, score, _ = process.extractOne(s, [text_lower], scorer=fuzz.partial_ratio)
            if score >= 95:
                found.add(s)
    return sorted(found)

# ---------------------------
# Match Scoring
# ---------------------------
def cosine_sim(a, b):
    import numpy as np
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float((a * b).sum())

def match_score(resume_text, job_text, resume_skills, job_skills, w_sem=0.7, w_skills=0.3):
    emb_r = model.encode([resume_text], normalize_embeddings=True)[0]
    emb_j = model.encode([job_text], normalize_embeddings=True)[0]
    sem = cosine_sim(emb_r, emb_j)

    rs, js = set(resume_skills), set(job_skills)
    inter = len(rs & js)
    union = len(rs | js) or 1
    jacc = inter / union

    final_score = (w_sem * sem) + (w_skills * jacc)
    return round(final_score * 100, 2), {
        "semantic_similarity": round(sem, 3),
        "skills_jaccard": round(jacc, 3),
        "matched_skills": sorted(rs & js),
        "missing_skills": sorted(js - rs),
        "extra_skills": sorted(rs - js)
    }

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Resume Scanner", layout="wide")

st.title("🧠 AI-Powered Resume & Job Matcher")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf", "docx", "doc", "txt"])
with col2:
    job_file = st.file_uploader("Upload Job Description", type=["pdf", "docx", "doc", "txt"])

w_sem = st.slider("Semantic Weight", 0.0, 1.0, 0.7, 0.05)
st.write(f"Skill Weight = **{round(1 - w_sem, 2)}**")

if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)

    resume_skills = extract_skills(resume_text, ALL_SKILLS)
    job_skills = extract_skills(job_text, ALL_SKILLS)

    score, details = match_score(resume_text, job_text, resume_skills, job_skills, w_sem=w_sem)

    st.subheader(f"✅ Match Score: **{score}%**")
    st.metric("Semantic Similarity", details['semantic_similarity'])
    st.metric("Skills Overlap", details['skills_jaccard'])

    st.write("**Matched Skills:**", ", ".join(details['matched_skills']) or "None")
    st.write("**Missing Skills:**", ", ".join(details['missing_skills']) or "None")

else:
    st.info("Please upload both files to start analysis.")
