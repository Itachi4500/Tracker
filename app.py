# --- Imports (same as before, unchanged) ---
import io, re, base64, json
from pathlib import Path
import streamlit as st
from typing import List, Dict, Tuple
from sentence_transformers import SentenceTransformer
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from rapidfuzz import fuzz, process
from unidecode import unidecode
import numpy as np
import spacy

# ✅ Load spaCy model safely
try:
    nlp = spacy.load("en_core_web_sm")
except:
    st.error("Please run: python -m spacy download en_core_web_sm")

# --- Skills Dictionary (same as before) ---
SKILL_SETS = {...}  # (keep as in your code)
ALL_SKILLS = sorted({s.lower() for v in SKILL_SETS.values() for s in v})

@st.cache_resource
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = get_model()

# --- File Extractors (same as before) ---
# ... [Keep your read_pdf, read_docx, read_txt, extract_text functions]

# --- Skill Extraction (same) ---
# --- cosine_sim + match_score (same) ---
# --- ats_checks (same) ---

# ✅ NEW: Status Classification Function
def classify_score(score):
    if score >= 90:
        return "✅ APPROVED", "green"
    elif score >= 70:
        return "⚠️ IMPROVEMENTS NEEDED", "orange"
    else:
        return "❌ NOT APPROVED", "red"

# ✅ NEW: Improvement Suggestions Function
def generate_suggestions(info, checks):
    suggestions = []
    if info['missing_skills']:
        suggestions.append(f"Add or highlight missing skills: {', '.join(info['missing_skills'])}.")
    if not checks["Has_Contact_Info"]:
        suggestions.append("Include your email and phone number.")
    if not checks["Has_Experience_Section"]:
        suggestions.append("Add a dedicated Work Experience section.")
    if not checks["Has_Skills_Section"]:
        suggestions.append("Add a Skills section to clearly list tools and technologies.")
    if not checks["Uses_Bullets"]:
        suggestions.append("Use bullet points instead of paragraphs to improve readability.")
    return suggestions if suggestions else ["Everything looks good! Great job."]

# --- Streamlit UI ---
st.set_page_config(page_title="AI Resume Scanner", page_icon="🧠", layout="wide")
st.title("🧠 AI-Powered Resume Scanner & Job Matcher")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"])
with col2:
    job_file = st.file_uploader("💼 Upload Job Description", type=["pdf", "docx", "txt"])

w_sem = st.slider("Weight: Semantic Similarity", 0.0, 1.0, 0.7)
w_skills = 1 - w_sem

if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)

    resume_skills = extract_skills(resume_text, ALL_SKILLS)
    job_skills = extract_skills(job_text, ALL_SKILLS)

    score, info = match_score(resume_text, job_text, resume_skills, job_skills, w_sem, w_skills)

    status_label, color = classify_score(score)
    st.markdown(f"### 🎯 Match Score: **{score}%**")
    st.markdown(f"#### <span style='color:{color};'>{status_label}</span>", unsafe_allow_html=True)

    colA, colB, colC = st.columns(3)
    colA.metric("Semantic Similarity", info["semantic_similarity"])
    colB.metric("Skill Match (Jaccard)", info["skills_jaccard"])
    colC.metric("Matched Skills", len(info["matched_skills"]))

    # Display skills
    st.subheader("🧩 Skills Analysis")
    st.write("✅ Matched:", ", ".join(info["matched_skills"]) or "None")
    st.write("⚠ Missing:", ", ".join(info["missing_skills"]) or "None")
    st.write("➕ Extra:", ", ".join(info["extra_skills"]) or "None")

    # ATS Checks
    st.subheader("🛡 ATS Resume Quality Check")
    st.write({k: ("✅" if v else "❌") for k, v in ats_checks(resume_text).items()})

    # Suggestions
    st.subheader("💡 Suggestions for Improvement")
    suggestions = generate_suggestions(info, ats_checks(resume_text))
    for s in suggestions:
        st.write("- " + s)

    # Download Report
    report = {"score": score, "status": status_label, "skills": info}
    b64 = base64.b64encode(json.dumps(report, indent=2).encode()).decode()
    st.download_button("⬇ Download JSON Report", base64.b64decode(b64), "match_report.json", "application/json")

else:
    st.info("⬆ Upload both Resume & Job Description to get started.")
