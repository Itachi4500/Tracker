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

import spacy
nlp = spacy.load("en_core_web_sm")

# ---------------------------
# Small skills ontology (extend as you like)
# ---------------------------
SKILL_SETS = {
    "Programming": [
        "python","java","javascript","typescript","c++","c#","go","rust","r","matlab","sql","scala","bash","powershell"
    ],
    "Data & ML": [
        "pandas","numpy","scikit-learn","tensorflow","pytorch","keras","xgboost","lightgbm","opencv",
        "nlp","spacy","transformers","hugging face","statsmodels","matplotlib","plotly","seaborn","power bi","tableau",
        "snowflake","databricks","airflow","dbt","mlflow"
    ],
    "Cloud & DevOps": [
        "aws","azure","gcp","docker","kubernetes","terraform","ansible","linux","git","ci/cd","jenkins","github actions"
    ],
    "Web & APIs":[
        "react","node","express","flask","django","fastapi","rest","graphql"
    ],
    "Soft Skills":[
        "communication","leadership","stakeholder management","problem solving","presentation","teamwork","mentoring"
    ],
}

# Normalize skill list
ALL_SKILLS = sorted({s.lower() for v in SKILL_SETS.values() for s in v})

# ---------------------------
# Embedding model (small, fast & effective)
# ---------------------------
@st.cache_resource
def get_model():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

model = get_model()

# ---------------------------
# Helpers: file parsing
# ---------------------------
def read_txt(file: io.BytesIO) -> str:
    return file.read().decode(errors="ignore")

def read_docx(file: io.BytesIO) -> str:
    file.seek(0)
    doc = Document(file)
    return "\n".join(p.text for p in doc.paragraphs)

def read_pdf(file: io.BytesIO) -> str:
    file.seek(0)
    return pdf_extract_text(file)

def extract_text(uploaded) -> str:
    suffix = Path(uploaded.name).suffix.lower()
    data = uploaded.read()
    file = io.BytesIO(data)
    if suffix == ".pdf":
        text = read_pdf(file)
    elif suffix in [".docx", ".doc"]:
        text = read_docx(file)
    else:
        text = read_txt(io.BytesIO(data))
    return clean_text(text)

def clean_text(text: str) -> str:
    text = unidecode(text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

# ---------------------------
# Skills extraction
# ---------------------------
def extract_skills(text: str, skills: List[str]) -> List[str]:
    # Fuzzy match skills; tolerate minor spelling/case differences
    found = set()
    low = text.lower()
    for s in skills:
        # Try exact token hit first
        if re.search(rf"\b{re.escape(s)}\b", low):
            found.add(s)
        else:
            # Fuzzy fallback
            best, score, _ = process.extractOne(s, [low], scorer=fuzz.partial_ratio)
            if score >= 95:  # conservative threshold
                found.add(s)
    return sorted(found)

# ---------------------------
# Scoring
# ---------------------------
def cosine_sim(a, b) -> float:
    # model.encode returns numpy; SentenceTransformer has util, but keep it simple
    import numpy as np
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float((a * b).sum())

def match_score(resume_text: str, job_text: str,
                resume_skills: List[str], job_skills: List[str],
                w_sem: float = 0.7, w_skills: float = 0.3) -> Tuple[float, Dict]:
    emb_resume = model.encode([resume_text], normalize_embeddings=True)[0]
    emb_job = model.encode([job_text], normalize_embeddings=True)[0]
    sem = cosine_sim(emb_resume, emb_job)  # 0..1 approx

    # skills overlap Jaccard
    rs, js = set(resume_skills), set(job_skills)
    inter = len(rs & js)
    union = len(rs | js) or 1
    jacc = inter / union

    final = (w_sem * sem) + (w_skills * jacc)
    explain = {
        "semantic_similarity": round(sem, 3),
        "skills_jaccard": round(jacc, 3),
        "resume_skills_count": len(rs),
        "job_skills_count": len(js),
        "matched_skills": sorted(rs & js),
        "missing_skills": sorted(js - rs),
        "extra_skills": sorted(rs - js)
    }
    return round(final * 100, 1), explain  # percentage

# ---------------------------
# Simple ATS sanity checks
# ---------------------------
def ats_checks(text: str) -> Dict[str, bool]:
    checks = {
        "Has_Contact_Info": bool(re.search(r"\b(\+?\d[\d\-\s]{7,}\d|@\w+\.)", text)),
        "Has_Experience_Section": bool(re.search(r"\bexperience|work history|employment\b", text, re.I)),
        "Has_Education_Section": bool(re.search(r"\beducation|degree|bachelor|master|ph\.?d\b", text, re.I)),
        "Has_Skills_Section": bool(re.search(r"\bskills\b", text, re.I)),
        "Uses_Bullets": "•" in text or bool(re.search(r"(\n- |\n\* )", text)),
    }
    return checks

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="AI Resume Scanner & Job Matcher", page_icon="🧠", layout="wide")

st.title("🧠 AI-Powered Resume Scanner & Job Matcher (Python)")
st.caption("Upload a resume and a job description. Get a match score, matched/missing skills, and ATS hints.")

col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("Upload Resume (PDF/DOCX/TXT)", type=["pdf","docx","doc","txt"], key="resume")
with col2:
    job_file = st.file_uploader("Upload Job Description (PDF/DOCX/TXT)", type=["pdf","docx","doc","txt"], key="job")

w_sem = st.slider("Weight: Semantic Content", 0.0, 1.0, 0.7, 0.05)
w_skills = round(1.0 - w_sem, 2)
st.write(f"Weight: Skills Overlap = **{w_skills}**")

if resume_file and job_file:
    resume_text = extract_text(resume_file)
    job_text = extract_text(job_file)

    # Extract skills (also pull noun chunks as potential skills/keywords)
    job_doc = nlp(job_text)
    jd_chunks = {c.text.lower().strip() for c in job_doc.noun_chunks if 2 <= len(c.text) <= 40}
    job_skills = sorted(set(extract_skills(job_text, ALL_SKILLS)) | (jd_chunks & set(ALL_SKILLS)))

    resume_skills = extract_skills(resume_text, ALL_SKILLS)

    score, info = match_score(resume_text, job_text, resume_skills, job_skills, w_sem=w_sem, w_skills=w_skills)

    st.subheader(f"Overall Match: **{score}%**")
    c1, c2, c3 = st.columns(3)
    c1.metric("Semantic Similarity", info["semantic_similarity"])
    c2.metric("Skills Overlap (Jaccard)", info["skills_jaccard"])
    c3.metric("Matched Skills", len(info["matched_skills"]))

    with st.expander("✅ Matched Skills"):
        if info["matched_skills"]:
            st.write(", ".join(info["matched_skills"]))
        else:
            st.write("_None yet_")

    with st.expander("⚠️ Missing Skills (from JD)"):
        if info["missing_skills"]:
            st.write(", ".join(info["missing_skills"]))
            st.info("Tip: Add a ‘Skills’ section and quantify these where applicable.")
        else:
            st.write("_None_")

    with st.expander("➕ Extra Skills (in resume but not in JD)"):
        if info["extra_skills"]:
            st.write(", ".join(info["extra_skills"]))
        else:
            st.write("_None_")

    st.subheader("ATS Sanity Checks")
    checks = ats_checks(resume_text)
    st.write({k: ("✅" if v else "❌") for k, v in checks.items()})

    st.subheader("Highlights")
    # Lightweight highlighting (show top sentences that align to JD)
    import numpy as np
    resume_sents = [s.text.strip() for s in nlp(resume_text).sents if len(s.text.split()) > 3]
    # Encode once
    r_emb = model.encode(resume_sents, normalize_embeddings=True)
    jd_emb = model.encode([job_text], normalize_embeddings=True)[0]
    sims = np.dot(r_emb, jd_emb)
    top_idx = np.argsort(-sims)[:5]
    for i in top_idx:
        st.write(f"- {resume_sents[i]} _(similarity: {sims[i]:.3f})_")

    # Downloadable analysis
    import json
    report = {
        "overall_match_percent": score,
        "weights": {"semantic": w_sem, "skills": w_skills},
        "semantic_similarity": info["semantic_similarity"],
        "skills_jaccard": info["skills_jaccard"],
        "matched_skills": info["matched_skills"],
        "missing_skills": info["missing_skills"],
        "extra_skills": info["extra_skills"],
        "ats_checks": checks,
    }
    b64 = base64.b64encode(json.dumps(report, indent=2).encode()).decode()
    st.download_button("Download JSON Report", data=base64.b64decode(b64), file_name="match_report.json", mime="application/json")

else:
    st.info("Upload both a resume and a job description to get started.")
