import streamlit as st
from utils.model_loader import load_models
from utils.resume_parser import extract_text_from_file
from utils.skill_matching import get_match_score
from utils.ats_checker import ats_checks

# Page settings
st.set_page_config(page_title="AI Resume Matcher", page_icon="🧠", layout="wide")

# Custom CSS for styling
try:
    with open("static/style.css") as css:
        st.markdown(f"<style>{css.read()}</style>", unsafe_allow_html=True)
except:
    pass  # Ignore if CSS is missing

# Title
st.title("🧠 AI-Powered Resume Scanner & Job Matcher")
st.caption("Upload your resume and job description to get a match score, ATS analysis, and approval status.")

# Load models once (spaCy + Sentence Transformer)
nlp, model = load_models()

# Upload components
col1, col2 = st.columns(2)
with col1:
    resume_file = st.file_uploader("📄 Upload Resume", type=["pdf", "docx", "txt"])
with col2:
    jd_file = st.file_uploader("💼 Upload Job Description", type=["pdf", "docx", "txt"])

# Select weights for scoring
w_sem = st.slider("🎯 Weight of Semantic Similarity", 0.0, 1.0, 0.7, 0.05)
w_skills = 1 - w_sem
st.write(f"📊 Skills Weight = **{round(w_skills, 2)}**")

# Function to classify score
def classify_score(score):
    if score >= 90:
        return "✅ Approved", "green"
    elif score >= 70:
        return "⚠️ Improvements Needed", "orange"
    else:
        return "❌ Not Approved", "red"

# Process button
if st.button("🚀 Analyze") and resume_file and jd_file:
    try:
        resume_text = extract_text_from_file(resume_file)
        jd_text = extract_text_from_file(jd_file)

        score, info = get_match_score(resume_text, jd_text, nlp, model, w_sem, w_skills)
        ats_result = ats_checks(resume_text)

        status_text, color = classify_score(score)

        # Display results
        st.markdown(f"### 🎯 Match Score: **{score}%**")
        st.markdown(f"<b style='color:{color}; font-size:20px'>{status_text}</b>", unsafe_allow_html=True)

        colA, colB, colC = st.columns(3)
        colA.metric("Semantic Similarity", info["semantic_similarity"])
        colB.metric("Skills Overlap (Jaccard)", info["skills_jaccard"])
        colC.metric("Matched Skills", len(info["matched_skills"]))

        with st.expander("✅ Matched Skills"):
            st.write(", ".join(info["matched_skills"]) or "No Matched Skills")

        with st.expander("⚠️ Missing Skills for This Role"):
            st.write(", ".join(info["missing_skills"]) or "None")

        with st.expander("➕ Extra Skills in Resume"):
            st.write(", ".join(info["extra_skills"]) or "None")

        st.subheader("🛡 ATS Resume Quality Check")
        st.write({k: "✅" if v else "❌" for k, v in ats_result.items()})

    except Exception as e:
        st.error(f"Error processing files: {e}")

elif not resume_file or not jd_file:
    st.info("Please upload both Resume and Job Description.")
