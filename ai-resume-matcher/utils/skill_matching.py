import re
from typing import List, Dict, Tuple
from rapidfuzz import fuzz, process
import numpy as np

SKILL_SETS = {
    "Programming": ["python","java","javascript","typescript","c++","c#","go","rust","r","matlab","sql","scala","bash","powershell"],
    "Data & ML": ["pandas","numpy","scikit-learn","tensorflow","pytorch","keras","xgboost","lightgbm","opencv","nlp","spacy","transformers","hugging face","statsmodels","matplotlib","plotly","seaborn","power bi","tableau","snowflake","databricks","airflow","dbt","mlflow"],
    "Cloud & DevOps": ["aws","azure","gcp","docker","kubernetes","terraform","ansible","linux","git","ci/cd","jenkins","github actions"],
    "Web & APIs": ["react","node","express","flask","django","fastapi","rest","graphql"],
    "Soft Skills": ["communication","leadership","stakeholder management","problem solving","presentation","teamwork","mentoring"],
}
ALL_SKILLS = sorted({s.lower() for v in SKILL_SETS.values() for s in v})

def _extract_skills(text: str, skills: List[str]) -> List[str]:
    found = set()
    low = text.lower()
    for s in skills:
        # Exact token match
        if re.search(rf"\b{re.escape(s)}\b", low):
            found.add(s)
        else:
            # Fuzzy fallback (conservative)
            best, score, _ = process.extractOne(s, [low], scorer=fuzz.partial_ratio)
            if score >= 95:
                found.add(s)
    return sorted(found)

def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a / (np.linalg.norm(a) + 1e-9)
    b = b / (np.linalg.norm(b) + 1e-9)
    return float((a * b).sum())

def _jaccard(a, b) -> float:
    A, B = set(a), set(b)
    inter = len(A & B)
    union = len(A | B) or 1
    return inter / union

def get_match_score(resume_text: str, jd_text: str, nlp, model,
                    w_sem: float = 0.7, w_skills: float = 0.3) -> Tuple[float, Dict]:
    # Skills
    resume_skills = _extract_skills(resume_text, ALL_SKILLS)
    jd_skills = _extract_skills(jd_text, ALL_SKILLS)

    # (Optional) noun chunks from JD intersect with known skills
    jd_chunks = {c.text.lower().strip() for c in nlp(jd_text).noun_chunks if 2 <= len(c.text) <= 40}
    jd_skills = sorted(set(jd_skills) | (jd_chunks & set(ALL_SKILLS)))

    # Embeddings (semantic)
    emb_resume = model.encode([resume_text], normalize_embeddings=True)[0]
    emb_job = model.encode([jd_text], normalize_embeddings=True)[0]
    sem = _cosine_sim(emb_resume, emb_job)  # ~0..1

    # Skills overlap
    jacc = _jaccard(resume_skills, jd_skills)  # 0..1

    final = (w_sem * sem) + (w_skills * jacc)
    score = round(final * 100, 1)

    info = {
        "semantic_similarity": round(sem, 3),
        "skills_jaccard": round(jacc, 3),
        "resume_skills_count": len(resume_skills),
        "job_skills_count": len(jd_skills),
        "matched_skills": sorted(set(resume_skills) & set(jd_skills)),
        "missing_skills": sorted(set(jd_skills) - set(resume_skills)),
        "extra_skills": sorted(set(resume_skills) - set(jd_skills)),
    }
    return score, info
