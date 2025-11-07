from typing import Dict, List, Optional
import regex as re

CORE_SECTIONS = [
    "summary", "objective", "experience", "work experience", "professional experience",
    "education", "skills", "projects", "certifications"
]

BAD_CHARS = set(["□", "■", "", "", "★", "☆"])  # weird bullets/icons often break OCR/ATS

def _has_tables_or_columns(text: str) -> bool:
    # crude heuristic: presence of many '|' or table-like rows
    bars = text.count("|")
    tabs = text.count("\t")
    return (bars >= 5) or (tabs >= 20)

def _bullet_density_warning(text: str) -> Optional[str]:
    bullets = len(re.findall(r"[•\-▪▶◆●■]", text))
    lines = len(text.splitlines()) or 1
    dens = bullets / lines
    if dens > 0.5:
        return "Bullets appear in most lines; simplify to short sentences for better parsing."
    return None

def _length_check(page_count: Optional[int], text: str) -> Optional[str]:
    if page_count and page_count > 2:
        return f"Resume is {page_count} pages; try to keep it within 1–2 pages."
    # fallback by characters
    chars = len(text)
    if page_count is None and chars > 14000:
        return "Resume is very long; aim for concise, ATS-friendly content."
    return None

def _bad_chars_present(text: str) -> Optional[str]:
    if any(c in text for c in BAD_CHARS):
        return "Found special decorative bullets/characters; replace with standard ASCII bullets (-) or plain text."
    return None

def _sections_present(text: str) -> List[str]:
    lowers = text.lower()
    present = []
    for s in CORE_SECTIONS:
        if s in lowers:
            present.append(s)
    return present

def _keyword_coverage(extracted_skills, jd_keywords) -> float:
    if not jd_keywords:
        return 1.0
    rs = set([s.lower() for s in extracted_skills])
    jk = set([k.lower() for k in jd_keywords])
    return len(rs & jk) / len(jk)

def ats_check(
    text: str,
    filetype: str,
    page_count: Optional[int],
    sections: Dict[str, str],
    extracted_skills: List[str],
    jd_keywords: List[str],
) -> Dict:
    good = []
    warnings = []

    if filetype.lower() not in {"pdf", "docx", "txt"}:
        warnings.append("Use PDF, DOCX, or TXT for best ATS compatibility.")

    if _has_tables_or_columns(text):
        warnings.append("Tables/columns detected; prefer single-column layouts for ATS.")

    bdw = _bullet_density_warning(text)
    if bdw:
        warnings.append(bdw)

    lw = _length_check(page_count, text)
    if lw:
        warnings.append(lw)

    bc = _bad_chars_present(text)
    if bc:
        warnings.append(bc)

    present_secs = _sections_present(text)
    for req in ["experience", "education", "skills"]:
        if not any(req in s for s in present_secs):
            warnings.append(f"Missing or unclear **{req.title()}** section header.")

    if "skills" in sections and len(sections["skills"].split()) >= 10:
        good.append("Skills section present with substantial content.")

    cov = _keyword_coverage(extracted_skills, jd_keywords)
    if cov >= 0.6:
        good.append("Good coverage of JD keywords in resume.")
    elif cov >= 0.3:
        warnings.append("Partial coverage of JD keywords. Consider adding missing relevant skills.")
    else:
        warnings.append("Low coverage of JD keywords. Tailor your skills and achievements to the JD.")

    # Simple heuristic ATS score (0..1)
    score = 1.0
    if warnings:
        score -= min(0.6, 0.08 * len(warnings))
    score = max(0.05, min(1.0, score))

    return {
        "ats_score": score,
        "warnings": warnings,
        "good": good,
        "page_count": page_count,
        "filetype": filetype,
    }
