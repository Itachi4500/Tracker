import io
import re
from dataclasses import dataclass
from typing import Dict, Optional

import pdfplumber
import docx2txt
from unidecode import unidecode

SECTION_HEADERS = [
    "summary", "objective", "experience", "work experience", "professional experience",
    "education", "skills", "technical skills", "projects", "certifications",
    "achievements", "awards", "publications", "interests",
]

EMAIL_RE = re.compile(r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}", re.I)
PHONE_RE = re.compile(r"(?:(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?)[-.\s]?)?\d{3,4}[-.\s]?\d{4}")
URL_RE = re.compile(r"(https?://[^\s]+)", re.I)
LINKEDIN_RE = re.compile(r"(linkedin\.com/in/[A-Za-z0-9\-_]+)", re.I)
GITHUB_RE = re.compile(r"(github\.com/[A-Za-z0-9\-_]+)", re.I)

@dataclass
class ParsedResume:
    text: str
    filetype: str
    page_count: Optional[int] = None

def _clean_text(text: str) -> str:
    text = unidecode(text or "")
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\r", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()

def _extract_text_from_pdf(file) -> ParsedResume:
    # file is a BytesIO-like object from st.file_uploader
    file_bytes = file.read()
    with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
        pages_text = []
        for page in pdf.pages:
            try:
                pages_text.append(page.extract_text() or "")
            except Exception:
                pages_text.append("")
        text = "\n".join(pages_text)
        return ParsedResume(text=_clean_text(text), filetype="pdf", page_count=len(pdf.pages))

def _extract_text_from_docx(file) -> ParsedResume:
    file_bytes = file.read()
    # docx2txt expects a path or buffer; we can pass bytes via temp buffer
    text = docx2txt.process(io.BytesIO(file_bytes))
    return ParsedResume(text=_clean_text(text), filetype="docx", page_count=None)

def _extract_text_from_txt(file) -> ParsedResume:
    text = file.read().decode("utf-8", errors="ignore")
    return ParsedResume(text=_clean_text(text), filetype="txt", page_count=None)

def parse_resume(file) -> ParsedResume:
    name = getattr(file, "name", "resume")
    lower = name.lower()
    file.seek(0)
    if lower.endswith(".pdf"):
        return _extract_text_from_pdf(file)
    elif lower.endswith(".docx"):
        return _extract_text_from_docx(file)
    elif lower.endswith(".txt"):
        return _extract_text_from_txt(file)
    else:
        # attempt pdf first as many resumes are pdf
        try:
            file.seek(0)
            return _extract_text_from_pdf(file)
        except Exception:
            file.seek(0)
            return _extract_text_from_txt(file)

def extract_sections(text: str) -> Dict[str, str]:
    """
    Split resume text by common section headers. Heuristic but effective.
    """
    content = {}
    t = text.replace("\r", "\n")
    # Build a regex that captures sections
    pattern = r"(?mi)^\s*(%s)\s*[:\n]" % "|".join([re.escape(h) for h in SECTION_HEADERS])
    splits = list(re.finditer(pattern, t))
    if not splits:
        return {"FULL_TEXT": t}

    for i, m in enumerate(splits):
        start = m.end()
        end = splits[i + 1].start() if (i + 1) < len(splits) else len(t)
        header = m.group(1).strip().title()
        chunk = t[start:end].strip()
        content[header] = chunk
    return content

def extract_contact_info(text: str) -> Dict[str, str]:
    emails = EMAIL_RE.findall(text)
    phones = PHONE_RE.findall(text)
    urls = URL_RE.findall(text)
    linkedin = LINKEDIN_RE.findall(text)
    github = GITHUB_RE.findall(text)

    return {
        "email": emails[0] if emails else "",
        "phone": phones[0] if phones else "",
        "linkedin": linkedin[0] if linkedin else "",
        "github": github[0] if github else "",
        "website": (urls[0] if urls else ""),
    }

def extract_skills_from_text(text: str, nlp=None, min_len: int = 3):
    """
    Lightweight skill extractor:
      - takes noun chunks and entities
      - filters stopwords/punctuation
      - lowercases and deduplicates
    """
    if nlp is None:
        from utils.model_loader import get_spacy_nlp
        nlp = get_spacy_nlp()

    doc = nlp(text)
    skills = set()

    # noun chunks
    for chunk in doc.noun_chunks:
        s = chunk.text.strip().lower()
        if len(s) >= min_len and s.isascii():
            skills.add(s)

    # entities (ORG, PRODUCT, etc.)
    for ent in doc.ents:
        s = ent.text.strip().lower()
        if len(s) >= min_len and s.isascii():
            skills.add(s)

    # Frequent separators list-like "Python, SQL, Pandas"
    for raw in re.split(r"[\n,;•|/]+", text):
        s = raw.strip().lower()
        if len(s) >= min_len and s.isascii() and " " not in s:
            # single-token skills (e.g., python)
            if re.match(r"^[a-z0-9+\-\.#]+$", s):
                skills.add(s)

    # Normalize whitespace
    skills = {re.sub(r"\s+", " ", s) for s in skills}
    return list(skills)
