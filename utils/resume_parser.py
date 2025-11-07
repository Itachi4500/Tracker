import io, re
from pathlib import Path
from pdfminer.high_level import extract_text as pdf_extract_text
from docx import Document
from unidecode import unidecode

ALLOWED_EXTS = {".pdf", ".docx", ".doc", ".txt"}
MAX_BYTES = 8 * 1024 * 1024  # 8 MB safety limit

def _clean_text(text: str) -> str:
    text = unidecode(text or "")
    text = re.sub(r"\s+", " ", text).strip()
    return text

def _read_pdf(buf: io.BytesIO) -> str:
    buf.seek(0)
    return pdf_extract_text(buf) or ""

def _read_docx(buf: io.BytesIO) -> str:
    buf.seek(0)
    doc = Document(buf)
    return "\n".join(p.text for p in doc.paragraphs)

def _read_txt(buf: io.BytesIO) -> str:
    buf.seek(0)
    return buf.read().decode(errors="ignore")

def extract_text_from_file(file_storage) -> str:
    filename = file_storage.filename or "file"
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTS:
        raise ValueError(f"Unsupported file type: {suffix}")

    data = file_storage.read()
    if len(data) > MAX_BYTES:
        raise ValueError("File too large. Please upload files under 8 MB.")
    buf = io.BytesIO(data)

    if suffix == ".pdf":
        text = _read_pdf(buf)
    elif suffix in {".docx", ".doc"}:
        text = _read_docx(buf)
    else:
        text = _read_txt(buf)

    return _clean_text(text)
