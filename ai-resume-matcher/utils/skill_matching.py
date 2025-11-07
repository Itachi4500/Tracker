from typing import List, Tuple, Dict
import numpy as np
import regex as re

def _embed(texts: List[str], model) -> np.ndarray:
    emb = model.encode(texts, normalize_embeddings=True)
    if isinstance(emb, list):
        emb = np.array(emb)
    return emb

def compute_semantic_similarity(resume_text: str, jd_text: str, model) -> float:
    """
    Cosine similarity between resume and JD embeddings (0..1).
    """
    if not resume_text.strip() or not jd_text.strip():
        return 0.0
    emb = _embed([resume_text, jd_text], model)
    v1, v2 = emb[0], emb[1]
    sim = float(np.clip(np.dot(v1, v2), 0.0, 1.0))
    return sim

def _normalize_tokens(words: List[str], min_len: int) -> List[str]:
    toks = []
    for w in words:
        w = w.strip().lower()
        w = re.sub(r"[^a-z0-9+#\.\- ]", "", w)
        w = re.sub(r"\s+", " ", w).strip()
        if len(w) >= min_len:
            toks.append(w)
    return toks

def extract_keywords_from_text(text: str, nlp=None, top_k: int = 50, min_len: int = 3) -> List[str]:
    """
    Simple keyword extractor:
      - keeps nouns, proper nouns, adjectives
      - lemmatizes and deduplicates
    """
    if nlp is None:
        from utils.model_loader import get_spacy_nlp
        nlp = get_spacy_nlp()

    doc = nlp(text)
    cands = []
    for tok in doc:
        if tok.is_stop or tok.is_punct or len(tok.text) < min_len:
            continue
        if tok.pos_ in {"NOUN", "PROPN", "ADJ"}:
            cands.append(tok.lemma_.lower())

    cands = _normalize_tokens(cands, min_len)
    # Keep most frequent up to top_k
    freq = {}
    for c in cands:
        freq[c] = freq.get(c, 0) + 1
    sorted_terms = [k for k, _ in sorted(freq.items(), key=lambda x: x[1], reverse=True)]
    return sorted_terms[:top_k]

def compute_overlap_score(resume_skills: List[str], jd_keywords: List[str]) -> float:
    """
    Jaccard-like overlap on sets (0..1).
    """
    rs = set([s.lower() for s in resume_skills])
    jk = set([k.lower() for k in jd_keywords])
    if not jk:
        return 0.0
    inter = len(rs & jk)
    return inter / len(jk)

def build_match_breakdown(sim_score: float, overlap_score: float, w_sem: float, w_kw: float) -> Tuple[float, Dict]:
    """
    Weighted final score on 0..100
    """
    final = (w_sem * sim_score + w_kw * overlap_score) * 100.0
    breakdown = {
        "semantic_similarity_pct": round(sim_score * 100.0, 2),
        "keyword_overlap_pct": round(overlap_score * 100.0, 2),
        "weights": {"semantic": w_sem, "keywords": w_kw},
        "formula": "final = (w_sem * semantic + w_kw * overlap) * 100",
        "final_score": round(final, 2),
    }
    return float(final), breakdown
