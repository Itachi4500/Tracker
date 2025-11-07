from functools import lru_cache
import importlib
from typing import Any

def _ensure_spacy_model():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:
        nlp = spacy.blank("en")
        # Only add components if available
        if "tagger" not in nlp.pipe_names:
            nlp.add_pipe("tagger", config={"model": None}, last=True)
        return nlp
        
@lru_cache(maxsize=1)
def get_spacy_nlp() -> Any:
    """
    Returns a cached spaCy NLP pipeline.
    Prefers en_core_web_sm; falls back to blank('en') if not present.
    """
    return _ensure_spacy_model()

@lru_cache(maxsize=2)
def get_sentence_model(model_name: str):
    """
    Returns a cached SentenceTransformer model.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except Exception as e:
        raise RuntimeError(
            "sentence-transformers not installed. Ensure it is in requirements.txt."
        ) from e
    return SentenceTransformer(model_name)

