from functools import lru_cache
from sentence_transformers import SentenceTransformer
import spacy

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"

@lru_cache(maxsize=1)
def load_models():
    # Load spaCy (assumes model installed via requirements.txt)
    nlp = spacy.load(SPACY_MODEL)
    # Load sentence-transformers model
    model = SentenceTransformer(MODEL_NAME)
    return nlp, model
