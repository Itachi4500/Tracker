import re
from typing import Dict

def ats_checks(text: str) -> Dict[str, bool]:
    return {
        "Has_Contact_Info": bool(re.search(r"\b(\+?\d[\d\-\s]{7,}\d|[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,})\b", text, re.I)),
        "Has_Experience_Section": bool(re.search(r"\b(experience|work history|employment)\b", text, re.I)),
        "Has_Education_Section": bool(re.search(r"\b(education|degree|bachelor|master|ph\.?d)\b", text, re.I)),
        "Has_Skills_Section": bool(re.search(r"\bskills\b", text, re.I)),
        "Uses_Bullets": ("•" in text) or bool(re.search(r"(\n- |\n\* )", text)),
    }
