from flask import Flask, render_template, request, jsonify
from utils.model_loader import load_models
from utils.resume_parser import extract_text_from_file
from utils.skill_matching import get_match_score
from utils.ats_checker import ats_checks

app = Flask(__name__)
nlp, model = load_models()  # cached, loads once

CLASSIFY_RULES = [
    (90, "✅ Approved"),
    (70, "⚠️ Improvements Needed"),
    (0,  "❌ Not Approved"),
]

def classify(score: float) -> str:
    for th, label in CLASSIFY_RULES:
        if score >= th:
            return label
    return "❌ Not Approved"

@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    if "resume" not in request.files or "jd" not in request.files:
        return jsonify({"error": "Both 'resume' and 'jd' files are required."}), 400

    resume_file = request.files["resume"]
    jd_file = request.files["jd"]

    # Parse files to text
    resume_text = extract_text_from_file(resume_file)
    jd_text = extract_text_from_file(jd_file)

    # Score + explanation
    score, info = get_match_score(resume_text, jd_text, nlp, model)

    # ATS checks on resume
    ats = ats_checks(resume_text)

    return jsonify({
        "score": score,
        "status": classify(score),
        "skills": info,
        "ats": ats,
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=True)
