import os
import re
import webbrowser
from datetime import datetime
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS
import numpy as np

try:
    from groq import Groq
except ImportError:
    Groq = None

# Initialize Flask app to serve static files directly from the current directory
app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app)

MUTATION_LIBRARY = {
    "M184V": {"drug_classes": ["NRTI"], "drug": "Lamivudine", "severity": "High"},
    "K103N": {"drug_classes": ["NNRTI"], "drug": "Efavirenz", "severity": "Critical"},
    "L100I": {"drug_classes": ["NNRTI"], "drug": "Nevirapine", "severity": "Moderate"},
    "T215Y": {"drug_classes": ["NRTI"], "drug": "Zidovudine", "severity": "High"},
    "Y181C": {"drug_classes": ["NNRTI"], "drug": "Etravirine", "severity": "High"},
    "N155H": {"drug_classes": ["INSTI"], "drug": "Raltegravir", "severity": "High"},
    "G48V": {"drug_classes": ["INSTI"], "drug": "Dolutegravir", "severity": "Moderate"},
    "I54V": {"drug_classes": ["PI"], "drug": "Darunavir", "severity": "Moderate"},
}


def normalize_sequence(sequence: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", sequence).upper()


def parse_mutations(raw_text: str) -> list[str]:
    normalized = normalize_sequence(raw_text)
    matches = re.findall(r"[A-Z]\d+[A-Z]", normalized)
    if matches:
        return sorted(dict.fromkeys(matches))

    compact = re.sub(r"\s+", "", normalized)
    short_matches = [token for token in compact.split(",") if token]
    return [token for token in short_matches if token][:10]


def build_resistance_table(mutations: list[str], selected_drugs: list[str]) -> list[dict]:
    rows = []
    for mutation in mutations:
        entry = MUTATION_LIBRARY.get(mutation)
        if not entry:
            continue
        if any(cl in entry["drug_classes"] for cl in selected_drugs):
            rows.append(
                {
                    "Mutation": mutation,
                    "Drug": entry["drug"],
                    "Class": ", ".join(entry["drug_classes"]),
                    "Severity": entry["severity"],
                }
            )
    if not rows:
        rows.append(
            {
                "Mutation": "No matching callouts",
                "Drug": "Review manual annotation",
                "Class": "—",
                "Severity": "Low",
            }
        )
    return rows


def build_risk_profile(mutations: list[str], selected_drugs: list[str]) -> dict:
    table = build_resistance_table(mutations, selected_drugs)
    severity_order = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
    highest = max(table, key=lambda item: severity_order.get(item["Severity"], 0)) if table else {"Severity": "Low"}
    highest_severity = highest["Severity"]
    resistance_probability = min(92, 24 + len(table) * 13 + severity_order.get(highest_severity, 0) * 5)
    fitness_score = round(max(0.2, 0.9 - len(table) * 0.04), 2)
    return {
        "fitness_score": fitness_score,
        "resistance_probability": int(resistance_probability),
        "mutation_count": len(mutations),
        "summary": highest_severity,
        "table": table,
    }


def build_guidance(mutations: list[str], selected_drugs: list[str]) -> str:
    table = build_resistance_table(mutations, selected_drugs)
    if table[0]["Mutation"] == "No matching callouts":
        return "The current input does not match the built-in mutation catalogue. Add a known mutation code or upload a variant list for a more specific recommendation."

    severity_order = {"Low": 1, "Moderate": 2, "High": 3, "Critical": 4}
    dominant = max(table, key=lambda item: severity_order.get(item["Severity"], 0))

    if dominant["Severity"] in {"Critical", "High"}:
        return (
            f"Prioritize a regimen review around {dominant['Mutation']} because it is associated with "
            f"{dominant['Drug']} and a {dominant['Severity'].lower()} resistance signal."
        )
    return "The observed markers are moderate, so keep the active regimen under review and confirm the mutation list before changing therapy."


# --- PAGE ROUTES ---

# Serves the landing page on root navigation
@app.route("/")
def landing():
    return send_from_directory(".", "landing.html")


# Serves the main analysis application dashboard
@app.route("/app")
def main_app():
    return send_from_directory(".", "index.html")


# --- API ENDPOINTS ---

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    sequence_text = data.get("sequence", "")
    drug_classes = data.get("drug_classes", ["NNRTI", "NRTI"])

    mutations = parse_mutations(sequence_text)
    profile = build_risk_profile(mutations, drug_classes)
    guidance = build_guidance(mutations, drug_classes)

    heatmap_data = [[1.2, 0.8, 0.6, 0.4], [0.7, 1.0, 0.9, 0.5], [0.6, 0.8, 0.7, 0.4]]

    return jsonify({
        "mutations": mutations,
        "profile": profile,
        "guidance": guidance,
        "heatmap": {
            "z": heatmap_data,
            "x": ["NRTI", "NNRTI", "PI", "INSTI"],
            "y": ["Resistance", "Fitness", "Sensitivity"]
        }
    })


@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.json or {}
    prompt = data.get("prompt", "")
    assistant_log = data.get("assistant_log", [])

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        return jsonify({
            "response": "The AI assistant is running in offline mode. Use the structured resistance summary below for a clinician-facing overview."
        })

    try:
        client = Groq(api_key=api_key)
        messages = [
            {
                "role": "system",
                "content": "You are a concise HIV resistance assistant for a clinical dashboard. Do not make treatment decisions.",
            }
        ] + assistant_log + [{"role": "user", "content": prompt}]

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.2,
            max_tokens=400,
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"response": f"Error connecting to AI service: {str(e)}"})


import threading

if __name__ == "__main__":
    PORT = 5000
    url = f"http://127.0.0.1:{PORT}/"

    # Only launch the browser in the main Flask process (prevents debug reloader double-trigger)
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.2, lambda: webbrowser.open_new(url)).start()

    app.run(host="0.0.0.0", port=PORT, debug=True)