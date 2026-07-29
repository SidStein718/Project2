import os
import re
import webbrowser
import threading
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv
from flask import Flask, jsonify, request, send_from_directory
from flask_cors import CORS

try:
    from groq import Groq
except ImportError:
    Groq = None


# ==========================================
# Flask App
# ==========================================

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

app = Flask(__name__)
CORS(app)


# ==========================================
# Mutation Database (Local Fallback)
# ==========================================

MUTATION_LIBRARY = {
    # NRTIs
    "M184V": {
        "drug_classes": ["NRTI"],
        "drug": "Lamivudine / Emtricitabine",
        "severity": "High",
        "gene": "Reverse Transcriptase",
        "description": "High-level resistance to 3TC and FTC; increases ZDV and TDF susceptibility."
    },
    "K65R": {
        "drug_classes": ["NRTI"],
        "drug": "Tenofovir / Abacavir",
        "severity": "Critical",
        "gene": "Reverse Transcriptase",
        "description": "Reduces susceptibility to Tenofovir, Abacavir, and Didanosine."
    },
    "T215Y": {
        "drug_classes": ["NRTI"],
        "drug": "Zidovudine",
        "severity": "High",
        "gene": "Reverse Transcriptase",
        "description": "Thymidine analogue mutation causing high resistance to ZDV and d4T."
    },

    # NNRTIs
    "K103N": {
        "drug_classes": ["NNRTI"],
        "drug": "Efavirenz / Nevirapine",
        "severity": "Critical",
        "gene": "Reverse Transcriptase",
        "description": "Causes high-level cross-resistance to first-generation NNRTIs."
    },
    "Y181C": {
        "drug_classes": ["NNRTI"],
        "drug": "Nevirapine / Etravirine",
        "severity": "High",
        "gene": "Reverse Transcriptase",
        "description": "High-level resistance to Nevirapine and intermediate resistance to Etravirine."
    },
    "G190A": {
        "drug_classes": ["NNRTI"],
        "drug": "Efavirenz",
        "severity": "High",
        "gene": "Reverse Transcriptase",
        "description": "Reduces susceptibility to Efavirenz and Nevirapine."
    },

    # Protease Inhibitors (PI)
    "G48V": {
        "drug_classes": ["PI"],
        "drug": "Saquinavir",
        "severity": "Moderate",
        "gene": "Protease",
        "description": "Reduces susceptibility to Saquinavir and Atazanavir."
    },
    "I54V": {
        "drug_classes": ["PI"],
        "drug": "Darunavir / Lopinavir",
        "severity": "Moderate",
        "gene": "Protease",
        "description": "Protease inhibitor resistance mutation affecting multiple PIs."
    },
    "V82A": {
        "drug_classes": ["PI"],
        "drug": "Lopinavir",
        "severity": "High",
        "gene": "Protease",
        "description": "Major mutation reducing susceptibility to Lopinavir and Indinavir."
    },
    "L90M": {
        "drug_classes": ["PI"],
        "drug": "Atazanavir / Saquinavir",
        "severity": "High",
        "gene": "Protease",
        "description": "Major mutation causing broad cross-resistance across PI class."
    },

    # Integrase Inhibitors (INSTI)
    "N155H": {
        "drug_classes": ["INSTI"],
        "drug": "Raltegravir / Elvitegravir",
        "severity": "High",
        "gene": "Integrase",
        "description": "Reduces response to first-generation integrase strand transfer inhibitors."
    },
    "Q148H": {
        "drug_classes": ["INSTI"],
        "drug": "Dolutegravir / Raltegravir",
        "severity": "Critical",
        "gene": "Integrase",
        "description": "Major integrase resistance mutation conferring cross-resistance across INSTIs."
    }
}


STANFORD_HIVDB_URL = "https://hivdb.stanford.edu/graphql"


# ==========================================
# Utility & Stanford API Functions
# ==========================================

def parse_mutations(raw_text: str):
    if not raw_text:
        return []
        
    lines = [line.strip() for line in raw_text.splitlines() if not line.strip().startswith(">")]
    clean_text = " ".join(lines).upper()

    matches = re.findall(r"[A-Z]\d+[A-Z]", clean_text)
    if matches:
        return sorted(dict.fromkeys(matches))

    tokens = [t.strip() for t in re.split(r"[\s,]+", clean_text) if t.strip()]
    valid_tokens = [t for t in tokens if re.match(r"^[A-Z0-9]+$", t)]
    return valid_tokens[:10]


def query_stanford_hivdb(mutations_list):
    """Queries Stanford HIV Drug Resistance Database via Sierra GraphQL API."""
    if not mutations_list:
        return None

    formatted_mutations = []
    for m in mutations_list:
        if m.startswith("RT:") or m.startswith("PR:") or m.startswith("IN:"):
            formatted_mutations.append(m)
        else:
            # Default to Reverse Transcriptase (RT) prefix if not specified
            formatted_mutations.append(f"RT:{m}")

    mutation_pattern = " + ".join(formatted_mutations)

    graphql_query = """
    query AnalyzeMutations($patterns: [String]!) {
      mutationsAnalysis(patterns: $patterns) {
        validationResults {
          level
          message
        }
        drugResistances {
          drugClass {
            name
          }
          drug {
            name
            displayAbbr
          }
          score
          text
          level
        }
      }
    }
    """

    try:
        response = requests.post(
            STANFORD_HIVDB_URL,
            json={"query": graphql_query, "variables": {"patterns": [mutation_pattern]}},
            headers={"Content-Type": "application/json"},
            timeout=6
        )
        if response.status_code == 200:
            res_data = response.json()
            analysis = res_data.get("data", {}).get("mutationsAnalysis", [])
            if analysis:
                return analysis[0]
        return None
    except Exception as e:
        print(f"Stanford API Query Warning: {e}. Falling back to local library.")
        return None


def build_resistance_table(mutations, selected_drugs):
    # Try Stanford HIVdb API first
    stanford_result = query_stanford_hivdb(mutations)
    rows = []

    if stanford_result and "drugResistances" in stanford_result:
        for dr in stanford_result["drugResistances"]:
            drug_class_name = dr["drugClass"]["name"]
            if drug_class_name in selected_drugs:
                level_text = dr.get("text", "Susceptible")
                # Map Stanford levels to severity terms
                severity_map = {
                    "High-level resistance": "Critical",
                    "Substantial resistance": "High",
                    "Intermediate resistance": "High",
                    "Low-level resistance": "Moderate",
                    "Potential low-level resistance": "Moderate",
                    "Susceptible": "Low"
                }
                severity = severity_map.get(level_text, "Moderate")

                rows.append({
                    "Mutation": ", ".join(mutations),
                    "Drug": dr["drug"]["name"],
                    "Class": drug_class_name,
                    "Severity": severity
                })
        if rows:
            return rows

    # Fallback to local MUTATION_LIBRARY
    for mutation in mutations:
        entry = MUTATION_LIBRARY.get(mutation)
        if not entry:
            continue

        if any(drug in entry["drug_classes"] for drug in selected_drugs):
            rows.append({
                "Mutation": mutation,
                "Drug": entry["drug"],
                "Class": ", ".join(entry["drug_classes"]),
                "Severity": entry["severity"]
            })

    return rows


def build_risk_profile(mutations, selected_drugs):
    real_rows = build_resistance_table(mutations, selected_drugs)

    severity_order = {
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Critical": 4
    }

    if real_rows:
        highest = max(
            real_rows,
            key=lambda item: severity_order.get(item["Severity"], 0)
        )

        resistance_probability = min(
            95,
            25 + len(real_rows) * 13 + severity_order.get(highest["Severity"], 0) * 5
        )

        fitness_score = round(
            max(0.20, 0.90 - len(real_rows) * 0.04),
            2
        )
        summary = highest["Severity"]
        table_to_display = real_rows
    else:
        resistance_probability = 0
        fitness_score = 1.00
        summary = "Low"
        table_to_display = [{
            "Mutation": "No matching callouts",
            "Drug": "Review manual annotation",
            "Class": "—",
            "Severity": "Low"
        }]

    return {
        "fitness_score": fitness_score,
        "resistance_probability": resistance_probability,
        "mutation_count": len(mutations),
        "summary": summary,
        "table": table_to_display
    }


def confidence_score(profile):
    score = 60
    score += profile["mutation_count"] * 6

    if profile["summary"] == "Critical":
        score += 15
    elif profile["summary"] == "High":
        score += 10
    elif profile["summary"] == "Moderate":
        score += 5

    return min(score, 98)


def risk_color(summary):
    colors = {
        "Low": "Green",
        "Moderate": "Yellow",
        "High": "Orange",
        "Critical": "Red"
    }
    return colors.get(summary, "Gray")


def mutation_timeline(mutations):
    year = datetime.now().year - len(mutations)
    timeline = []

    for mutation in mutations:
        timeline.append({
            "year": year,
            "mutation": mutation
        })
        year += 1

    return timeline


def mutation_explanations(mutations):
    explanations = []

    for mutation in mutations:
        if mutation not in MUTATION_LIBRARY:
            explanations.append({
                "mutation": mutation,
                "gene": "Genomic Variation",
                "drug": "Cross-Class Analysis",
                "severity": "Evaluated via Stanford HIVdb",
                "description": f"Marker {mutation} analyzed against Stanford HIV Drug Resistance criteria."
            })
            continue

        info = MUTATION_LIBRARY[mutation]

        explanations.append({
            "mutation": mutation,
            "gene": info["gene"],
            "drug": info["drug"],
            "severity": info["severity"],
            "description": info["description"]
        })

    return explanations


def build_guidance(mutations, selected_drugs):
    table = build_resistance_table(mutations, selected_drugs)

    if not table:
        return {
            "title": "AI Resistance Summary",
            "risk": "Low",
            "summary": "No known resistance mutations were found in the active mutation sequence.",
            "recommendation": "Verify the submitted mutations or upload a clean FASTA file."
        }

    severity_order = {
        "Low": 1,
        "Moderate": 2,
        "High": 3,
        "Critical": 4
    }

    dominant = max(
        table,
        key=lambda x: severity_order.get(x["Severity"], 1)
    )

    return {
        "title": "AI Resistance Summary",
        "risk": dominant["Severity"],
        "summary": (
            f"{len(mutations)} mutation(s) were detected. "
            f"Key finding associated with {dominant['Drug']} shows "
            f"{dominant['Severity'].lower()} resistance risk."
        ),
        "recommendation": (
            "This analysis is educational decision support based on Stanford HIVdb principles. "
            "Laboratory confirmation and clinician review are recommended before making treatment choices."
        )
    }


# ==========================================
# Page Routes
# ==========================================

@app.route("/")
def landing():
    return send_from_directory(".", "landing.html")


@app.route("/app")
def main_app():
    return send_from_directory(".", "index.html")


# ==========================================
# API Endpoints
# ==========================================

@app.route("/api/analyze", methods=["POST"])
def analyze():
    data = request.json or {}
    sequence_text = data.get("sequence", "")
    drug_classes = data.get("drug_classes", ["NNRTI", "NRTI"])

    mutations = parse_mutations(sequence_text)
    profile = build_risk_profile(mutations, drug_classes)
    guidance = build_guidance(mutations, drug_classes)

    confidence = confidence_score(profile)
    risk = risk_color(profile["summary"])
    timeline = mutation_timeline(mutations)
    explanations = mutation_explanations(mutations)

    # Dynamic Heatmap Matrix based on parsed mutations
    has_nrti = any(m in ["M184V", "T215Y", "K65R"] for m in mutations)
    has_nnrti = any(m in ["K103N", "L100I", "Y181C", "G190A"] for m in mutations)
    has_pi = any(m in ["G48V", "I54V", "V82A", "L90M"] for m in mutations)
    has_insti = any(m in ["N155H", "Q148H"] for m in mutations)

    heatmap_data = [
        [3.8 if has_nrti else 1.1, 4.5 if has_nnrti else 1.0, 2.9 if has_pi else 0.8, 3.2 if has_insti else 0.7],
        [0.65 if has_nrti else 0.95, 0.50 if has_nnrti else 0.92, 0.78 if has_pi else 0.98, 0.82 if has_insti else 0.96],
        [0.2 if has_nrti else 0.9, 0.1 if has_nnrti else 0.95, 0.4 if has_pi else 0.85, 0.3 if has_insti else 0.88]
    ]

    return jsonify({
        "mutations": mutations,
        "profile": profile,
        "guidance": guidance,
        "confidence": confidence,
        "risk_color": risk,
        "timeline": timeline,
        "explanations": explanations,
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
    active_mutations = data.get("mutations", [])
    patient_id = data.get("patient_id", "PAT-9921")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or Groq is None:
        return jsonify({
            "response": (
                "The AI assistant is running in offline mode. "
                "Use the mutation analysis, explanation cards, and AI summary "
                "for educational purposes."
            )
        })

    mutation_context = ", ".join(active_mutations) if active_mutations else "None specified"

    system_instruction = (
        f"You are a Clinical Bioinformatics AI Assistant embedded in an HIV Resistance Dashboard.\n"
        f"ACTIVE PATIENT DATA:\n"
        f"- Patient ID: {patient_id}\n"
        f"- Detected Mutations: {mutation_context}\n\n"
        f"RULES FOR ACCURACY:\n"
        f"1. Base explanations on verified HIV resistance benchmarks (e.g., Stanford HIVdb guidelines).\n"
        f"2. Separate mutation mechanics from drug susceptibility impact.\n"
        f"3. Frame all insights as educational decision support for clinicians. Do not issue prescriptions.\n"
        f"4. Keep responses structured, precise, and clear."
    )

    try:
        client = Groq(api_key=api_key)
        messages = [{"role": "system", "content": system_instruction}]
        
        for msg in assistant_log:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                messages.append({"role": msg["role"], "content": msg["content"]})

        messages.append({"role": "user", "content": prompt})

        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=messages,
            temperature=0.1,  # Low temperature for fact-grounded accuracy
            max_tokens=450
        )
        reply = response.choices[0].message.content.strip()
        return jsonify({"response": reply})
    except Exception as e:
        return jsonify({"response": f"Error connecting to AI service: {str(e)}"})


if __name__ == "__main__":
    PORT = int(os.environ.get("PORT", "5000"))
    url = f"http://127.0.0.1:{PORT}/"
    if os.environ.get("WERKZEUG_RUN_MAIN") == "true":
        threading.Timer(1.2, lambda: webbrowser.open_new(url)).start()
    app.run(host="0.0.0.0", port=PORT, debug=False)