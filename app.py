
import os
import time

from flask import Flask, render_template, request, jsonify
from google import genai
from google.genai import types

app = Flask(__name__)

# Gemini configuration
API_KEY = os.getenv("GEMINI_API_KEY")
MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

client = genai.Client(api_key=API_KEY) if API_KEY else None


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/ask", methods=["POST"])
def ask_bortan():
    data = request.get_json(silent=True) or {}
    question = str(data.get("question", "")).strip()

    if not question:
        return jsonify({
            "error": "Please enter a question."
        }), 400

    if client is None:
        return jsonify({
            "error": (
                "GEMINI_API_KEY is not set. "
                "Set your Gemini API key in PowerShell."
            )
        }), 500

    max_retries = 3

    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=question,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are BORTAN AI, a helpful and intelligent AI assistant. "
                        "BORTAN stands for Brain-Optimized Raged "
                        "Terabyte Algorithmic-analytic Neutral-node. "
                        "Answer clearly, accurately, and naturally. "
                        "Explain difficult topics in simple language. "
                        "For questions about religion, philosophy, or God, "
                        "present different viewpoints respectfully and neutrally. "
                        "If you are unsure, clearly say so. "
                        "Do not claim to have performed actions you cannot perform."
                    ),
                    temperature=0.7,
                    max_output_tokens=2048
                )
            )

            answer = response.text

            if not answer:
                answer = "I could not generate an answer."

            return jsonify({
                "answer": answer
            }), 200

        except Exception as error:
            error_text = str(error)

            print(
                f"BORTAN ERROR (attempt {attempt + 1}/{max_retries}):",
                repr(error)
            )

            # Retry temporary Gemini server errors
            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
                or "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    time.sleep(wait_time)
                    continue

                return jsonify({
                    "error": (
                        "Gemini is temporarily busy or rate-limited. "
                        "Please wait a moment and try again."
                    )
                }), 503

            # Handle invalid model errors
            if "404" in error_text or "NOT_FOUND" in error_text:
                return jsonify({
                    "error": (
                        f"The Gemini model '{MODEL}' is unavailable. "
                        "Check the model name in app.py."
                    )
                }), 404

            # Handle authentication errors
            if (
                "401" in error_text
                or "403" in error_text
                or "PERMISSION_DENIED" in error_text
            ):
                return jsonify({
                    "error": (
                        "Gemini API authentication failed. "
                        "Check your API key and permissions."
                    )
                }), 403

            # General error
            return jsonify({
                "error": (
                    "BORTAN could not process the request. "
                    "Please check the Flask terminal."
                )
            }), 500

    return jsonify({
        "error": "BORTAN could not complete the request."
    }), 500


if __name__ == "__main__":
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )
