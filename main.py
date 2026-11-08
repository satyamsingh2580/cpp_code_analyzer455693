import os
import json
import logging

import requests as http_requests
from dotenv import load_dotenv
from flask import Flask, request, jsonify, send_file

# ---------------------------------------------------------------------------
# Environment & Configuration
# ---------------------------------------------------------------------------
load_dotenv()

API_KEY = os.getenv("LLM_API_KEY")
if not API_KEY:
    raise RuntimeError(
        "LLM_API_KEY is not set. Please add it to the .env file."
    )

GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# ---------------------------------------------------------------------------
# API Endpoints
# ---------------------------------------------------------------------------
# Primary: Google Gemini REST API (gemini-2.5-flash)
PRIMARY_MODEL = "gemini-2.5-flash"
GEMINI_URL = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{PRIMARY_MODEL}:generateContent?key={API_KEY}"
)

# Secondary: GitHub Models API (GPT-4o)
GITHUB_MODELS_URL = "https://models.inference.ai.azure.com/chat/completions"
FALLBACK_MODEL = "gpt-4o"

# Tertiary: Groq API (llama-3.1-8b-instant)
GROQ_URL = "https://api.groq.com/openai/v1/chat/completions"
GROQ_MODEL = "llama-3.1-8b-instant"

# Quaternary: OpenRouter API (meta-llama/llama-3.1-8b-instruct:free)
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
OPENROUTER_MODEL = "meta-llama/llama-3.1-8b-instruct:free"

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMPLATES_INDEX = os.path.join(BASE_DIR, "templates", "index.html")
INDEX_HTML = TEMPLATES_INDEX if os.path.exists(TEMPLATES_INDEX) else os.path.join(BASE_DIR, "index.html")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Flask App
# ---------------------------------------------------------------------------
app = Flask(__name__)

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------
SYSTEM_PROMPT = """\
You are a rigorous LeetCode judge and algorithm optimization expert.

The user will provide:
1. A problem description.
2. Their C++ solution code.

Your task:
- Analyze the problem description and classify its standard LeetCode difficulty as "easy", "medium", or "hard".
- Analyse the code for correctness and efficiency.
- Identify every inefficiency.
- Provide the optimal C++ solution.

CRITICAL: All generated C++ code must strictly adhere to the C++17 standard. You are explicitly forbidden from using C++20 or C++23 features (such as <ranges>, concepts, std::format, or modules). The output must compile flawlessly on LeetCode's C++17 environment.

CRITICAL FORMATTING RULE:
You MUST respond with ONLY a single, valid JSON object with these exact keys:

{
  "difficulty": "easy" | "medium" | "hard",
  "time_complexity": "<Big-O time complexity of the user's code>",
  "space_complexity": "<Big-O space complexity of the user's code>",
  "inefficiencies_found": ["<list of identified inefficiencies>"],
  "optimal_cpp_code": "<complete optimised C++ solution as a single string>",
  "explanation": "<detailed explanation of improvements made>"
}
"""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _extract_gemini_result(response_json: dict) -> dict:
    """Extract and validate the structured JSON result from a Gemini response."""
    parts = (
        response_json.get("candidates", [{}])[0]
        .get("content", {})
        .get("parts", [])
    )
    raw_text = "".join(
        p.get("text", "") for p in parts if p.get("text") and not p.get("thought")
    ).strip()

    if not raw_text:
        raise ValueError("Gemini response contained no text.")

    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    result = json.loads(raw_text)

    if isinstance(result.get("inefficiencies_found"), str):
        result["inefficiencies_found"] = [result["inefficiencies_found"]]
    elif not isinstance(result.get("inefficiencies_found"), list):
        result["inefficiencies_found"] = []

    # Unescape literal newlines in optimal_cpp_code if double-escaped
    if isinstance(result.get("optimal_cpp_code"), str):
        code = result["optimal_cpp_code"]
        if "\\n" in code and "\n" not in code:
            code = code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")
        result["optimal_cpp_code"] = code

    if "difficulty" not in result or result["difficulty"] not in ("easy", "medium", "hard"):
        diff = str(result.get("difficulty", "medium")).lower().strip()
        result["difficulty"] = diff if diff in ("easy", "medium", "hard") else "medium"

    required_keys = {
        "time_complexity",
        "space_complexity",
        "inefficiencies_found",
        "optimal_cpp_code",
        "explanation",
    }
    missing = required_keys - result.keys()
    if missing:
        raise ValueError(f"Gemini response missing keys: {missing}")

    return result


def _parse_chat_completion_content(content: str) -> dict:
    """Parse OpenAI/Groq/OpenRouter chat completion content into the required frontend JSON dictionary."""
    raw_text = content.strip()
    if raw_text.startswith("```"):
        lines = raw_text.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        raw_text = "\n".join(lines).strip()

    try:
        result = json.loads(raw_text)
        if isinstance(result, dict):
            if isinstance(result.get("inefficiencies_found"), str):
                result["inefficiencies_found"] = [result["inefficiencies_found"]]
            elif not isinstance(result.get("inefficiencies_found"), list):
                result["inefficiencies_found"] = []

            # Unescape literal newlines in optimal_cpp_code if double-escaped
            if isinstance(result.get("optimal_cpp_code"), str):
                code = result["optimal_cpp_code"]
                if "\\n" in code and "\n" not in code:
                    code = code.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\t", "    ")
                result["optimal_cpp_code"] = code

            if "difficulty" not in result or result["difficulty"] not in ("easy", "medium", "hard"):
                diff = str(result.get("difficulty", "medium")).lower().strip()
                result["difficulty"] = diff if diff in ("easy", "medium", "hard") else "medium"

            required_keys = {
                "time_complexity",
                "space_complexity",
                "inefficiencies_found",
                "optimal_cpp_code",
                "explanation",
            }
            if required_keys.issubset(result.keys()):
                return result

            return {
                "difficulty": result.get("difficulty", "medium"),
                "time_complexity": result.get("time_complexity", "O(N)"),
                "space_complexity": result.get("space_complexity", "O(1)"),
                "inefficiencies_found": result.get("inefficiencies_found", []),
                "optimal_cpp_code": result.get("optimal_cpp_code", raw_text),
                "explanation": result.get(
                    "explanation", "Optimized solution generated via AI fallback."
                ),
            }
    except json.JSONDecodeError:
        pass

    return {
        "difficulty": "medium",
        "time_complexity": "Optimized",
        "space_complexity": "Optimized",
        "inefficiencies_found": ["Optimized via AI fallback."],
        "optimal_cpp_code": raw_text,
        "explanation": "Optimized solution generated by AI fallback model.",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def serve_frontend():
    return send_file(INDEX_HTML)


@app.route("/analyze", methods=["POST"])
def analyze_code():
    body = request.get_json(silent=True)
    if not body:
        return jsonify({"detail": "Invalid JSON in request body."}), 400

    problem_description = (body.get("problem_description") or "").strip()
    user_code = (body.get("user_code") or "").strip()

    if not problem_description or not user_code:
        return jsonify(
            {"detail": "Both 'problem_description' and 'user_code' are required."}
        ), 422

    user_message = (
        f"### Problem Description\n{problem_description}\n\n"
        f"### User's C++ Code\n```cpp\n{user_code}\n```"
    )

    # Common OpenAI-compatible message array
    chat_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]

    # -----------------------------------------------------------------------
    # Layer 1: Primary Request (Gemini gemini-2.5-flash)
    # -----------------------------------------------------------------------
    gemini_payload = {
        "contents": [
            {
                "parts": [{"text": SYSTEM_PROMPT + "\n\n" + user_message}]
            }
        ],
        "generationConfig": {
            "temperature": 0.2,
            "responseMimeType": "application/json",
        },
    }

    try:
        logger.info("Executing primary request to Gemini (%s)...", PRIMARY_MODEL)
        gemini_resp = http_requests.post(GEMINI_URL, json=gemini_payload, timeout=60)
        gemini_resp.raise_for_status()

        result = _extract_gemini_result(gemini_resp.json())
        logger.info("Primary Gemini model (%s) succeeded.", PRIMARY_MODEL)
        return jsonify(result)

    except http_requests.exceptions.RequestException as exc:
        print(f"[GEMINI ERROR] Primary model failed: {exc}")
        logger.error("Primary Gemini model failed with error: %s", exc)
    except (json.JSONDecodeError, ValueError, KeyError) as exc:
        print(f"[GEMINI PARSE ERROR] Unparseable response: {exc}")
        logger.error("Primary Gemini model returned unparseable data: %s", exc)

    # -----------------------------------------------------------------------
    # Layer 2: Secondary Request (GitHub Models GPT-4o)
    # -----------------------------------------------------------------------
    try:
        logger.info("Triggering secondary fallback to GitHub Models (GPT-4o)...")
        github_headers = {
            "Authorization": f"Bearer {GITHUB_TOKEN}",
            "Content-Type": "application/json",
        }
        github_payload = {
            "model": FALLBACK_MODEL,
            "messages": chat_messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        github_resp = http_requests.post(
            GITHUB_MODELS_URL,
            headers=github_headers,
            json=github_payload,
            timeout=60,
        )
        github_resp.raise_for_status()

        github_data = github_resp.json()
        gpt_content = github_data["choices"][0]["message"]["content"]
        result = _parse_chat_completion_content(gpt_content)

        logger.info("Secondary model (%s) succeeded.", FALLBACK_MODEL)
        return jsonify(result)

    except Exception as github_exc:
        print(f"[GITHUB ERROR] Secondary model (GPT-4o) failed: {github_exc}")
        logger.error("Secondary model (GPT-4o) failed with error: %s", github_exc)

    # -----------------------------------------------------------------------
    # Layer 3: Tertiary Request (Groq API llama-3.1-8b-instant)
    # -----------------------------------------------------------------------
    try:
        logger.info("Triggering tertiary fallback to Groq API (%s)...", GROQ_MODEL)
        groq_headers = {
            "Authorization": f"Bearer {os.getenv('GROQ_API_KEY')}",
            "Content-Type": "application/json",
        }
        groq_payload = {
            "model": GROQ_MODEL,
            "messages": chat_messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        groq_resp = http_requests.post(
            GROQ_URL,
            headers=groq_headers,
            json=groq_payload,
            timeout=60,
        )

        # If llama-3.1-8b-instant returns 404 in this region, sub-fallback to active Groq model
        if groq_resp.status_code == 404:
            logger.warning("Groq model %s not found (404), falling back to openai/gpt-oss-20b...", GROQ_MODEL)
            groq_payload["model"] = "openai/gpt-oss-20b"
            groq_resp = http_requests.post(
                GROQ_URL,
                headers=groq_headers,
                json=groq_payload,
                timeout=60,
            )

        groq_resp.raise_for_status()

        groq_data = groq_resp.json()
        groq_content = groq_data["choices"][0]["message"]["content"]
        result = _parse_chat_completion_content(groq_content)

        logger.info("Tertiary model (Groq) succeeded.")
        return jsonify(result)

    except Exception as groq_exc:
        print(f"[GROQ ERROR] Tertiary model (Groq) failed: {groq_exc}")
        logger.error("Tertiary model (Groq) failed with error: %s", groq_exc)

    # -----------------------------------------------------------------------
    # Layer 4: Quaternary Request (OpenRouter meta-llama/llama-3.1-8b-instruct:free)
    # -----------------------------------------------------------------------
    try:
        logger.info("Triggering quaternary fallback to OpenRouter API (%s)...", OPENROUTER_MODEL)
        openrouter_headers = {
            "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
            "Content-Type": "application/json",
        }
        openrouter_payload = {
            "model": OPENROUTER_MODEL,
            "messages": chat_messages,
            "temperature": 0.2,
            "response_format": {"type": "json_object"},
        }

        openrouter_resp = http_requests.post(
            OPENROUTER_URL,
            headers=openrouter_headers,
            json=openrouter_payload,
            timeout=60,
        )

        # If model is unavailable (404), sub-fallback to active free model
        if openrouter_resp.status_code == 404:
            logger.warning(
                "OpenRouter model %s not found (404), falling back to nvidia/nemotron-3.5-lightning:free...",
                OPENROUTER_MODEL,
            )
            openrouter_payload["model"] = "nvidia/nemotron-3.5-lightning:free"
            openrouter_resp = http_requests.post(
                OPENROUTER_URL,
                headers=openrouter_headers,
                json=openrouter_payload,
                timeout=60,
            )

        openrouter_resp.raise_for_status()

        openrouter_data = openrouter_resp.json()
        openrouter_content = openrouter_data["choices"][0]["message"]["content"]
        result = _parse_chat_completion_content(openrouter_content)

        logger.info("Quaternary model (OpenRouter) succeeded.")
        return jsonify(result)

    except Exception as openrouter_exc:
        print(f"[OPENROUTER ERROR] Quaternary model (OpenRouter) failed: {openrouter_exc}")
        logger.error("Quaternary model (OpenRouter) failed with error: %s", openrouter_exc)

    # -----------------------------------------------------------------------
    # Step 7: Final Graceful Degradation (all four providers failed)
    # -----------------------------------------------------------------------
    logger.critical("All four AI providers (Gemini, GitHub, Groq, OpenRouter) failed.")
    return jsonify({
        "error": "All AI models are currently unavailable. Please try again."
    }), 503


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=True)
