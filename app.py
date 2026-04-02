"""Vernacular Loan Counselor: Sarvam STT/TTS + Groq LLM + FAQ retrieval + deterministic tools."""

from __future__ import annotations

import base64
import json
import os
import re
import uuid
from io import BytesIO
from typing import Any, Optional

import requests
import streamlit as st
from audio_recorder_streamlit import audio_recorder
from dotenv import load_dotenv
from groq import Groq

from rag_faq import (
    FAQDoc,
    build_langchain_faq_runtime,
    fetch_homefirst_faqs,
    retrieve_faq_context,
    run_langchain_faq_agent,
)
from tools import calculate_emi, check_eligibility


load_dotenv()

SARVAM_KEY = os.getenv("SARVAM_API_KEY")
GROQ_KEY = os.getenv("GROQ_API_KEY")
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

SARVAM_STT_URL = "https://api.sarvam.ai/speech-to-text"
SARVAM_TTS_URL = "https://api.sarvam.ai/text-to-speech"
LLM_MODEL = "llama-3.1-8b-instant"
TTS_MAX_CHARS = 500

_ONES = [
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
]
_TEENS = [
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
]
_TENS = [
    "",
    "",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
]

SYSTEM_PROMPT = """
You are HomeFirst Vernacular Loan Counselor.
Rules:
1) Always respond in the locked language: {locked_language}.
2) Use the proper script for the locked language: English for en-IN, Devanagari for hi-IN and mr-IN, Tamil script for ta-IN.
3) Do not transliterate the answer into Latin letters when the locked language is Hindi, Marathi, or Tamil.
4) Home-loan only. If user asks unrelated topics (personal loan, car loan, jokes, politics), politely redirect to home-loan counseling.
5) Never do eligibility math directly. Use tools.
6) Use FAQ context when user asks policy/document/process/eligibility questions.
7) Be concise and practical.
""".strip()

INTENT_KEYWORDS = [
    "apply",
    "loan",
    "home loan",
    "emi",
    "eligibility",
    "interest",
    "rate",
    "property",
    "documents",
    "co-applicant",
    "self-employed",
    "salaried",
]


TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "calculate_emi",
            "description": "Calculate EMI for a requested home loan amount.",
            "parameters": {
                "type": "object",
                "properties": {
                    "principal": {"type": "number"},
                    "annual_rate": {"type": "number", "default": 9.2},
                    "years": {"type": "integer", "default": 20},
                },
                "required": ["principal"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_eligibility",
            "description": "Check home loan eligibility using deterministic policy rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "monthly_income": {"type": "number"},
                    "loan_amount": {"type": "number"},
                    "property_value": {"type": "number"},
                    "employment_status": {"type": "string"},
                    "annual_rate": {"type": "number", "default": 9.2},
                    "years": {"type": "integer", "default": 20},
                },
                "required": ["monthly_income", "loan_amount", "property_value", "employment_status"],
            },
        },
    },
]


def require_env() -> None:
    missing = [name for name, value in {"SARVAM_API_KEY": SARVAM_KEY, "GROQ_API_KEY": GROQ_KEY}.items() if not value]
    if missing:
        st.error(f"Missing environment variables: {', '.join(missing)}")
        st.stop()


def detect_language(text: str) -> str:
    text_l = text.lower()
    if any("\u0b80" <= ch <= "\u0bff" for ch in text):
        return "ta-IN"
    if any("\u0900" <= ch <= "\u097f" for ch in text):
        marathi_markers = [
            "majha",
            "ahe",
            "tumhi",
            "marathi",
            "माझ",
            "आहे",
            "तुम्ही",
            "पाहिजे",
            "आहेत",
            "काय",
            "करायच",
            "हवे",
            "मला",
        ]
        if any(word in text_l for word in marathi_markers):
            return "mr-IN"
        return "hi-IN"
    return "en-IN"


def _is_definitely_english(text: str) -> bool:
    """Check if text is definitely English (pure ASCII, no Hinglish markers)."""
    if not text.strip():
        return False
    # If contains non-ASCII, it's not definitely English
    if any(ord(ch) > 127 for ch in text):
        return False
    # Check for common English greetings and phrases
    text_lower = text.lower()
    common_english = [
        "hello",
        "hi",
        "good",
        "morning",
        "afternoon",
        "want",
        "information",
        "loan",
        "help",
        "please",
        "my name",
        "i am",
        "interested",
    ]
    english_count = sum(1 for phrase in common_english if phrase in text_lower)
    return english_count >= 1


def detect_language_safe(text: str, client: Groq) -> str:
    """Detect language with heuristic first, LLM only for ambiguous cases."""
    # Try heuristic first
    heuristic_lang = detect_language(text)
    
    # If text is definitely English or clearly has script markers, trust heuristic
    if _is_definitely_english(text):
        return "en-IN"
    if any("\u0b80" <= ch <= "\u0bff" for ch in text):
        return "ta-IN"
    if any("\u0900" <= ch <= "\u097f" for ch in text):
        return detect_language(text)
    
    # For ambiguous cases, use LLM but don't let it override clear signals
    try:
        llm_lang = detect_language_with_llm(client, text)
        if llm_lang:
            return llm_lang
    except Exception:
        pass
    
    return heuristic_lang


def transcribe_audio(file_bytes: bytes, filename: str, language_code: Optional[str] = None) -> str:
    files = {"file": (filename, file_bytes, "audio/wav")}
    data = {"model": "saarika:v2.5"}
    if language_code:
        data["language_code"] = language_code
    headers = {"api-subscription-key": SARVAM_KEY}
    resp = requests.post(SARVAM_STT_URL, headers=headers, files=files, data=data, timeout=45)
    if resp.status_code != 200:
        raise RuntimeError(f"STT error {resp.status_code}: {resp.text}")
    payload = resp.json()
    return (payload.get("transcript") or "").strip()


def _extract_audio_bytes_from_json(payload: dict[str, Any]) -> bytes:
    keys = ["audio", "audio_base64", "base64_audio"]
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return base64.b64decode(value)

    audios = payload.get("audios")
    if isinstance(audios, list) and audios and isinstance(audios[0], str):
        return base64.b64decode(audios[0])

    outputs = payload.get("outputs")
    if isinstance(outputs, list) and outputs and isinstance(outputs[0], dict):
        first = outputs[0]
        for key in keys:
            value = first.get(key)
            if isinstance(value, str) and value.strip():
                return base64.b64decode(value)

    raise RuntimeError(f"TTS response missing audio payload. Keys: {list(payload.keys())}")


def _detect_audio_format(audio_bytes: bytes) -> str:
    if audio_bytes.startswith(b"RIFF"):
        return "audio/wav"
    if audio_bytes.startswith(b"ID3") or audio_bytes[:2] == b"\xff\xfb":
        return "audio/mp3"
    if audio_bytes.startswith(b"OggS"):
        return "audio/ogg"
    return "audio/wav"


def _number_to_english_words(number: int) -> str:
    if number < 0:
        return "minus " + _number_to_english_words(abs(number))
    if number < 10:
        return _ONES[number]
    if number < 20:
        return _TEENS[number - 10]
    if number < 100:
        tens, ones = divmod(number, 10)
        return _TENS[tens] + (f" {_ONES[ones]}" if ones else "")
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        prefix = f"{_ONES[hundreds]} hundred"
        return prefix + (f" and {_number_to_english_words(rest)}" if rest else "")
    if number < 100000:
        thousands, rest = divmod(number, 1000)
        prefix = f"{_number_to_english_words(thousands)} thousand"
        return prefix + (f" {_number_to_english_words(rest)}" if rest else "")
    if number < 10000000:
        lakhs, rest = divmod(number, 100000)
        prefix = f"{_number_to_english_words(lakhs)} lakh"
        return prefix + (f" {_number_to_english_words(rest)}" if rest else "")
    crores, rest = divmod(number, 10000000)
    prefix = f"{_number_to_english_words(crores)} crore"
    return prefix + (f" {_number_to_english_words(rest)}" if rest else "")


def _number_to_devanagari_words(number: int) -> str:
    ones = [
        "शून्य",
        "एक",
        "दोन",
        "तीन",
        "चार",
        "पाच",
        "सहा",
        "सात",
        "आठ",
        "नऊ",
    ]
    teens = [
        "दहा",
        "अकरा",
        "बारा",
        "तेरा",
        "चौदा",
        "पंधरा",
        "सोळा",
        "सतरा",
        "अठरा",
        "एकोणीस",
    ]
    tens = [
        "",
        "",
        "वीस",
        "तीस",
        "चाळीस",
        "पन्नास",
        "साठ",
        "सत्तर",
        "ऐंशी",
        "नव्वद",
    ]

    if number < 0:
        return "उणे " + _number_to_devanagari_words(abs(number))
    if number < 10:
        return ones[number]
    if number < 20:
        return teens[number - 10]
    if number < 100:
        tens_digit, ones_digit = divmod(number, 10)
        return tens[tens_digit] + (f" {ones[ones_digit]}" if ones_digit else "")
    if number < 1000:
        hundreds, rest = divmod(number, 100)
        prefix = f"{ones[hundreds]}शे"
        return prefix + (f" {_number_to_devanagari_words(rest)}" if rest else "")
    if number < 100000:
        thousands, rest = divmod(number, 1000)
        prefix = f"{_number_to_devanagari_words(thousands)} हजार"
        return prefix + (f" {_number_to_devanagari_words(rest)}" if rest else "")
    if number < 10000000:
        lakhs, rest = divmod(number, 100000)
        prefix = f"{_number_to_devanagari_words(lakhs)} लाख"
        return prefix + (f" {_number_to_devanagari_words(rest)}" if rest else "")
    crores, rest = divmod(number, 10000000)
    prefix = f"{_number_to_devanagari_words(crores)} कोटी"
    return prefix + (f" {_number_to_devanagari_words(rest)}" if rest else "")


def _normalize_tts_numbers(text: str, target_language: str) -> str:
    def replace_number(match: re.Match[str]) -> str:
        raw = match.group(0)
        digits = raw.replace(",", "")
        if not digits.isdigit():
            return raw
        number = int(digits)
        if target_language == "en-IN":
            return _number_to_english_words(number)
        if target_language in {"hi-IN", "mr-IN"}:
            return _number_to_devanagari_words(number)
        return digits

    # Convert full numeric tokens first (supports plain digits and Indian comma style like 1,00,00,000).
    text = re.sub(r"\b\d[\d,]*\b", replace_number, text)
    text = text.replace("₹", "rupees ")
    text = re.sub(r"\bINR\b", "rupees", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _prepare_tts_text(text: str, target_language: str, limit: int = TTS_MAX_CHARS) -> str:
    compact = " ".join(text.split())
    compact = _normalize_tts_numbers(compact, target_language)
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def speak_text(text: str, target_language: str) -> tuple[bytes, str, str]:
    tts_text = _prepare_tts_text(text, target_language)
    payload = {
        "inputs": [tts_text],
        "target_language_code": target_language,
        "speaker": "anushka",
        "model": "bulbul:v2",
    }
    headers = {"Content-Type": "application/json", "api-subscription-key": SARVAM_KEY}
    resp = requests.post(SARVAM_TTS_URL, json=payload, headers=headers, timeout=45)
    if resp.status_code != 200:
        raise RuntimeError(f"TTS error {resp.status_code}: {resp.text}")

    content_type = (resp.headers.get("Content-Type") or "").lower()
    if "application/json" in content_type:
        audio_bytes = _extract_audio_bytes_from_json(resp.json())
    else:
        audio_bytes = resp.content

    return audio_bytes, _detect_audio_format(audio_bytes), tts_text


def render_audio_player(audio_bytes: bytes, audio_mime: str, autoplay: bool = True) -> None:
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        autoplay_attr = "autoplay" if autoplay else ""
        player_html = f"""
        <audio controls {autoplay_attr} style=\"width:100%;\">
            <source src=\"data:{audio_mime};base64,{audio_b64}\" type=\"{audio_mime}\">
            Your browser does not support the audio element.
        </audio>
        """
        st.markdown(player_html, unsafe_allow_html=True)


def get_audio_bytes() -> Optional[bytes]:
    recorded = audio_recorder(text="Press to record", pause_threshold=2.0, sample_rate=16000)
    return recorded if recorded else None


def _extract_first_json_block(raw: str) -> dict[str, Any]:
    raw = (raw or "").strip()
    if not raw:
        return {}

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {}
    try:
        return json.loads(match.group(0))
    except json.JSONDecodeError:
        return {}


def _amount_to_rupees(raw_number: str, raw_unit: Optional[str]) -> Optional[int]:
    try:
        base = float(raw_number.replace(",", "").strip())
    except (TypeError, ValueError):
        return None

    unit = (raw_unit or "").strip().lower()
    # Normalize common Devanagari variants for the same sound (z/nukta forms).
    unit = unit.replace("ज़", "ज़")
    multiplier = 1
    if unit in {"lakh", "lakhs", "lac", "lacs", "लाख"}:
        multiplier = 100000
    elif unit in {"crore", "crores", "cr", "कोटी", "करोड"}:
        multiplier = 10000000
    elif unit in {"k", "thousand", "thousands", "हजार", "हज़ार", "हज़ार"}:
        multiplier = 1000

    return int(round(base * multiplier))


def _parse_indic_word_number(text: str) -> Optional[int]:
    words = {
        "एक": 1,
        "१": 1,
        "दो": 2,
        "दोन": 2,
        "दोनही": 2,
        "२": 2,
        "तीन": 3,
        "३": 3,
        "चार": 4,
        "४": 4,
        "पाच": 5,
        "पांच": 5,
        "५": 5,
        "सहा": 6,
        "छह": 6,
        "६": 6,
        "सात": 7,
        "७": 7,
        "आठ": 8,
        "आठ": 8,
        "८": 8,
        "नऊ": 9,
        "नौ": 9,
        "९": 9,
        "दहा": 10,
        "दस": 10,
        "वीस": 20,
        "बीस": 20,
        "तीस": 30,
        "चाळीस": 40,
        "चालीस": 40,
        "पन्नास": 50,
        "पचास": 50,
        "साठ": 60,
        "सत्तर": 70,
        "ऐंशी": 80,
        "अस्सी": 80,
        "नव्वद": 90,
        "नब्बे": 90,
    }

    tokens = [t for t in re.split(r"\s+", (text or "").strip()) if t]
    if not tokens:
        return None

    total = 0
    for token in tokens:
        if token in words:
            total += words[token]
        elif token.isdigit():
            total += int(token)
    return total if total > 0 else None


def _amount_words_to_rupees(raw_words: str, raw_unit: Optional[str]) -> Optional[int]:
    number = _parse_indic_word_number(raw_words)
    if number is None:
        return None

    unit = (raw_unit or "").strip().lower()
    unit = unit.replace("ज़", "ज़")
    multiplier = 1
    if unit in {"lakh", "lakhs", "lac", "lacs", "लाख"}:
        multiplier = 100000
    elif unit in {"crore", "crores", "cr", "कोटी", "करोड"}:
        multiplier = 10000000
    elif unit in {"k", "thousand", "thousands", "हजार", "हज़ार", "हज़ार"}:
        multiplier = 1000

    return int(round(number * multiplier))


def _extract_amount_for_field(text: str, field_patterns: list[str]) -> Optional[int]:
    for field_pattern in field_patterns:
        pattern = (
            rf"(?:{field_pattern})"
            r"[^\d₹]{0,40}"
            r"₹?\s*([0-9][0-9,]*(?:\.[0-9]+)?)\s*"
            r"(lakh|lakhs|lac|lacs|crore|crores|cr|k|thousand|thousands|हजार|हज़ार|हज़ार|लाख|कोटी|करोड)?"
        )
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            amount = _amount_to_rupees(match.group(1), match.group(2))
            if amount is not None:
                return amount

        word_pattern = (
            rf"(?:{field_pattern})"
            r"[^\w\d₹]{0,40}"
            r"([A-Za-z\u0900-\u097f\s]+?)\s*"
            r"(हजार|हज़ार|हज़ार|thousand|thousands|lakh|lakhs|lac|lacs|crore|crores|cr|लाख|कोटी|करोड)\s*"
            r"(?:रुपये|रुपया|rupees|rs\.?|$)"
        )
        word_match = re.search(word_pattern, text, flags=re.IGNORECASE)
        if word_match:
            amount = _amount_words_to_rupees(word_match.group(1), word_match.group(2))
            if amount is not None:
                return amount
    return None


def _extract_structured_amounts(user_text: str) -> dict[str, Optional[int]]:
    text = user_text or ""
    monthly_income = _extract_amount_for_field(
        text,
        [
            r"monthly\s+income",
            r"income",
            r"मासिक\s+उत्पन्न",
            r"उत्पन्न",
            r"मासिक\s+आय",
            r"मंथली\s+इनकम",
        ],
    )
    loan_amount = _extract_amount_for_field(
        text,
        [
            r"loan\s+amount",
            r"home\s+loan",
            r"loan",
            r"कर्जाची\s+रक्कम",
            r"कर्ज\s+रक्कम",
            r"कर्ज",
            r"ऋणाची\s+रक्कम",
            r"ऋण\s+रक्कम",
            r"ऋण",
            r"ऋण\s+की\s+राशि",
            r"ऋण\s+राशि",
            r"लोन\s+राशि",
        ],
    )
    property_value = _extract_amount_for_field(
        text,
        [
            r"property\s+value",
            r"property",
            r"house\s+value",
            r"मालमत्तेची\s+किंमत",
            r"मालमत्ता\s+किंमत",
            r"मालमत्ता",
            r"संपत्ति\s+का\s+मूल्य",
            r"संपत्ति\s+मूल्य",
            r"प्रॉपर्टी\s+वैल्यू",
        ],
    )
    return {
        "monthly_income": monthly_income,
        "loan_amount_requested": loan_amount,
        "property_value": property_value,
    }


def _extract_employment_status(user_text: str) -> str:
    text_l = (user_text or "").lower().strip()
    text_l = text_l.replace("ज़", "ज़")
    if any(k in text_l for k in ["unemployed", "not employed", "no job", "बेरोजगार"]):
        return "unknown"
    if any(
        k in text_l
        for k in [
            "self-employed",
            "self employed",
            "business owner",
            "own business",
            "स्वनियोजित",
            "स्वतःची नोकरी",
            "स्वतःचा व्यवसाय",
            "व्यवसाय करतो",
            "व्यवसाय करते",
            "स्व-नियोजित",
            "स्वनियोजित",
            "स्व-रोजगार",
            "खुदकामी",
        ]
    ):
        return "self-employed"
    if any(
        k in text_l
        for k in [
            "salaried",
            "salary",
            "job",
            "employed",
            "नोकरी",
            "पगारदार",
            "नियमित",
            "नियमित नौकरी",
            "नौकरी की स्थिति नियमित",
            "रोजगार की स्थिति नौकरी",
            "रोजगार का स्थिति नौकरी",
            "नौकरी",
            "नौकरी पेशा",
            "नौकरीपेशा",
            "नौकरी पेशा कर्मचारी",
            "सैलरी",
            "वेतन",
            "वेतनभोगी",
            "कर्मचारी",
            "सेवा",
        ]
    ):
        return "salaried"
    return "unknown"


def extract_entities(client: Groq, user_text: str) -> dict[str, Any]:
    prompt = (
        "Extract entities from user input and return strict JSON only with keys: "
        "monthly_income, annual_income, property_value, loan_amount_requested, employment_status, tenure_years, interest_rate_percent, intent. "
        "Normalize all money values into rupees as numbers. For example, '5 lakh' becomes 500000. "
        "Normalize time values into integer years. If the user gives annual income, monthly_income may be inferred as annual_income/12. "
        "If the user asks for EMI, keep loan_amount_requested and tenure_years whenever mentioned. "
        "employment_status should be exactly salaried, self-employed, or unknown. "
        "Use null when unknown."
    )

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        max_completion_tokens=220,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_text},
        ],
    )
    content = resp.choices[0].message.content or "{}"
    data = _extract_first_json_block(content)

    # Use deterministic extraction for critical financial fields to avoid LLM hallucinations.
    parsed_amounts = _extract_structured_amounts(user_text)
    employment_status = _extract_employment_status(user_text)

    normalized = {
        "monthly_income": parsed_amounts.get("monthly_income"),
        "annual_income": None,
        "property_value": parsed_amounts.get("property_value"),
        "loan_amount_requested": parsed_amounts.get("loan_amount_requested"),
        "employment_status": employment_status,
        "tenure_years": data.get("tenure_years"),
        "interest_rate_percent": data.get("interest_rate_percent"),
        "intent": data.get("intent") or "unknown",
    }

    # Deterministic correction for common monetary misreads (for example, 70,000 -> 700,000)
    # and ensure parsed amounts always win over LLM output.
    for key, value in parsed_amounts.items():
        if value is not None:
            normalized[key] = value

    if normalized.get("monthly_income") not in (None, "", 0):
        normalized["annual_income"] = int(round(float(normalized["monthly_income"]) * 12))

    return normalized


def detect_language_with_llm(client: Groq, user_text: str) -> Optional[str]:
    prompt = (
        "Classify the user's primary language for this conversation. "
        "Choose exactly one code from: en-IN, hi-IN, mr-IN, ta-IN. "
        "Decide from meaning and wording, not only script. "
        "Return strict JSON only: {\"language_code\": \"...\"}."
    )

    try:
        resp = client.chat.completions.create(
            model=LLM_MODEL,
            temperature=0,
            max_completion_tokens=80,
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_text},
            ],
        )
        content = resp.choices[0].message.content or "{}"
        data = _extract_first_json_block(content)
        code = (data.get("language_code") or "").strip()
        if code in {"en-IN", "hi-IN", "mr-IN", "ta-IN"}:
            return code
    except Exception:
        return None

    return None


def _is_out_of_domain(user_text: str) -> bool:
    text_l = user_text.lower()
    blocked = ["personal loan", "car loan", "credit card", "stocks", "crypto", "politics", "joke"]
    return any(x in text_l for x in blocked)


def _is_irrelevant_or_nonsense(user_text: str) -> bool:
    text_l = (user_text or "").lower().strip()
    if not text_l:
        return True

    if _is_out_of_domain(text_l):
        return True

    greetings = [
        "hello",
        "hi",
        "hey",
        "good morning",
        "good evening",
        "namaste",
        "नमस्ते",
        "हेलो",
        "vanakkam",
        "வணக்கம்",
    ]
    intro_markers = [
        "my name is",
        "i am",
        "मेरा नाम",
        "माझं नाव",
        "माझे नाव",
        "என் பெயர்",
    ]
    domain_terms = [
        "home loan",
        "loan",
        "emi",
        "eligibility",
        "interest",
        "rate",
        "property",
        "tenure",
        "document",
        "apply",
        "mortgage",
        "repay",
        "prepayment",
        "co-applicant",
        "salaried",
        "self-employed",
        "होम लोन",
        "गृह कर्ज",
        "पात्रता",
        "पात्रता शोध",
        "ईएमआई",
        "मासिक हप्ता",
        "व्याज",
        "मालमत्ता",
        "कर्ज",
        "ऋण",
        "कर्जाची रक्कम",
        "ऋणाची रक्कम",
        "गृह ऋण",
        "eligibilty",
        "पात्रता जांच",
        "होम लोन जानकारी",
        "வீட்டு கடன்",
        "தகுதி",
        "EMI கணக்கு",
    ]

    if any(g in text_l for g in greetings) and len(text_l.split()) <= 10:
        return False

    if any(marker in text_l for marker in intro_markers):
        return False

    if any(t in text_l for t in domain_terms):
        return False

    words = re.findall(r"[a-zA-Z\u0900-\u097f\u0b80-\u0bff]+", user_text)
    return len(words) >= 3


def _build_out_of_scope_reply(locked_language: str) -> str:
    if locked_language == "hi-IN":
        return "माफ कीजिए, मैं होम लोन असिस्टेंट हूं। मैं केवल होम लोन से जुड़ी जानकारी में ही मदद कर सकता हूं।"
    if locked_language == "mr-IN":
        return "माफ करा, मी होम लोन असिस्टंट आहे. मी फक्त होम लोनसंबंधित माहितीमध्येच मदत करू शकतो."
    if locked_language == "ta-IN":
        return "மன்னிக்கவும், நான் ஹோம் லோன் உதவியாளர். ஹோம் லோன் தொடர்பான தகவல்களில் மட்டும் உதவ முடியும்."
    return "Sorry, I am a home loan assistant. I can only help with relevant home loan information."


def _is_faq_query(user_text: str) -> bool:
    text_l = user_text.lower()
    faq_keys = ["document", "documents", "eligibility", "co-applicant", "prepayment", "tenure", "interest", "repay", "tax"]
    return any(x in text_l for x in faq_keys)


def _dedupe_paragraphs(text: str) -> str:
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    if not parts:
        return text.strip()

    seen: set[str] = set()
    keep: list[str] = []
    for part in parts:
        key = re.sub(r"\s+", " ", part.lower()).strip()
        if key in seen:
            continue
        seen.add(key)
        keep.append(part)
    return "\n\n".join(keep)


def _has_non_english_script(text: str) -> bool:
    for ch in text:
        if ("\u0900" <= ch <= "\u097f") or ("\u0b80" <= ch <= "\u0bff"):
            return True
    return False


def _has_latin_script(text: str) -> bool:
    return any(("A" <= ch <= "Z") or ("a" <= ch <= "z") for ch in text)


def _locked_language_style_instruction(locked_language: str) -> str:
    if locked_language == "en-IN":
        return "Write only in English. Do not use Devanagari or Tamil script."
    if locked_language in {"hi-IN", "mr-IN"}:
        return "Write only in Devanagari script. Do not transliterate into Latin letters."
    if locked_language == "ta-IN":
        return "Write only in Tamil script. Do not transliterate into Latin letters."
    return "Write only in the locked language."


def _rewrite_in_locked_language(client: Groq, text: str, locked_language: str) -> str:
    rewrite_prompt = (
        "Rewrite the assistant response in the exact locked language only. "
        "Keep facts unchanged, remove repetition, keep it concise (max 120 words), "
        "and stay only in home-loan counseling scope. "
        + _locked_language_style_instruction(locked_language)
    )
    out = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.1,
        max_completion_tokens=220,
        messages=[
            {"role": "system", "content": f"Locked language: {locked_language}. {rewrite_prompt}"},
            {"role": "user", "content": text},
        ],
    )
    return (out.choices[0].message.content or "").strip() or text


def _detect_hinglish(text: str) -> bool:
    """Detect romanized Hindi (Hinglish) patterns in Latin script."""
    text_lower = text.lower()
    hinglish_markers = [
        r"\baap\b",
        r"\bkya\b",
        r"\bhain\b",
        r"\bmain\b",
        r"\baapke\b",
        r"\baapki\b",
        r"\bmadad\b",
        r"\byahaan\b",
        r"\bke liye\b",
        r"\baur\b",
        r"\bpar\b",
        r"\bjeena\b",
        r"\bliya\b",
        r"\biye\b",
        r"\boge\b",
        r"\bhoga\b",
        r"\bsakte\b",
        r"\bsakti\b",
        r"\bdiya\b",
        r"\bdiye\b",
    ]
    return sum(1 for marker in hinglish_markers if re.search(marker, text_lower)) >= 2


def _english_fallback_reply(raw_reply: str) -> str:
    if not raw_reply.strip():
        return "I can help with home-loan questions. Please share your income, loan amount, and property value."
    return "I can help with home-loan questions. Please share your income, loan amount, and property value."


def finalize_reply(client: Groq, raw_reply: str, locked_language: str) -> str:
    cleaned = _dedupe_paragraphs(raw_reply or "")
    cleaned = re.sub(r"\s+", " ", cleaned).strip()

    # Force the proper script for the locked language.
    if locked_language == "en-IN":
        # Check for non-English Unicode script OR Hinglish
        if _has_non_english_script(cleaned) or _detect_hinglish(cleaned):
            cleaned = _rewrite_in_locked_language(client, cleaned, locked_language)
            if _has_non_english_script(cleaned) or _detect_hinglish(cleaned):
                cleaned = _english_fallback_reply(cleaned)
    elif locked_language in {"hi-IN", "mr-IN"} and _has_latin_script(cleaned):
        cleaned = _rewrite_in_locked_language(client, cleaned, locked_language)
    elif locked_language == "ta-IN" and _has_latin_script(cleaned):
        cleaned = _rewrite_in_locked_language(client, cleaned, locked_language)

    # Keep UI and TTS stable; avoid long repetitive outputs.
    if len(cleaned) > 700:
        cleaned = cleaned[:697].rstrip() + "..."

    return cleaned


def _looks_uncertain_answer(text: str) -> bool:
    t = (text or "").lower()
    uncertain_markers = [
        "i do not know",
        "i don't know",
        "not sure",
        "cannot determine",
        "can't determine",
        "please provide more details",
    ]
    return any(m in t for m in uncertain_markers)


def _is_high_intent(user_text: str, entities: dict[str, Any]) -> bool:
    text_l = user_text.lower()
    intent_signal = any(x in text_l for x in ["apply", "process", "start", "proceed", "next step", "loan chahiye", "apply now"])
    complete_financials = all(
        entities.get(k) not in (None, "", 0) for k in ["monthly_income", "property_value", "loan_amount_requested"]
    ) and (entities.get("employment_status") or "unknown") != "unknown"
    return intent_signal and complete_financials


def _format_inr(value: Any) -> str:
    try:
        number = int(round(float(value or 0)))
    except (TypeError, ValueError):
        return "0"

    sign = "-" if number < 0 else ""
    s = str(abs(number))
    if len(s) <= 3:
        return sign + s

    last_three = s[-3:]
    rest = s[:-3]
    parts: list[str] = []
    while len(rest) > 2:
        parts.insert(0, rest[-2:])
        rest = rest[:-2]
    if rest:
        parts.insert(0, rest)
    return sign + ",".join(parts + [last_three])


def build_compact_eligibility_reply(entities: dict[str, Any], tool_result: dict[str, Any]) -> str:
    monthly_income = _format_inr(entities.get("monthly_income"))
    property_value = _format_inr(entities.get("property_value"))
    loan_amount = _format_inr(entities.get("loan_amount_requested"))
    status = (entities.get("employment_status") or "unknown").replace("_", " ").title()
    eligible = bool(tool_result.get("eligible"))

    decision_line = (
        "I have put your details into our eligibility calculator, and you are eligible for the home loan."
        if eligible
        else "I have put your details into our eligibility calculator, and currently you are not eligible for the home loan."
    )

    return (
        f"{decision_line}\n\n"
        f"Here are the details I checked:\n"
        f"- Monthly Income: INR {monthly_income}\n"
        f"- Property Value: INR {property_value}\n"
        f"- Loan Amount: INR {loan_amount}\n"
        f"- Employment Status: {status}"
    )


def _missing_eligibility_fields(entities: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if entities.get("monthly_income") in (None, "", 0):
        missing.append("monthly income")
    if entities.get("loan_amount_requested") in (None, "", 0):
        missing.append("loan amount")
    if entities.get("property_value") in (None, "", 0):
        missing.append("property value")
    if (entities.get("employment_status") or "unknown") == "unknown":
        missing.append("employment status (salaried or self-employed)")
    return missing


def _is_eligibility_query(user_text: str, entities: dict[str, Any]) -> bool:
    text_l = (user_text or "").lower()
    intent = str(entities.get("intent") or "").lower()
    hints = ["eligible", "eligibility", "am i eligible", "loan check", "check eligibility", "home loan"]
    return ("elig" in intent) or any(h in text_l for h in hints)


def _build_missing_fields_reply(missing_fields: list[str], locked_language: str) -> str:
    if locked_language == "hi-IN":
        return (
            "पात्रता जांच के लिए कृपया केवल ये जानकारी दें: "
            "मासिक आय, ऋण राशि, संपत्ति का मूल्य, और रोजगार की स्थिति "
            "(नौकरीपेशा या स्व-नियोजित)।"
        )
    if locked_language == "mr-IN":
        return (
            "पात्रता तपासणीसाठी कृपया फक्त ही माहिती द्या: "
            "मासिक उत्पन्न, कर्जाची रक्कम, मालमत्तेची किंमत, आणि रोजगार स्थिती "
            "(नोकरीपेशा किंवा स्वनियोजित)."
        )
    if locked_language == "ta-IN":
        return (
            "தகுதி சரிபார்க்க, தயவுசெய்து இந்த தகவல்கள் மட்டும் அளிக்கவும்: "
            "மாத வருமானம், கடன் தொகை, சொத்து மதிப்பு, வேலை நிலை (ஊதியப்பணி அல்லது சுயதொழில்)."
        )

    if not missing_fields:
        return "Please share monthly income, loan amount, property value, and employment status to check eligibility."
    needed = ", ".join(missing_fields)
    return f"To check eligibility, please share only these details: {needed}."


def _choose_user_flow(user_text: str, entities: dict[str, Any]) -> str:
    text_l = (user_text or "").lower()

    emi_hints = [
        "emi",
        "monthly installment",
        "installment",
        "calculate emi",
        "ईएमआई",
        "मासिक हप्ता",
        "emi गणना",
        "ईएमआई कॅल्क्युलेट",
        "ஈஎம்ஐ",
    ]
    elig_hints = [
        "eligible",
        "eligibility",
        "am i eligible",
        "check eligibility",
        "loan check",
        "पात्रता",
        "पात्रता शोध",
        "पात्रता जांच",
        "योग्य",
        "தகுதி",
    ]

    if any(h in text_l for h in emi_hints):
        return "emi"
    if any(h in text_l for h in elig_hints):
        return "eligibility"
    return "general"


def _explicit_requested_flow(user_text: str) -> Optional[str]:
    text_l = (user_text or "").lower()
    emi_markers = [
        "emi",
        "ईएमआई",
        "ईएमआय",
        "मासिक हप्ता",
        "emi गणना",
        "ईएमआय कॅल्क्युलेट",
        "ईएमआई कॅल्क्युलेट",
        "कॅल्क्युलेट",
        "calculate emi",
        "ஈஎம்ஐ",
    ]
    eligibility_markers = ["eligibility", "eligible", "पात्रता", "पात्रता शोध", "पात्रता तपास", "தகுதி"]

    if any(x in text_l for x in emi_markers):
        return "emi"
    if any(x in text_l for x in eligibility_markers):
        return "eligibility"
    return None


def _build_welcome_menu_reply() -> str:
    return (
        "Hello! Welcome to Home Loan Counselor. How can I help you today?\n"
        "I can help with:\n"
        "1) Home loan information\n"
        "2) Eligibility check\n"
        "3) Monthly EMI calculation\n"
        "Please tell me which one you want."
    )


def _missing_emi_fields(entities: dict[str, Any]) -> list[str]:
    missing: list[str] = []
    if entities.get("loan_amount_requested") in (None, "", 0):
        missing.append("loan amount")
    if entities.get("tenure_years") in (None, "", 0):
        missing.append("tenure in years")
    return missing


def _build_missing_emi_fields_reply(missing_fields: list[str]) -> str:
    if not missing_fields:
        return "To calculate EMI, please share loan amount and tenure in years."
    needed = ", ".join(missing_fields)
    return f"To calculate EMI, please share only these details: {needed}."


def build_compact_emi_reply(entities: dict[str, Any], tool_result: dict[str, Any]) -> str:
    loan_amount = _format_inr(entities.get("loan_amount_requested"))
    years = int(tool_result.get("years") or entities.get("tenure_years") or 20)
    emi = _format_inr(tool_result.get("emi"))
    return (
        f"Loan Amount: INR {loan_amount}\n"
        f"Tenure: {years} years\n"
        f"Estimated Monthly EMI: INR {emi}"
    )


def _merge_entities(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    merged = dict(previous)
    for key in [
        "monthly_income",
        "annual_income",
        "property_value",
        "loan_amount_requested",
        "tenure_years",
        "interest_rate_percent",
    ]:
        value = current.get(key)
        if value not in (None, "", 0):
            merged[key] = value

    current_status = current.get("employment_status")
    if current_status and current_status != "unknown":
        merged["employment_status"] = current_status

    current_intent = current.get("intent")
    if current_intent and current_intent != "unknown":
        merged["intent"] = current_intent

    return merged


def _maybe_log_conversation(
    session_id: str,
    user_text: str,
    assistant_text: str,
    locked_language: str,
    entities: dict[str, Any],
    tool_result: Optional[dict[str, Any]],
    rag_mode: str,
) -> str:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "SUPABASE_NOT_CONFIGURED"

    try:
        from supabase import create_client

        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        row = {
            "session_id": session_id,
            "user_text": user_text,
            "assistant_text": assistant_text,
            "locked_language": locked_language,
            "rag_mode": rag_mode,
            "entities_json": entities,
            "tool_result_json": tool_result,
        }
        client.table("conversation_logs").insert(row).execute()
        return "SUPABASE_CONVERSATION_LOGGED"
    except Exception as exc:
        return f"SUPABASE_CONVERSATION_LOG_FAILED: {str(exc)[:140]}"


def _maybe_log_handoff(entities: dict[str, Any], eligibility_payload: dict[str, Any]) -> str:
    if not SUPABASE_URL or not SUPABASE_KEY:
        return "SUPABASE_NOT_CONFIGURED"

    try:
        from supabase import create_client

        client = create_client(SUPABASE_URL, SUPABASE_KEY)
        row = {
            "customer_name": "Unknown",
            "monthly_income": entities.get("monthly_income") or 0,
            "loan_amount": entities.get("loan_amount_requested") or 0,
            "status": "Eligible" if eligibility_payload.get("eligible") else "Rejected",
        }
        client.table("loan_leads").insert(row).execute()
        return "SUPABASE_LOGGED"
    except Exception:
        return "SUPABASE_LOG_FAILED"


def _build_assistant_reply(
    client: Groq,
    user_text: str,
    locked_language: str,
    faq_context: list[FAQDoc],
    tool_result: Optional[dict[str, Any]],
    out_of_domain: bool,
    conversation_history: Optional[list[dict[str, str]]] = None,
) -> str:
    faq_block = "\n".join([f"Q: {d.question}\nA: {d.answer}" for d in faq_context])
    policy_msg = (
        "User asked out-of-domain question. Politely redirect to home-loan counseling only."
        if out_of_domain
        else "Stay focused on home-loan counseling."
    )

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(locked_language=locked_language)},
        {"role": "system", "content": policy_msg},
        {"role": "system", "content": "FAQ context:\n" + faq_block},
    ]

    if conversation_history:
        for m in conversation_history[-6:]:
            if m.get("role") in {"user", "assistant"} and m.get("content"):
                messages.append({"role": m["role"], "content": m["content"]})

    messages.append({"role": "user", "content": user_text})

    if tool_result:
        messages.append({"role": "system", "content": "Deterministic tool output (source of truth): " + json.dumps(tool_result)})

    out = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        max_completion_tokens=500,
        messages=messages,
    )
    return out.choices[0].message.content or "Please share a bit more detail about your home loan requirement."


def maybe_run_tools(client: Groq, entities: dict[str, Any], requested_flow: str = "general") -> tuple[bool, Optional[dict[str, Any]]]:
    enough_for_eligibility = all(
        entities.get(k) not in (None, "", 0) for k in ["monthly_income", "property_value", "loan_amount_requested"]
    ) and entities.get("employment_status") not in (None, "", "unknown")

    enough_for_emi = all(entities.get(k) not in (None, "", 0) for k in ["loan_amount_requested", "tenure_years"])

    if requested_flow == "eligibility" and not enough_for_eligibility:
        return False, None
    if requested_flow == "emi" and not enough_for_emi:
        return False, None
    if requested_flow == "general":
        return False, None

    if not enough_for_eligibility and not enough_for_emi:
        return False, None

    tenure_years = int(float(entities.get("tenure_years") or 20))

    # Restrict available tools by selected flow and available fields.
    if requested_flow == "eligibility":
        available_tools = [TOOL_SCHEMAS[1]]
        chosen_tool = {"type": "function", "function": {"name": "check_eligibility"}}
    elif requested_flow == "emi":
        available_tools = [TOOL_SCHEMAS[0]]
        chosen_tool = {"type": "function", "function": {"name": "calculate_emi"}}
    else:
        available_tools = [TOOL_SCHEMAS[1], TOOL_SCHEMAS[0]] if enough_for_eligibility else [TOOL_SCHEMAS[0]]
        chosen_tool = "auto" if enough_for_eligibility else {"type": "function", "function": {"name": "calculate_emi"}}

    # Ask model to pick function call; fallback to deterministic direct call.
    tool_messages = [
        {
            "role": "system",
            "content": "If eligibility fields exist, call check_eligibility. Otherwise if loan amount exists, call calculate_emi.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "monthly_income": entities.get("monthly_income"),
                    "loan_amount": entities.get("loan_amount_requested"),
                    "property_value": entities.get("property_value"),
                    "employment_status": entities.get("employment_status"),
                    "years": tenure_years,
                }
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        max_completion_tokens=300,
        tools=available_tools,
        tool_choice=chosen_tool,
        messages=tool_messages,
    )

    msg = resp.choices[0].message
    tool_calls = msg.tool_calls or []
    for call in tool_calls:
        name = call.function.name
        args = json.loads(call.function.arguments or "{}")
        if name == "calculate_emi":
            emi = calculate_emi(
                principal=float(args.get("principal", entities.get("loan_amount_requested") or 0)),
                annual_rate=float(args.get("annual_rate", 9.2)),
                years=int(args.get("years", tenure_years)),
            )
            return True, {"tool": "calculate_emi", "emi": emi, "years": int(args.get("years", tenure_years))}
        if name == "check_eligibility":
            result = check_eligibility(
                monthly_income=float(args.get("monthly_income", entities.get("monthly_income") or 0)),
                loan_amount=float(args.get("loan_amount", entities.get("loan_amount_requested") or 0)),
                property_value=float(args.get("property_value", entities.get("property_value") or 0)),
                employment_status=str(args.get("employment_status", entities.get("employment_status") or "unknown")),
                annual_rate=float(args.get("annual_rate", 9.2)),
                years=int(args.get("years", tenure_years)),
            )
            return True, {"tool": "check_eligibility", **result.to_dict()}

    # Deterministic fallback.
    if enough_for_eligibility:
        result = check_eligibility(
            monthly_income=float(entities.get("monthly_income") or 0),
            loan_amount=float(entities.get("loan_amount_requested") or 0),
            property_value=float(entities.get("property_value") or 0),
            employment_status=str(entities.get("employment_status") or "unknown"),
            years=tenure_years,
        )
        return True, {"tool": "check_eligibility_fallback", **result.to_dict()}

    emi = calculate_emi(
        principal=float(entities.get("loan_amount_requested") or 0),
        years=tenure_years,
    )
    return True, {"tool": "calculate_emi_fallback", "emi": emi, "years": tenure_years}


def _render_transcript(messages: list[dict[str, str]]) -> None:
    st.markdown("### Conversation transcript")
    for m in messages:
        role = "User" if m["role"] == "user" else "Assistant"
        st.write(f"**{role}:** {m['content']}")


def main() -> None:
    st.title("Vernacular Loan Counselor")
    st.caption("Sarvam STT/TTS + Groq LLaMA + deterministic loan tools + FAQ retrieval")
    require_env()

    if "locked_language" not in st.session_state:
        st.session_state.locked_language = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "faq_docs" not in st.session_state:
        st.session_state.faq_docs = fetch_homefirst_faqs(limit=30)
    if "lc_runtime" not in st.session_state:
        st.session_state.lc_runtime = build_langchain_faq_runtime(groq_api_key=GROQ_KEY, model_name=LLM_MODEL)
    if "last_debug" not in st.session_state:
        st.session_state.last_debug = {}
    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if "active_flow" not in st.session_state:
        st.session_state.active_flow = None
    if "entity_memory" not in st.session_state:
        st.session_state.entity_memory = {
            "monthly_income": None,
            "annual_income": None,
            "property_value": None,
            "loan_amount_requested": None,
            "employment_status": "unknown",
            "tenure_years": None,
            "interest_rate_percent": None,
            "intent": "unknown",
        }

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Record via mic**")
        recorded_bytes = get_audio_bytes()
    with col2:
        st.markdown("**Or upload audio file**")
        uploaded_file = st.file_uploader("Upload audio", type=["wav", "mp3", "m4a", "ogg", "flac"])

    source_bytes = None
    filename = "mic.wav"
    if recorded_bytes:
        source_bytes = recorded_bytes
    elif uploaded_file:
        source_bytes = uploaded_file.read()
        filename = uploaded_file.name

    forced_lang = st.selectbox("Force language (optional)", ["Auto-detect", "en-IN", "hi-IN", "mr-IN", "ta-IN"], index=0)

    if st.button("Process", disabled=source_bytes is None):
        try:
            # Language detection logic:
            # 1. If user explicitly selects a language (not Auto-detect), lock to that.
            # 2. If user keeps Auto-detect, lock on the first utterance and keep it for session.
            if forced_lang != "Auto-detect":
                st.session_state.locked_language = forced_lang

            lang_hint = st.session_state.locked_language
            transcript = transcribe_audio(source_bytes, filename, language_code=lang_hint)

            client = Groq(api_key=GROQ_KEY)

            # Auto-detect only once per session; then stay locked.
            if forced_lang == "Auto-detect" and st.session_state.locked_language is None:
                st.session_state.locked_language = detect_language_safe(transcript, client)

            locked_lang = st.session_state.locked_language
            st.success(f"Language locked: {locked_lang}")
            st.write("**Transcript:**", transcript)

            current_entities = extract_entities(client, transcript)
            entities = _merge_entities(st.session_state.entity_memory, current_entities)
            st.session_state.entity_memory = entities
            faq_context = retrieve_faq_context(transcript, st.session_state.faq_docs, top_k=3)
            out_of_domain = _is_out_of_domain(transcript)
            selected_flow = _choose_user_flow(transcript, entities)

            explicit_flow = _explicit_requested_flow(transcript)
            if explicit_flow:
                st.session_state.active_flow = explicit_flow
            elif st.session_state.active_flow is None and selected_flow != "general":
                st.session_state.active_flow = selected_flow
            requested_flow = st.session_state.active_flow or selected_flow

            tool_called, tool_result = maybe_run_tools(client, entities, requested_flow=requested_flow)

            tool_name = str((tool_result or {}).get("tool") or "")
            # Out-of-scope guard applies only in general flow, not in active eligibility/EMI journeys.
            if requested_flow == "general" and _is_irrelevant_or_nonsense(transcript):
                rag_mode = "deterministic_out_of_scope"
                reply = _build_out_of_scope_reply(locked_lang)
            elif tool_name.startswith("check_eligibility"):
                rag_mode = "deterministic_tool_summary"
                reply = build_compact_eligibility_reply(entities, tool_result or {})
            elif tool_name.startswith("calculate_emi"):
                rag_mode = "deterministic_emi_summary"
                reply = build_compact_emi_reply(entities, tool_result or {})
            elif requested_flow == "eligibility":
                missing_fields = _missing_eligibility_fields(entities)
                if missing_fields:
                    rag_mode = "deterministic_missing_fields"
                    reply = _build_missing_fields_reply(missing_fields, locked_lang)
                else:
                    rag_mode = "direct"
                    reply = _build_assistant_reply(
                        client=client,
                        user_text=transcript,
                        locked_language=locked_lang,
                        faq_context=faq_context,
                        tool_result=tool_result,
                        out_of_domain=out_of_domain,
                        conversation_history=st.session_state.messages,
                    )
            elif requested_flow == "emi":
                missing_emi_fields = _missing_emi_fields(entities)
                if missing_emi_fields:
                    rag_mode = "deterministic_missing_emi_fields"
                    reply = _build_missing_emi_fields_reply(missing_emi_fields)
                else:
                    rag_mode = "direct"
                    reply = _build_assistant_reply(
                        client=client,
                        user_text=transcript,
                        locked_language=locked_lang,
                        faq_context=faq_context,
                        tool_result=tool_result,
                        out_of_domain=out_of_domain,
                        conversation_history=st.session_state.messages,
                    )
            elif requested_flow == "general" and len(st.session_state.messages) == 0:
                rag_mode = "guided_welcome"
                reply = _build_welcome_menu_reply()
            elif _is_faq_query(transcript):
                rag_mode = "langchain"
                try:
                    reply = run_langchain_faq_agent(
                        runtime=st.session_state.lc_runtime,
                        user_query=transcript,
                        locked_language=locked_lang,
                        out_of_domain=out_of_domain,
                        tool_result=tool_result,
                    )
                    if _looks_uncertain_answer(reply) and faq_context:
                        rag_mode = "langchain_uncertain_fallback"
                        reply = _build_assistant_reply(
                            client=client,
                            user_text=transcript,
                            locked_language=locked_lang,
                            faq_context=faq_context,
                            tool_result=tool_result,
                            out_of_domain=out_of_domain,
                            conversation_history=st.session_state.messages,
                        )
                except Exception:
                    rag_mode = "fallback"
                    reply = _build_assistant_reply(
                        client=client,
                        user_text=transcript,
                        locked_language=locked_lang,
                        faq_context=faq_context,
                        tool_result=tool_result,
                        out_of_domain=out_of_domain,
                        conversation_history=st.session_state.messages,
                    )
            else:
                rag_mode = "direct"
                reply = _build_assistant_reply(
                    client=client,
                    user_text=transcript,
                    locked_language=locked_lang,
                    faq_context=faq_context,
                    tool_result=tool_result,
                    out_of_domain=out_of_domain,
                    conversation_history=st.session_state.messages,
                )

            reply = finalize_reply(client, reply, locked_lang)

            handoff_triggered = False
            handoff_status = "NOT_EVALUATED"
            if tool_result and tool_result.get("eligible") and _is_high_intent(transcript, entities):
                handoff_triggered = True
                handoff_status = _maybe_log_handoff(entities, tool_result)
                st.warning("[HANDOFF TRIGGERED: Routing to Human RM]")

            st.write("**LLM Reply:**", reply)
            tts_audio, tts_format, tts_text = speak_text(reply, locked_lang)
            render_audio_player(tts_audio, tts_format, autoplay=True)

            st.session_state.messages.append({"role": "user", "content": transcript})
            st.session_state.messages.append({"role": "assistant", "content": reply})

            conversation_log_status = _maybe_log_conversation(
                session_id=st.session_state.session_id,
                user_text=transcript,
                assistant_text=reply,
                locked_language=locked_lang,
                entities=entities,
                tool_result=tool_result,
                rag_mode=rag_mode,
            )

            st.session_state.last_debug = {
                "session_id": st.session_state.session_id,
                "locked_language": locked_lang,
                "extracted_json": entities,
                "tool_called": tool_called,
                "tool_result": tool_result,
                "out_of_domain": out_of_domain,
                "retrieved_faqs": [{"q": d.question, "a": d.answer[:120]} for d in faq_context],
                "rag_mode": rag_mode,
                "langchain_init_error": getattr(st.session_state.lc_runtime, "init_error", None),
                "handoff_triggered": handoff_triggered,
                "handoff_log_status": handoff_status,
                "conversation_log_status": conversation_log_status,
                "tts_input_chars": len(tts_text),
                "faq_corpus_size": len(st.session_state.faq_docs),
            }

        except Exception as exc:
            st.error(f"Error: {exc}")

    if st.session_state.messages:
        _render_transcript(st.session_state.messages)

    with st.expander("Debug panel", expanded=True):
        st.json(st.session_state.last_debug)

    c1, c2 = st.columns(2)
    with c1:
        if st.button("Reset language lock"):
            st.session_state.locked_language = None
            st.info("Language lock cleared.")
    with c2:
        if st.button("Clear conversation"):
            st.session_state.messages = []
            st.session_state.last_debug = {}
            st.session_state.active_flow = None
            st.session_state.entity_memory = {
                "monthly_income": None,
                "annual_income": None,
                "property_value": None,
                "loan_amount_requested": None,
                "employment_status": "unknown",
                "tenure_years": None,
                "interest_rate_percent": None,
                "intent": "unknown",
            }
            st.info("Conversation cleared.")


if __name__ == "__main__":
    main()
