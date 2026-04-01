"""Vernacular Loan Counselor: Sarvam STT/TTS + Groq LLM + FAQ retrieval + deterministic tools."""

from __future__ import annotations

import base64
import json
import os
import re
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

SYSTEM_PROMPT = """
You are HomeFirst Vernacular Loan Counselor.
Rules:
1) Always respond in the locked language: {locked_language}.
2) Home-loan only. If user asks unrelated topics (personal loan, car loan, jokes, politics), politely redirect to home-loan counseling.
3) Never do eligibility math directly. Use tools.
4) Use FAQ context when user asks policy/document/process/eligibility questions.
5) Be concise and practical.
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
        # Hindi and Marathi both use Devanagari; keep Hindi default for TTS compatibility.
        if any(word in text_l for word in ["majha", "ahe", "tumhi", "marathi"]):
            return "mr-IN"
        return "hi-IN"
    return "en-IN"


def transcribe_audio(file_bytes: bytes, filename: str, language_code: str = "hi-IN") -> str:
    files = {"file": (filename, file_bytes, "audio/wav")}
    data = {"model": "saarika:v2.5", "language_code": language_code}
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


def _prepare_tts_text(text: str, limit: int = TTS_MAX_CHARS) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 3].rstrip() + "..."


def speak_text(text: str, target_language: str) -> tuple[bytes, str, str]:
    tts_text = _prepare_tts_text(text)
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


def extract_entities(client: Groq, user_text: str) -> dict[str, Any]:
    prompt = (
        "Extract entities from user input. Return strict JSON only with keys: "
        "monthly_income, property_value, loan_amount_requested, employment_status, intent. "
        "Use null when unknown. employment_status should be salaried/self-employed/unknown."
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

    normalized = {
        "monthly_income": data.get("monthly_income"),
        "property_value": data.get("property_value"),
        "loan_amount_requested": data.get("loan_amount_requested"),
        "employment_status": data.get("employment_status") or "unknown",
        "intent": data.get("intent") or "unknown",
    }
    return normalized


def _is_out_of_domain(user_text: str) -> bool:
    text_l = user_text.lower()
    blocked = ["personal loan", "car loan", "credit card", "stocks", "crypto", "politics", "joke"]
    return any(x in text_l for x in blocked)


def _is_faq_query(user_text: str) -> bool:
    text_l = user_text.lower()
    faq_keys = ["document", "documents", "eligibility", "co-applicant", "prepayment", "tenure", "interest", "repay", "tax"]
    return any(x in text_l for x in faq_keys)


def _is_high_intent(user_text: str, entities: dict[str, Any]) -> bool:
    text_l = user_text.lower()
    intent_signal = any(x in text_l for x in ["apply", "process", "start", "proceed", "next step", "loan chahiye", "apply now"])
    complete_financials = all(
        entities.get(k) not in (None, "", 0) for k in ["monthly_income", "property_value", "loan_amount_requested"]
    ) and (entities.get("employment_status") or "unknown") != "unknown"
    return intent_signal and complete_financials


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
        {"role": "user", "content": user_text},
    ]

    if tool_result:
        messages.append({"role": "system", "content": "Deterministic tool output (source of truth): " + json.dumps(tool_result)})

    out = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0.3,
        max_completion_tokens=500,
        messages=messages,
    )
    return out.choices[0].message.content or "Please share a bit more detail about your home loan requirement."


def maybe_run_tools(client: Groq, entities: dict[str, Any]) -> tuple[bool, Optional[dict[str, Any]]]:
    enough_for_eligibility = all(
        entities.get(k) not in (None, "", 0) for k in ["monthly_income", "property_value", "loan_amount_requested"]
    ) and entities.get("employment_status") not in (None, "", "unknown")

    if not enough_for_eligibility:
        return False, None

    # Ask model to pick function call; fallback to deterministic direct call.
    tool_messages = [
        {
            "role": "system",
            "content": "If all fields exist, call check_eligibility. If only principal present, call calculate_emi.",
        },
        {
            "role": "user",
            "content": json.dumps(
                {
                    "monthly_income": entities.get("monthly_income"),
                    "loan_amount": entities.get("loan_amount_requested"),
                    "property_value": entities.get("property_value"),
                    "employment_status": entities.get("employment_status"),
                }
            ),
        },
    ]

    resp = client.chat.completions.create(
        model=LLM_MODEL,
        temperature=0,
        max_completion_tokens=300,
        tools=TOOL_SCHEMAS,
        tool_choice="auto",
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
                years=int(args.get("years", 20)),
            )
            return True, {"tool": "calculate_emi", "emi": emi}
        if name == "check_eligibility":
            result = check_eligibility(
                monthly_income=float(args.get("monthly_income", entities.get("monthly_income") or 0)),
                loan_amount=float(args.get("loan_amount", entities.get("loan_amount_requested") or 0)),
                property_value=float(args.get("property_value", entities.get("property_value") or 0)),
                employment_status=str(args.get("employment_status", entities.get("employment_status") or "unknown")),
                annual_rate=float(args.get("annual_rate", 9.2)),
                years=int(args.get("years", 20)),
            )
            return True, {"tool": "check_eligibility", **result.to_dict()}

    # Deterministic fallback.
    result = check_eligibility(
        monthly_income=float(entities.get("monthly_income") or 0),
        loan_amount=float(entities.get("loan_amount_requested") or 0),
        property_value=float(entities.get("property_value") or 0),
        employment_status=str(entities.get("employment_status") or "unknown"),
    )
    return True, {"tool": "check_eligibility_fallback", **result.to_dict()}


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
            lang_hint = st.session_state.locked_language or (None if forced_lang == "Auto-detect" else forced_lang)
            transcript = transcribe_audio(source_bytes, filename, language_code=lang_hint or "hi-IN")

            if not st.session_state.locked_language:
                st.session_state.locked_language = forced_lang if forced_lang != "Auto-detect" else detect_language(transcript)

            locked_lang = st.session_state.locked_language
            st.success(f"Language locked: {locked_lang}")
            st.write("**Transcript:**", transcript)

            client = Groq(api_key=GROQ_KEY)
            entities = extract_entities(client, transcript)
            faq_context = retrieve_faq_context(transcript, st.session_state.faq_docs, top_k=3)
            out_of_domain = _is_out_of_domain(transcript)
            tool_called, tool_result = maybe_run_tools(client, entities)

            rag_mode = "langchain"
            try:
                reply = run_langchain_faq_agent(
                    runtime=st.session_state.lc_runtime,
                    user_query=transcript,
                    locked_language=locked_lang,
                    out_of_domain=out_of_domain,
                    tool_result=tool_result,
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
                )

            handoff_triggered = False
            handoff_status = "NOT_EVALUATED"
            if tool_result and tool_result.get("eligible") and _is_high_intent(transcript, entities):
                handoff_triggered = True
                handoff_status = _maybe_log_handoff(entities, tool_result)
                st.warning("[HANDOFF TRIGGERED: Routing to Human RM]")

            st.write("**LLM Reply:**", reply)
            tts_audio, tts_format, tts_text = speak_text(reply, locked_lang)
            st.audio(BytesIO(tts_audio), format=tts_format)

            st.session_state.messages.append({"role": "user", "content": transcript})
            st.session_state.messages.append({"role": "assistant", "content": reply})

            st.session_state.last_debug = {
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
            st.info("Conversation cleared.")


if __name__ == "__main__":
    main()
