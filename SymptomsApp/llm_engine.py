"""
SymptoTrackAI Generative LLM Dialogue Engine & Clinical Triage Manager.
Provides non-rule-based conversational understanding, multi-turn clinical follow-up questioning,
selective symptom diagnosis, severity classification, and multilingual capabilities.
"""

import os
import re
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from .medical_kb import (
    DISEASE_KB,
    check_red_flags,
    get_follow_up_questions,
    detect_language,
    detect_intent,
    TRANSLATIONS
)

# Global LLM instance
_LLM_TOKENIZER = None
_LLM_MODEL = None

def get_llm():
    """Lazily load and cache the local FLAN-T5 LLM model."""
    global _LLM_TOKENIZER, _LLM_MODEL
    if _LLM_TOKENIZER is None or _LLM_MODEL is None:
        model_name = "google/flan-t5-small"
        try:
            print(f"Loading Generative LLM: {model_name}...")
            _LLM_TOKENIZER = AutoTokenizer.from_pretrained(model_name)
            _LLM_MODEL = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            print(f"Generative LLM {model_name} loaded successfully.")
        except Exception as e:
            print(f"Error loading {model_name}: {e}")
            _LLM_TOKENIZER = None
            _LLM_MODEL = None
    return _LLM_TOKENIZER, _LLM_MODEL


# Session memory dictionary: session_id -> { "history": [...], "symptoms": [...], "turn": int }
SESSION_MEMORY = {}

def get_session_data(session_id):
    if session_id not in SESSION_MEMORY:
        SESSION_MEMORY[session_id] = {
            "history": [],
            "accumulated_symptoms": [],
            "last_turn_type": "none",
            "turn_count": 0
        }
    return SESSION_MEMORY[session_id]


class SymptoTrackLLM:
    """
    LLM-powered healthcare dialogue assistant.
    Understands conversational context, asks clinical follow-up questions,
    and selectively diagnoses only when appropriate.
    """

    @staticmethod
    def generate_llm_text(prompt, max_length=120):
        """Generates text from the local FLAN-T5 LLM model."""
        tokenizer, model = get_llm()
        if tokenizer is None or model is None:
            return None
        try:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
            outputs = model.generate(**inputs, max_new_tokens=max_length, do_sample=True, temperature=0.7, top_p=0.9)
            return tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        except Exception as e:
            print(f"LLM Generation error: {e}")
            return None

    @classmethod
    def process_message(cls, message, session_id="default", lang=None, faiss_index=None, Y=None, question_encoder_fn=None):
        """
        Main dialogue processing pipeline:
        1. Multi-language detection
        2. Emergency Red-Flag detection
        3. Semantic Dialogue Classification (Greeting, Bot inquiry, Preliminary symptom, Follow-up answer, Comprehensive symptoms)
        4. Contextual Response or Clinical Assessment
        """
        message_clean = message.strip()
        if not message_clean:
            return "Please describe your symptoms or ask a health question.", None

        # Detect language
        if not lang or lang not in TRANSLATIONS:
            lang = detect_language(message_clean)
        t = TRANSLATIONS.get(lang, TRANSLATIONS["en"])

        session = get_session_data(session_id)
        session["turn_count"] += 1
        session["history"].append({"user": message_clean})

        # 1. CRITICAL EMERGENCY RED-FLAG CHECK
        if check_red_flags(message_clean):
            emergency_resp = (
                f"⚠️ {t['emergency_title']} ⚠️\n"
                f"────────────────────────────────────────\n"
                f"{t['emergency_msg']}\n\n"
                f"Please do not wait or rely on online tools. Seek immediate emergency medical care (call 112 / 911 / 108) or visit the nearest emergency room."
            )
            session["last_turn_type"] = "emergency"
            return emergency_resp, "EMERGENCY"

        # 2. SEMANTIC INTENT CLASSIFICATION
        intent = detect_intent(message_clean)

        # GREETING
        if intent == "GREETING":
            session["last_turn_type"] = "greeting"
            if lang == "hi":
                greeting_msg = "नमस्ते! 👋 मैं SymptoTrackAI हूँ, आपका AI स्वास्थ्य एवं लक्षण निगरानी सहायक। आप आज कैसा महसूस कर रहे हैं?\n\nकृपया अपने लक्षणों का वर्णन करें (उदाहरण के लिए: 'मुझे 2 दिनों से तेज बुखार है', 'सिर में तेज दर्द है', या 'खांसी और सीने में जकड़न')।"
            elif lang == "te":
                greeting_msg = "నమస్కారం! 👋 నేను SymptoTrackAI, మీ ఆరోగ్య మరియు లక్షణాల పర్యవేక్షణ AI సహాయకుడిని. ఈరోజు మీ ఆరోగ్యం ఎలా ఉంది?\n\nదయచేసి మీ లక్షణాలను వివరించండి (ఉదాహరణకు: 'నాకు జ్వరం మరియు తలనొప్పి ఉంది', 'దగ్గు మరియు ఛాతీలో అసౌకర్యం')."
            elif lang == "es":
                greeting_msg = "¡Hola! 👋 Soy SymptoTrackAI, tu asistente de salud y monitoreo de síntomas. ¿Cómo te sientes hoy?\n\nPor favor describe los síntomas que estás experimentando (ej. 'tengo fiebre alta', 'dolor de cabeza pulsátil', o 'tos persistente')."
            else:
                greeting_msg = "Hello! 👋 I am SymptoTrackAI, your intelligent health and symptom monitoring assistant. How are you feeling today?\n\nPlease describe any symptoms you are experiencing (for example: 'I have a fever with chills', 'severe throbbing headache', or 'dry cough and chest tightness')."
            return greeting_msg, None

        # ABOUT BOT / CAPABILITIES
        if intent == "BOT_INFO":
            session["last_turn_type"] = "bot_info"
            return t["bot_info"], None

        # GRATITUDE
        if intent == "GRATITUDE":
            session["last_turn_type"] = "gratitude"
            return t["gratitude"], None

        # FAREWELL
        if intent == "FAREWELL":
            session["last_turn_type"] = "farewell"
            return t["farewell"], None

        lower_msg = message_clean.lower()
        medical_terms = [
            "fever", "cough", "headache", "pain", "ache", "rash", "itching", "itch",
            "swell", "swelling", "throat", "vomit", "vomiting", "nausea", "dizzy",
            "dizziness", "fatigue", "tired", "weak", "weakness", "breath", "breathing",
            "chill", "chills", "burn", "burning", "chest", "stomach", "abdomen",
            "joint", "back", "peeling", "lesion", "blister", "ulcer", "diarrhea",
            "shivering", "sneezing", "congestion", "phlegm", "cold", "jaundice", "yellow",
            "migraine", "asthma", "wheezing", "heartburn", "acid", "reflux", "bp",
            "pressure", "infection", "bukhar", "dard", "khansi", "jvaram", "noppi", "daggu"
        ]

        detected_terms = [term for term in medical_terms if term in lower_msg]

        # Non-medical general chat or gibberish
        if not detected_terms and len(lower_msg.split()) < 3 and session["last_turn_type"] != "waiting_followup":
            return t["unclear"], None

        # Add detected terms to accumulated session symptoms
        for term in detected_terms:
            if term not in session["accumulated_symptoms"]:
                session["accumulated_symptoms"].append(term)

        # CHECK IF PRELIMINARY SYMPTOM (e.g. user just said "I have a headache" or "I have a cough")
        # In a real clinical consultation, a doctor does NOT give a final diagnosis on a 2-word symptom.
        # They ask follow-up questions first!
        is_brief_symptom = len(detected_terms) <= 2 and len(lower_msg.split()) <= 6
        is_first_symptom_turn = session["last_turn_type"] not in ["waiting_followup", "evaluating"]

        if is_brief_symptom and is_first_symptom_turn and not any(w in lower_msg for w in ["diagnose", "what disease", "what condition", "tell me what i have"]):
            # Ask targeted follow-up questions instead of prematurely diagnosing!
            session["last_turn_type"] = "waiting_followup"
            primary_term = detected_terms[0] if detected_terms else "symptoms"

            if lang == "hi":
                followup_reply = (
                    f"मैं समझ सकता हूँ कि आप {primary_term} से परेशान हैं। आपके लक्षणों को ठीक से समझने के लिए, कृपया कुछ बातें बताएं:\n\n"
                    f"1. यह लक्षण कितने दिनों से महसूस हो रहे हैं (जैसे: 1-2 दिन या एक हफ्ता)?\n"
                    f"2. क्या इसके साथ बुखार, ठंड लगना, मतली या कमजोरी भी है?\n"
                    f"3. 1 से 10 के पैमाने पर यह तकलीफ कितनी तेज है?"
                )
            elif lang == "te":
                followup_reply = (
                    f"మీరు {primary_term} తో బాధపడుతున్నట్లు గమనించాను. మీ పరిస్థితిని స్పష్టంగా అంచనా వేయడానికి, దయచేసి కొన్ని వివరాలు తెలపండి:\n\n"
                    f"1. ఈ సమస్య ఎన్ని రోజులుగా ఉంది (ఉదా: 1-2 రోజులు లేదా ఎక్కువ)?\n"
                    f"2. దీంతో పాటు జ్వరం, చలి, నీరసం లేదా ఇతర లక్షణాలు ఏమైనా ఉన్నాయా?\n"
                    f"3. నొప్పి లేదా అసౌకర్యం ఎంత తీవ్రంగా ఉంది (1 నుండి 10 స్కేలులో)?"
                )
            elif lang == "es":
                followup_reply = (
                    f"Lamento que estés experimentando {primary_term}. Para comprender mejor tu situación, por favor responde:\n\n"
                    f"1. ¿Cuántos días llevas con estos síntomas (ej. 1-2 días o más de una semana)?\n"
                    f"2. ¿Tienes también fiebre, escalofríos, náuseas o fatiga?\n"
                    f"3. En una escala del 1 al 10, ¿qué tan intenso es el malestar?"
                )
            else:
                followup_reply = (
                    f"I understand you are experiencing {primary_term}. To help provide an accurate clinical assessment, could you share a bit more detail?\n\n"
                    f"1. How long have you had these symptoms (e.g., just started, 2-3 days, or over a week)?\n"
                    f"2. Are you experiencing any accompanying symptoms like fever, chills, body aches, or nausea?\n"
                    f"3. On a scale of 1 to 10, how severe is your discomfort?"
                )
            return followup_reply, None

        # 4. COMPREHENSIVE CLINICAL ASSESSMENT (User provided detailed symptoms or answered follow-ups)
        # Perform retrieval-augmented clinical assessment
        if faiss_index is not None and Y is not None and question_encoder_fn is not None:
            # Combine current query with accumulated session symptoms for rich context
            combined_query = message_clean
            if session["accumulated_symptoms"]:
                combined_query += " " + " ".join(session["accumulated_symptoms"])

            query_vec = question_encoder_fn(combined_query)
            distances, indices = faiss_index.search(query_vec, k=1)
            pred_disease = Y[indices[0][0]]

            kb_info = DISEASE_KB.get(pred_disease, None)
            if not kb_info:
                for k in DISEASE_KB:
                    if k.lower() == pred_disease.lower():
                        kb_info = DISEASE_KB[k]
                        break

            if not kb_info:
                kb_info = {
                    "name": pred_disease,
                    "severity": "MODERATE",
                    "overview": f"Clinical condition matching reported symptoms: {pred_disease}.",
                    "self_care": ["Rest and maintain good hydration.", "Monitor symptoms closely."],
                    "when_to_see_doctor": "Consult a healthcare professional if symptoms persist.",
                    "trusted_source": "WHO Health Topics",
                    "source_url": "https://www.who.int/health-topics"
                }

            severity = kb_info["severity"]
            badge_text = t.get(severity.lower(), t["moderate"])
            follow_ups = get_follow_up_questions(pred_disease, combined_query)

            session["last_turn_type"] = "diagnosed"

            # Format structured clinical assessment
            lines = [
                f"🩺 {t['assessment_title'].upper()}",
                "────────────────────────────────────────",
                f"• {t['predicted_condition']}: {kb_info['name']}",
                f"• {t['severity_label']}: {badge_text}",
                f"\n📖 {t['overview_label']}:",
                kb_info['overview'],
                f"\n📋 {t['self_care_label']}:"
            ]
            for c in kb_info['self_care']:
                lines.append(f"  ✓ {c}")

            lines.append(f"\n⚠️ {t['warning_label']}:")
            lines.append(f"  {kb_info['when_to_see_doctor']}")

            lines.append(f"\n💬 {t['followup_label']}:")
            for i, q in enumerate(follow_ups, 1):
                lines.append(f"  {i}. {q}")

            lines.append(f"\n🌐 {t['source_label']}:")
            lines.append(f"  {kb_info['trusted_source']} ({kb_info['source_url']})")

            lines.append(f"\nℹ️ {t['disclaimer']}")

            clinical_response = "\n".join(lines)
            return clinical_response, f"{kb_info['name']} [{severity}]"

        # Fallback if vector model not loaded
        return t["unclear"], None
