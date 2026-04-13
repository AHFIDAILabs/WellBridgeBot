# modules/intent_classifier.py: Hybrid intent classifier for WellBridge chatbot
# Uses keyword pre-filter (free) + LLM classifier (1 call) for uncertain cases

import logging
import re
from typing import Tuple

logger = logging.getLogger(__name__)


# ------------------- Intent Types -------------------

INTENT_CASUAL = "casual"
INTENT_HEALTH = "health"
INTENT_OFF_TOPIC = "off_topic"
INTENT_UNCERTAIN = "uncertain"


# ------------------- Keyword Dictionaries -------------------

# Casual / greeting patterns across all supported languages
CASUAL_KEYWORDS = {
    # English
    "hello", "hi", "hey", "good morning", "good afternoon", "good evening",
    "how are you", "what's up", "whats up", "sup", "howdy", "thanks",
    "thank you", "bye", "goodbye", "see you", "who are you", "what is your name",
    "what's your name", "your name", "how are u", "what can you do",
    "help me", "good night", "welcome",

    # Yoruba
    "bawo ni", "e kaaro", "e kaasan", "e kaaale", "e ku irole",
    "o dabo", "e se", "e ṣe", "ṣé o wa", "oruko mi ni",
    "ki ni oruko re", "pẹlẹ o", "e nlẹ", "e pẹlẹ",

    # Igbo
    "kedu", "nnọọ", "dalụ", "ka ọ dị", "i meela", "ka emesia",
    "kedụ ka ị mere", "gịnị bụ aha gị", "aha m bụ",

    # Hausa
    "sannu", "barka dai", "barka da safe", "barka da rana",
    "yaya dai", "ina kwana", "na gode", "sai anjima",
    "me sunanka", "sunana", "yaya kake",

    # Pidgin
    "how far", "how you dey", "wetin dey happen", "na wa",
    "abeg", "no wahala", "e don do", "i dey fine", "how body",
    "wetin be your name", "who you be", "thank you o", "how e dey go",
}

# Health/medical keywords (expanded from config.py HEALTH_KEYWORDS)
HEALTH_KEYWORDS = {
    # English - TB specific
    "tuberculosis", "tb", "mycobacterium", "pulmonary", "latent tb",
    "active tb", "xdr-tb", "mdr-tb", "dots", "mantoux", "bcg vaccine",
    # English - general health
    "disease", "infection", "treatment", "prevention", "symptoms", "cure",
    "medicine", "doctor", "hospital", "clinic", "diagnosis", "vaccine",
    "cough", "coughing", "fever", "weight loss", "night sweats",
    "chest pain", "blood", "sputum", "lungs", "breathing", "health",
    "medical", "patient", "medication", "antibiotic", "x-ray", "xray",
    "test", "testing", "hiv", "aids", "immune", "contagious", "spread",
    "transmission", "quarantine", "isolation", "wellness", "healthcare",
    "pharmacy", "drug", "dosage", "side effect", "resistant",

    # Yoruba
    "àrùn", "ìwòsàn", "àìsàn", "ọgbẹ́ni", "ẹ̀jẹ̀", "ilera",
    "ikọ", "ikọ́", "ogun", "oògùn", "àìlera", "iba",
    "ẹ̀dọ̀fóró", "ara", "iwosan",

    # Igbo
    "ọrịa", "ahụike", "ọgwụ", "dọkịta", "ụkwara", "ụlọọgwụ",
    "nsogbu ahụ", "ike gwụrụ", "ara ọkụ",

    # Hausa
    "cuta", "lafiya", "magani", "likita", "tari", "asibiti",
    "jinya", "matsala", "jiki", "zazzabi",

    # Pidgin
    "sickness", "sick", "well", "body no well", "coff", "koff",
    "doctor", "medicine man", "pharmacy", "injection", "tablet",
}

# Off-topic indicators
OFF_TOPIC_KEYWORDS = {
    # Politics
    "election", "president", "governor", "politician", "vote", "party",
    "campaign", "democracy", "senate", "politics", "buhari", "tinubu",

    # Sports
    "football", "soccer", "match", "goal", "premier league", "champions league",
    "world cup", "player", "team", "score", "basketball", "olympics",

    # Entertainment
    "movie", "nollywood", "music", "artist", "song", "album",
    "netflix", "streaming", "celebrity", "actor", "actress",

    # Finance/business
    "bitcoin", "crypto", "stock", "forex", "naira", "dollar",
    "business", "investment", "trading",

    # Tech (non-health)
    "iphone", "android", "laptop", "programming", "code", "software",
    "instagram", "tiktok", "facebook", "whatsapp group",

    # Food/cooking (non-health context)
    "recipe", "jollof", "suya", "amala", "fufu", "egusi",

    # Religion
    "church", "mosque", "pastor", "imam", "prayer", "bible", "quran",

    # General
    "weather", "travel", "school", "university", "exam", "wedding",
    "salary", "job", "work", "rent",
}


# ------------------- Keyword Pre-Filter -------------------

def _normalize(text: str) -> str:
    """Lowercase and strip punctuation for keyword matching."""
    return re.sub(r"[^\w\sàáèéìíòóùúẹọṣụñ]", "", text.lower()).strip()


def _keyword_match(text: str, keyword_set: set) -> int:
    """Count how many keywords from the set appear in the text.
    Uses word-boundary matching for short keywords (<=3 chars) to avoid
    false positives like 'tb' inside 'football'.
    """
    normalized = _normalize(text)
    count = 0
    for kw in keyword_set:
        if len(kw) <= 3:
            # Short keywords need word-boundary matching
            if re.search(r'\b' + re.escape(kw) + r'\b', normalized):
                count += 1
        else:
            # Longer keywords: substring is fine
            if kw in normalized:
                count += 1
    return count


def keyword_classify(query: str) -> str:
    """Fast, free keyword-based classification.

    Returns one of: INTENT_CASUAL, INTENT_HEALTH, INTENT_OFF_TOPIC, INTENT_UNCERTAIN
    """
    casual_hits = _keyword_match(query, CASUAL_KEYWORDS)
    health_hits = _keyword_match(query, HEALTH_KEYWORDS)
    off_topic_hits = _keyword_match(query, OFF_TOPIC_KEYWORDS)

    logger.debug(
        f"Keyword hits — casual: {casual_hits}, health: {health_hits}, off_topic: {off_topic_hits}"
    )

    # Short messages (< 5 words) that match casual patterns are almost certainly casual
    word_count = len(query.split())
    if casual_hits > 0 and health_hits == 0 and word_count <= 6:
        return INTENT_CASUAL

    # Clear health signal
    if health_hits >= 2 and health_hits > off_topic_hits:
        return INTENT_HEALTH
    if health_hits >= 1 and casual_hits == 0 and off_topic_hits == 0:
        return INTENT_HEALTH

    # Clear off-topic signal (even 1 hit with no health signal)
    if off_topic_hits >= 1 and health_hits == 0:
        return INTENT_OFF_TOPIC

    # Mixed signals or no signal → uncertain
    return INTENT_UNCERTAIN


# ------------------- LLM Classifier (1 call) -------------------

LLM_CLASSIFY_PROMPT = """You are the intent classifier for WellBridge, a Nigerian health chatbot focused on tuberculosis (TB) and general health.

Classify the user's message into EXACTLY one category:
- CASUAL: Greetings, small talk, asking about the bot, saying thanks, goodbye, etc.
- HEALTH: Any health, medical, wellness, TB, disease, symptoms, treatment question or concern.
- OFF_TOPIC: Politics, sports, entertainment, finance, tech, or anything clearly unrelated to health or conversation.

User message: "{query}"

Reply with ONLY one word: CASUAL, HEALTH, or OFF_TOPIC"""


def llm_classify(query: str, invoke_fn) -> str:
    """Use a single LLM call to classify intent when keywords are uncertain.

    Args:
        query: The user's message
        invoke_fn: A callable(prompt: str) -> str  (e.g., invoke_llm_with_fallback)
    """
    try:
        prompt = LLM_CLASSIFY_PROMPT.format(query=query)
        result = invoke_fn(prompt).strip().upper()

        # Parse the LLM response
        if "CASUAL" in result:
            return INTENT_CASUAL
        elif "HEALTH" in result:
            return INTENT_HEALTH
        elif "OFF_TOPIC" in result or "OFF-TOPIC" in result:
            return INTENT_OFF_TOPIC
        else:
            # If LLM response is unexpected, default to health (safe fallback)
            logger.warning(f"LLM classifier returned unexpected: '{result}', defaulting to HEALTH")
            return INTENT_HEALTH

    except Exception as e:
        logger.error(f"LLM classification failed: {e}, defaulting to HEALTH")
        return INTENT_HEALTH  # Safe fallback — goes through KB pipeline


# ------------------- Main Classifier -------------------

def classify_intent(query: str, invoke_fn=None) -> str:
    """Hybrid intent classifier: keyword pre-filter → LLM for uncertain cases.

    Args:
        query: The user's raw message (any language)
        invoke_fn: Optional callable for LLM classification (invoke_llm_with_fallback).
                   If None, uncertain queries default to HEALTH.

    Returns:
        One of: INTENT_CASUAL, INTENT_HEALTH, INTENT_OFF_TOPIC
    """
    # Step 1: Free keyword check
    intent = keyword_classify(query)
    logger.info(f"Keyword classifier → {intent} for: '{query[:60]}...'")

    if intent != INTENT_UNCERTAIN:
        return intent

    # Step 2: Uncertain — use LLM (1 call) if available
    if invoke_fn is not None:
        intent = llm_classify(query, invoke_fn)
        logger.info(f"LLM classifier → {intent} for: '{query[:60]}...'")
        return intent

    # No LLM available — default to health (safe fallback)
    logger.info("No LLM invoke function provided, defaulting to HEALTH")
    return INTENT_HEALTH
