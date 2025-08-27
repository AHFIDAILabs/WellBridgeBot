# modules/llm_handler.py: Multilingual LLM handler for Nigerian languages
import logging
from typing import Dict, Any, List

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.tools import DuckDuckGoSearchRun

from modules.language_utils import (
    detect_language, is_pidgin, get_language_name
)

from config import OPENROUTER_API_KEY, LLM_MODEL, MIN_SCORE

logger = logging.getLogger(__name__)


# ------------------- LLM Initialization -------------------

def get_llm():
    """Initialize the LLM with optimized settings for multilingual use."""
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set")
        raise ValueError("OPENROUTER_API_KEY environment variable is missing")

    return ChatOpenAI(
        model=LLM_MODEL,
        base_url="https://openrouter.ai/api/v1",
        api_key=OPENROUTER_API_KEY,
        temperature=0.2,
        max_tokens=600,
        timeout=35,
    )


# ------------------- Translation -------------------

def translate_to_english(text: str, source_lang: str) -> str:
    """Translate Nigerian languages to English for KB search."""
    if source_lang == "en":
        return text

    try:
        llm = get_llm()
        translation_prompts = {
            "yo": f"Translate this Yoruba health question to clear English. Preserve the meaning exactly.\n\nYoruba: {text}\n\nEnglish translation:",
            "ig": f"Translate this Igbo health question to clear English. Preserve the meaning exactly.\n\nIgbo: {text}\n\nEnglish translation:",
            "ha": f"Translate this Hausa health question to clear English. Preserve the meaning exactly.\n\nHausa: {text}\n\nEnglish translation:",
            "pidgin": f"Translate this Nigerian Pidgin health question to standard English. Preserve the meaning exactly.\n\nPidgin: {text}\n\nEnglish translation:"
        }
        prompt = translation_prompts.get(source_lang, f"Translate to English: {text}")
        response = llm.invoke([HumanMessage(content=prompt)])
        return response.content.strip()
    except Exception as e:
        logger.error(f"Translation from {source_lang} failed: {e}")
        return text


def translate_from_english(text: str, target_lang: str) -> str:
    """Translate English responses back to Nigerian languages with proper standardized language."""
    if target_lang == "en":
        return text

    try:
        llm = get_llm()
        
        # Enhanced prompts for proper, standardized language output
        response_prompts = {
            "yo": f"""Translate this English health information to STANDARD Yoruba language.
Use proper Yoruba diacritics (à, á, è, é, ì, í, ò, ó, ù, ú, ẹ, ọ, ṣ).
Make sure the Yoruba is grammatically correct and easily understandable.
Use common, everyday Yoruba words that ordinary people use.

English: {text}

Standard Yoruba translation:""",
            
            "ig": f"""Translate this English health information to STANDARD Igbo language.
Use proper Igbo orthography and diacritics where needed.
Make sure the Igbo is grammatically correct and easily understandable.
Use common, everyday Igbo words that ordinary people use.

English: {text}

Standard Igbo translation:""",
            
            "ha": f"""Translate this English health information to STANDARD Hausa language.
Use proper Hausa spelling and orthography.
Make sure the Hausa is grammatically correct and easily understandable.
Use common, everyday Hausa words that ordinary people use.

English: {text}

Standard Hausa translation:""",
            
            "pidgin": f"""Convert this English health information to STANDARD Nigerian Pidgin.
Make it conversational but clear and accurate.
Use proper Nigerian Pidgin that is widely understood across Nigeria.
Keep medical terms simple and explain them in Pidgin.

English: {text}

Nigerian Pidgin translation:"""
        }
        
        prompt = response_prompts.get(target_lang, f"Translate to {get_language_name(target_lang)}: {text}")
        response = llm.invoke([HumanMessage(content=prompt)])
        translated = response.content.strip()
        
        # Log the translation for debugging
        logger.info(f"Translation to {target_lang}: {translated[:100]}...")
        
        return translated
    except Exception as e:
        logger.error(f"Translation to {target_lang} failed: {e}")
        return text


# ------------------- KB Search -------------------

def create_multilingual_search_variations(query: str, detected_lang: str) -> List[str]:
    """Create search variations for better KB retrieval across Nigerian languages."""
    variations = [query]

    if detected_lang != "en":
        english_query = translate_to_english(query, detected_lang)
        if english_query and english_query != query:
            variations.append(english_query)

    if "tb" not in query.lower() and "tuberculosis" not in query.lower():
        variations.append(f"{query} tuberculosis")

    return list(dict.fromkeys(variations))  # deduplicate, keep order


def get_qa_chain(vector_store):
    """Create RetrievalQA chain for multilingual queries."""
    llm = get_llm()

    qa_template = """You are an expert tuberculosis (TB) health assistant.
Use the provided context to give accurate, clear, and medically correct answers.
Provide specific, actionable information when possible.

If the context does not contain enough information, say so clearly.

Context: {context}
Question: {question}

Detailed Answer:"""

    QA_PROMPT = PromptTemplate(
        template=qa_template,
        input_variables=["context", "question"]
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )


def calculate_result_quality_score(result_text: str, sources: List, query: str) -> float:
    """Score KB search results for relevance and completeness."""
    score = 0.0

    if not result_text or len(result_text.strip()) < 10:
        return 0.0

    # Length scoring
    text_length = len(result_text.strip())
    if 50 <= text_length <= 400:
        score += 3.0
    elif 20 <= text_length < 50:
        score += 1.5
    elif text_length > 400:
        score += 2.0

    # Source docs
    if sources:
        score += min(len(sources), 3)

    # Negative indicators
    negatives = ["don't know", "not sure", "cannot answer", "insufficient information"]
    if any(ind in result_text.lower() for ind in negatives):
        score -= 5.0
    else:
        score += 2.0

    # TB relevance
    tb_keywords = ["tuberculosis", "TB", "infection", "treatment", "symptoms", "prevention", "lungs"]
    tb_matches = sum(1 for kw in tb_keywords if kw.lower() in result_text.lower())
    score += min(tb_matches * 0.5, 3.0)

    # Query overlap
    query_words = set(query.lower().split())
    result_words = set(result_text.lower().split())
    overlap = query_words.intersection(result_words)
    if query_words:
        score += (len(overlap) / len(query_words)) * 2.0

    return score


def search_kb_with_multiple_strategies(query_variations: List[str], vector_store) -> Dict[str, Any]:
    """Try multiple KB search strategies and return the best scored result."""
    best_result = None
    best_score = 0

    for variation in query_variations:
        try:
            qa_chain = get_qa_chain(vector_store)
            response = qa_chain.invoke({"query": variation})
            result_text = response.get("result", "")
            sources = response.get("source_documents", [])

            score = calculate_result_quality_score(result_text, sources, variation)
            logger.info(f"KB search score for '{variation}': {score:.2f}")

            if score > best_score:
                best_score = score
                best_result = {
                    "result": result_text,
                    "source_documents": sources,
                    "query_used": variation,
                    "score": score
                }
        except Exception as e:
            logger.warning(f"KB search failed for '{variation}': {e}")
            continue

    return best_result


# ------------------- Web Search Fallback -------------------

def web_search_fallback(query: str, target_lang: str) -> str:
    """Fallback using DuckDuckGo search for TB info."""
    try:
        search = DuckDuckGoSearchRun()
        search_query = f"{query} tuberculosis site:who.int OR site:cdc.gov OR site:nhs.uk"
        results = search.run(search_query)

        if not results:
            return "No relevant information found online."

        llm = get_llm()
        summary_prompt = f"""
        Summarize the following search results into accurate, clear health information
        about tuberculosis (TB). Keep the facts correct and concise.
        Do not invent data. Aim for 3-5 sentences.

        Results:
        {results}

        Summary:"""
        response = llm.invoke([HumanMessage(content=summary_prompt)])
        summary = response.content.strip()

        if target_lang != "en":
            summary = translate_from_english(summary, target_lang)

        return summary
    except Exception as e:
        logger.error(f"Web search fallback failed: {e}")
        return "Unable to fetch updated information at this time."


# ------------------- Main Response -------------------

def get_response(query: str, vector_store) -> Dict[str, Any]:
    """Main function to get multilingual responses."""
    try:
        # Detect the language of the query
        detected_lang = detect_language(query)
        
        # Check for Pidgin specifically (but don't override English detection)
        if detected_lang == "en" and is_pidgin(query):
            detected_lang = "pidgin"
        
        # Ensure we have a valid language code
        valid_langs = ["en", "yo", "ig", "ha", "pidgin"]
        if detected_lang not in valid_langs:
            detected_lang = "en"  # Default to English if unknown
        
        # The target language for response should match the query language
        target_lang = detected_lang
        
        logger.info(f"Query: '{query[:50]}...'")
        logger.info(f"Raw detected language: {detect_language(query)}")
        logger.info(f"Final detected language: {detected_lang} ({get_language_name(detected_lang)})")
        logger.info(f"Target response language: {target_lang} ({get_language_name(target_lang)})")

        # Create search variations
        search_variations = create_multilingual_search_variations(query, detected_lang)
        
        # Search knowledge base
        kb_result = search_kb_with_multiple_strategies(search_variations, vector_store)

        if kb_result:
            logger.info(f"Best KB score: {kb_result['score']:.2f} | Threshold: {MIN_SCORE:.2f}")

        if kb_result and kb_result["score"] >= MIN_SCORE:
            logger.info("Decision: Using KB result (above threshold)")
            answer = kb_result["result"].strip()
            
            # Translate the answer to the target language if needed
            if target_lang != "en":
                logger.info(f"Translating response from English to {get_language_name(target_lang)}")
                answer = translate_from_english(answer, target_lang)

            sources = kb_result.get("source_documents", [])
            filenames = []
            for doc in sources[:3]:
                if hasattr(doc, "metadata") and "source" in doc.metadata:
                    filenames.append(doc.metadata["source"].split("/")[-1])
            source_info = "knowledge_base"
            if filenames:
                source_info += f" ({', '.join(set(filenames))})"

            return {
                "source": source_info,
                "answer": answer,
                "lang": target_lang,
                "detected_lang": detected_lang
            }

        # Fallback to web search
        if kb_result:
            logger.info(f"Decision: KB insufficient (score {kb_result['score']:.2f} < threshold {MIN_SCORE:.2f}), using internet fallback")
        else:
            logger.info("Decision: No KB result available, using internet fallback")

        fallback_answer = web_search_fallback(query, target_lang)

        return {
            "source": "internet_search",
            "answer": fallback_answer,
            "lang": target_lang,
            "detected_lang": detected_lang
        }

    except Exception as e:
        logger.error(f"Error in get_response: {e}", exc_info=True)
        return {
            "source": "error",
            "answer": "An error occurred while processing your question. Please try again.",
            "lang": "en",
            "detected_lang": "en"
        }