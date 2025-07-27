# LLM handling utilities for LightHouse Connect: modules/llm_handler.py
import requests
import json
import logging

from langchain_openai import ChatOpenAI as OpenRouterChatLLM
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

from config import OPENROUTER_API_KEY, LLM_MODEL

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def get_llm():
    """
    Initializes and returns the LLM from OpenRouter using LangChain's ChatOpenAI compatible client.
    """
    if not OPENROUTER_API_KEY:
        logger.error("OPENROUTER_API_KEY is not set in config.py.")
        raise ValueError("OPENROUTER_API_KEY environment variable is missing.")

    logger.info(f"Initializing LLM with model: {LLM_MODEL} via OpenRouter.")
    
    return OpenRouterChatLLM(
        model_name=LLM_MODEL,
        openai_api_base="https://openrouter.ai/api/v1",
        openai_api_key=OPENROUTER_API_KEY,
        temperature=0.3,
    )

def get_qa_chain(vector_store):
    """
    Creates and returns a RetrievalQA chain.
    """
    llm = get_llm()
    
    qa_template = """Use the following pieces of context to answer the question at the end. 
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    
    {context}
    
    Question: {question}
    Helpful Answer:"""
    QA_PROMPT = PromptTemplate(template=qa_template, input_variables=["context", "question"])

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": QA_PROMPT}
    )
    logger.info("RetrievalQA chain initialized.")
    return qa_chain

def search_internet(query):
    """
    Searches the internet for a given query using DuckDuckGo.
    """
    url = f"https://api.duckduckgo.com/?q={query}&format=json"
    try:
        logger.info(f"Attempting internet search for query: '{query}'")
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        abstract = data.get("Abstract", "").strip()
        related_topics = data.get("RelatedTopics", [])

        snippets = []
        if abstract:
            snippets.append(abstract)

        for topic in related_topics:
            if isinstance(topic, dict) and topic.get("Text"):
                snippets.append(topic["Text"].strip())
            elif isinstance(topic, dict) and topic.get("Topics"):
                for sub_topic in topic["Topics"]:
                    if isinstance(sub_topic, dict) and sub_topic.get("Text"):
                        snippets.append(sub_topic["Text"].strip())

        combined_snippets = "\n\n".join(filter(None, snippets))
        
        if combined_snippets:
            logger.info("Internet search successful.")
            return combined_snippets
        else:
            logger.info("Internet search found no useful information.")
            return "No useful information found online for this query."

    except requests.exceptions.RequestException as e:
        logger.error(f"Internet search failed for query '{query}': {e}")
        return f"Internet search failed: {e}"
    except Exception as e:
        logger.error(f"An unexpected error occurred during internet search for '{query}': {e}")
        return f"An unexpected error occurred during internet search: {e}"

def get_response(query, vector_store):
    """
    Gets a response from the chatbot.
    Prioritizes the local knowledge base, falls back to internet search if needed.
    """
    try:
        qa_chain = get_qa_chain(vector_store)
        logger.info(f"Attempting to retrieve answer from knowledge base for query: '{query}'")
        kb_response = qa_chain.invoke({"query": query})
        
        # --- FIX: Correctly extract the string answer from the chain's result ---
        # The key for the answer from RetrievalQA is 'result'.
        # The result from a ChatModel is a message object (e.g., AIMessage).
        kb_result_obj = kb_response.get("result")
        source_documents = kb_response.get("source_documents", [])
        
        answer = ""
        if kb_result_obj:
            # If it's a message object, get its content. Otherwise, convert to string.
            if hasattr(kb_result_obj, 'content'):
                answer = kb_result_obj.content.strip()
            else:
                answer = str(kb_result_obj).strip()
        # --- END FIX ---

        # Check if the KB answer is confident and relevant
        if "don't know" in answer.lower() or "no information" in answer.lower() or not source_documents:
            logger.info("Knowledge base did not provide a confident answer. Falling back to internet search.")
            internet_context = search_internet(query)
            
            llm = get_llm()
            messages = [
                SystemMessage(content="You are a helpful assistant. Answer the user's question based ONLY on the provided internet search results."),
                HumanMessage(content=f"Question: '{query}'\n\nInternet search results:\n{internet_context}\n\nIf the internet results do not contain the answer, state that the information is not available."),
            ]
            final_answer_obj = llm.invoke(messages)
            final_answer = final_answer_obj.content
            
            return {"source": "web", "answer": final_answer}
        else:
            logger.info("Answer found in knowledge base.")
            sources = ", ".join(set([doc.metadata.get("source", "N/A") for doc in source_documents]))
            return {"source": f"knowledge_base (Sources: {sources})", "answer": answer}

    except Exception as e:
        logger.error(f"An unhandled error occurred in get_response: {e}", exc_info=True)
        return {"source": "error", "answer": f"An unexpected error occurred: {e}"}