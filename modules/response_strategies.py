# modules/response_strategies.py: Strategy pattern for response generation
"""
Response Strategies: Implements different response generation strategies
- KnowledgeBaseStrategy: Search private knowledge base
- WebSearchStrategy: Fallback to web search
- ResponseOrchestrator: Coordinates strategies
"""

import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

from modules.config_manager import get_config
from modules.cache_manager import get_cache
from modules.exceptions import KBSearchError, WebSearchError, LLMResponseError
from modules.language_service import LanguageService

logger = logging.getLogger(__name__)


class ResponseStrategy(ABC):
    """Base class for response strategies"""
    
    @abstractmethod
    def get_response(self, query: str, lang: str, **kwargs) -> Dict[str, Any]:
        """
        Get response for a query
        
        Returns:
            {
                'success': bool,
                'answer': str,
                'source': str,
                'score': float,
                'error': Optional[str]
            }
        """
        pass


class KnowledgeBaseStrategy(ResponseStrategy):
    """
    Search knowledge base strategy with quality scoring
    """
    
    def __init__(self, vector_store, llm, language_service: LanguageService):
        self.vector_store = vector_store
        self.llm = llm
        self.language_service = language_service
        self.cache = get_cache()
        self.config = get_config()
        logger.info("KnowledgeBaseStrategy initialized")
    
    def get_response(self, query: str, lang: str, **kwargs) -> Dict[str, Any]:
        """Search KB and return response if quality is sufficient"""
        try:
            logger.info(f"KB Strategy: Searching for query in {lang}")
            
            # Check cache first
            cached_result = self.cache.get_kb_result(query, lang)
            if cached_result:
                logger.info("KB cache hit!")
                return cached_result
            
            # Create search variations
            search_variations = self._create_search_variations(query, lang)
            
            # Search KB with variations
            best_result = self._search_with_variations(search_variations)
            
            if not best_result:
                return {
                    'success': False,
                    'answer': '',
                    'source': 'kb_no_results',
                    'score': 0.0,
                    'error': 'No KB results found'
                }
            
            # Score the result
            score = self._score_result(
                best_result['result'],
                best_result.get('source_documents', []),
                query
            )
            
            logger.info(f"KB score: {score:.2f} (threshold: {self.config.retrieval.min_score})")
            
            # Check if score meets threshold
            if score >= self.config.retrieval.min_score:
                # Extract source information
                source_info = self._extract_sources(best_result.get('source_documents', []))
                
                result = {
                    'success': True,
                    'answer': best_result['result'].strip(),
                    'source': source_info,
                    'score': score,
                    'query_used': best_result.get('query_used', query)
                }
                
                # Cache the successful result
                self.cache.set_kb_result(query, result, lang)
                
                return result
            else:
                return {
                    'success': False,
                    'answer': best_result['result'].strip(),
                    'source': 'kb_low_score',
                    'score': score,
                    'error': f'KB score {score:.2f} below threshold {self.config.retrieval.min_score}'
                }
                
        except Exception as e:
            logger.error(f"KB Strategy failed: {e}")
            return {
                'success': False,
                'answer': '',
                'source': 'kb_error',
                'score': 0.0,
                'error': str(e)
            }
    
    def _create_search_variations(self, query: str, lang: str) -> List[str]:
        """Create multiple search variations for better retrieval"""
        variations = [query]
        
        # Add English translation if not English
        if lang != "en":
            try:
                english_query = self.language_service.translate_to_english(query, lang)
                if english_query and english_query != query:
                    variations.append(english_query)
            except Exception as e:
                logger.warning(f"Failed to create English variation: {e}")
        
        # Add TB-specific variation
        if "tb" not in query.lower() and "tuberculosis" not in query.lower():
            variations.append(f"{query} tuberculosis")
        
        # Remove duplicates while preserving order
        variations = list(dict.fromkeys(variations))
        logger.info(f"Search variations created: {variations}")
        return variations
    
    def _search_with_variations(self, variations: List[str]) -> Optional[Dict[str, Any]]:
        """Try multiple search variations and return best result"""
        best_result = None
        best_score = 0
        
        qa_chain = self._create_qa_chain()
        
        for variation in variations:
            try:
                response = qa_chain.invoke({"query": variation})
                result_text = response.get("result", "")
                sources = response.get("source_documents", [])
                
                # Quick score to find best variation
                score = self._quick_score(result_text, sources)
                
                logger.debug(f"Variation '{variation[:50]}...' scored: {score:.2f}")
                
                if score > best_score:
                    best_score = score
                    best_result = {
                        "result": result_text,
                        "source_documents": sources,
                        "query_used": variation
                    }
                    
            except Exception as e:
                logger.warning(f"Search failed for variation '{variation[:50]}...': {e}")
                continue
        
        return best_result
    
    def _create_qa_chain(self) -> RetrievalQA:
        """Create RetrievalQA chain"""
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
            llm=self.llm,
            chain_type="stuff",
            retriever=self.vector_store.as_retriever(
                search_kwargs={"k": self.config.retrieval.retrieval_k}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": QA_PROMPT}
        )
    
    def _quick_score(self, text: str, sources: List) -> float:
        """Quick scoring for variation comparison"""
        score = 0.0
        
        if not text or len(text.strip()) < 10:
            return 0.0
        
        # Length scoring
        if 100 <= len(text) <= 500:
            score += 2.0
        
        # Source scoring
        if sources:
            score += len(sources) * 0.5
        
        return score
    
    def _score_result(self, result_text: str, sources: List, query: str) -> float:
        """
        Comprehensive quality scoring for KB results
        Higher score = better quality
        """
        score = 0.0
        
        if not result_text or len(result_text.strip()) < 10:
            return 0.0
        
        result_lower = result_text.lower()
        
        # Negative indicators (immediate disqualification)
        negative_indicators = [
            "don't know", "not sure", "cannot answer", "insufficient information",
            "not enough information", "context does not contain", "i don't have",
            "unable to answer", "no information", "not available in"
        ]
        
        if any(ind in result_lower for ind in negative_indicators):
            logger.info("KB result contains negative indicator")
            return 0.0
        
        # Length scoring (substantial answers)
        text_length = len(result_text.strip())
        if 100 <= text_length <= 500:
            score += 4.0
        elif 50 <= text_length < 100:
            score += 2.0
        elif text_length > 500:
            score += 3.0
        elif text_length < 50:
            score += 0.5
        
        # Source documents scoring
        if sources and len(sources) >= 2:
            score += 3.0
        elif sources and len(sources) == 1:
            score += 1.5
        else:
            score -= 2.0
        
        # Base positive score
        score += 2.0
        
        # TB relevance
        tb_keywords = [
            "tuberculosis", "tb", "infection", "treatment", "symptoms",
            "prevention", "lungs", "bacteria", "disease", "diagnosis"
        ]
        tb_matches = sum(1 for kw in tb_keywords if kw in result_lower)
        
        if tb_matches >= 3:
            score += 3.0
        elif tb_matches >= 1:
            score += 1.5
        else:
            score -= 2.0
        
        # Query overlap (answer addresses the question)
        query_words = set(w.lower() for w in query.split() if len(w) > 3)
        result_words = set(w.lower() for w in result_text.split())
        overlap = query_words.intersection(result_words)
        
        if query_words:
            overlap_ratio = len(overlap) / len(query_words)
            if overlap_ratio >= 0.5:
                score += 3.0
            elif overlap_ratio >= 0.3:
                score += 1.5
            else:
                score -= 1.0
        
        logger.debug(
            f"Score breakdown: length={text_length}, sources={len(sources)}, "
            f"tb_matches={tb_matches}, final={score:.2f}"
        )
        
        return score
    
    def _extract_sources(self, source_documents: List) -> str:
        """Extract source information from documents"""
        if not source_documents:
            return "knowledge_base"
        
        filenames = []
        for doc in source_documents[:3]:
            if hasattr(doc, "metadata") and "source" in doc.metadata:
                filename = doc.metadata["source"].split("/")[-1]
                filenames.append(filename)
        
        if filenames:
            unique_files = list(set(filenames))
            return f"knowledge_base ({', '.join(unique_files)})"
        
        return "knowledge_base"


class WebSearchStrategy(ResponseStrategy):
    """
    Web search fallback strategy using DuckDuckGo
    """
    
    def __init__(self, llm, language_service: LanguageService):
        self.llm = llm
        self.language_service = language_service
        self.config = get_config()
        logger.info("WebSearchStrategy initialized")
    
    def get_response(self, query: str, lang: str, **kwargs) -> Dict[str, Any]:
        """Search web and generate response"""
        try:
            logger.info(f"Web Search Strategy: Searching for query in {lang}")
            
            # Perform web search
            search_results = self._search_web(query)
            
            if not search_results or len(search_results.strip()) < 20:
                return {
                    'success': False,
                    'answer': '',
                    'source': 'web_no_results',
                    'score': 0.0,
                    'error': 'No web search results found'
                }
            
            # Summarize results with LLM
            summary = self._summarize_results(query, search_results)
            
            # Translate if needed
            if lang != "en":
                summary = self.language_service.translate_from_english(summary, lang)
            
            return {
                'success': True,
                'answer': summary,
                'source': 'internet_search',
                'score': 1.0,  # Web search always "succeeds" if it returns results
            }
            
        except Exception as e:
            logger.error(f"Web Search Strategy failed: {e}")
            return {
                'success': False,
                'answer': '',
                'source': 'web_error',
                'score': 0.0,
                'error': str(e)
            }
    
    def _search_web(self, query: str) -> str:
        """Perform DuckDuckGo search"""
        try:
            wrapper = DuckDuckGoSearchAPIWrapper(
                max_results=5,
                region="wt-wt",
                safesearch="moderate",
                time="y"
            )
            
            # Enhanced search with medical sources
            search_terms = f"tuberculosis {query} site:who.int OR site:cdc.gov OR site:nhs.uk OR site:mayoclinic.org"
            
            logger.info(f"Web search: {search_terms}")
            results = wrapper.run(search_terms)
            
            if not results or len(results.strip()) < 20:
                # Try broader search
                search_terms = f"tuberculosis {query} health medical"
                results = wrapper.run(search_terms)
            
            logger.info(f"Web search returned {len(results)} characters")
            return results
            
        except Exception as e:
            logger.error(f"Web search failed: {e}")
            raise WebSearchError(f"Web search error: {e}")
    
    def _summarize_results(self, query: str, results: str) -> str:
        """Use LLM to summarize search results"""
        try:
            summary_prompt = f"""You are a tuberculosis (TB) health expert. Based on the following search results,
provide a clear, accurate, and helpful answer to the user's question.

User's Question: {query}

Search Results:
{results}

Instructions:
- Provide factual, medically accurate information
- Be specific and actionable when possible
- Keep the response between 100-300 words
- Focus on tuberculosis-related information
- Do not invent information not in the search results
- If the results don't contain relevant TB information, say so

Answer:"""
            
            response = self.llm.invoke([HumanMessage(content=summary_prompt)])
            summary = response.content.strip()
            
            logger.info(f"Web search summary generated: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.error(f"Failed to summarize web results: {e}")
            raise LLMResponseError(f"Summary generation failed: {e}")


class ResponseOrchestrator:
    """
    Coordinates between different response strategies
    Tries strategies in order until one succeeds
    """
    
    def __init__(self, strategies: List[ResponseStrategy]):
        self.strategies = strategies
        logger.info(f"ResponseOrchestrator initialized with {len(strategies)} strategies")
    
    def get_response(self, query: str, lang: str) -> Dict[str, Any]:
        """
        Get response using the first successful strategy
        
        Returns:
            {
                'answer': str,
                'source': str,
                'lang': str,
                'detected_lang': str,
                'strategy_used': str,
                'score': float
            }
        """
        logger.info(f"Orchestrator: Processing query in {lang}")
        logger.info("=" * 70)
        
        for idx, strategy in enumerate(self.strategies):
            strategy_name = strategy.__class__.__name__
            logger.info(f"Trying strategy {idx + 1}/{len(self.strategies)}: {strategy_name}")
            
            try:
                result = strategy.get_response(query, lang)
                
                if result.get('success'):
                    logger.info(f"✓ Strategy {strategy_name} succeeded")
                    logger.info("=" * 70)
                    
                    return {
                        'answer': result['answer'],
                        'source': result['source'],
                        'lang': lang,
                        'detected_lang': lang,
                        'strategy_used': strategy_name,
                        'score': result.get('score', 0.0)
                    }
                else:
                    logger.info(
                        f"✗ Strategy {strategy_name} failed: {result.get('error', 'Unknown')}"
                    )
                    
            except Exception as e:
                logger.error(f"✗ Strategy {strategy_name} error: {e}")
                continue
        
        # All strategies failed
        logger.error("All response strategies failed")
        logger.info("=" * 70)
        
        return {
            'answer': "I apologize, but I encountered an error while processing your question. Please try again.",
            'source': 'error',
            'lang': lang,
            'detected_lang': lang,
            'strategy_used': 'none',
            'score': 0.0
        }


def create_response_orchestrator(vector_store, llm, language_service: LanguageService) -> ResponseOrchestrator:
    """Factory function to create response orchestrator with all strategies"""
    strategies = [
        KnowledgeBaseStrategy(vector_store, llm, language_service),
        WebSearchStrategy(llm, language_service)
    ]
    return ResponseOrchestrator(strategies)
