"""
Explanation Engine for FactChk RAG Application.

This module handles generation of fact-checking explanations using Google Gemini API
by comparing user claims against historically retrieved statements.
"""

import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

import google.generativeai as genai

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VerdicType(str, Enum):
    """Enumeration of possible fact-checking verdicts."""
    
    TRUE = "TRUE"
    FALSE = "FALSE"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class ExplanationConfig:
    """Configuration for the explanation engine."""
    
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0.3  # Lower temperature for more consistent outputs
    max_tokens: int = 500
    top_p: float = 0.8
    top_k: int = 40


class ExplanationResult:
    """
    Result of a fact-checking explanation.
    
    Attributes:
        verdict: The fact-checking verdict (TRUE, FALSE, or UNCERTAIN)
        explanation: Detailed explanation of the verdict
        confidence: Confidence level (0.0 to 1.0)
        retrieved_sources: List of retrieved similar statements used for reasoning
    """
    
    def __init__(
        self,
        verdict: VerdicType,
        explanation: str,
        confidence: float,
        retrieved_sources: List[Dict[str, Any]]
    ) -> None:
        """Initialize an ExplanationResult."""
        self.verdict = verdict
        self.explanation = explanation
        self.confidence = confidence
        self.retrieved_sources = retrieved_sources
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for serialization."""
        return {
            'verdict': self.verdict.value,
            'explanation': self.explanation,
            'confidence': self.confidence,
            'retrieved_sources': self.retrieved_sources
        }


class ExplanationEngine:
    """
    Generates fact-checking explanations using Gemini API.
    
    This engine takes a user claim and similar historical statements
    from the retrieval engine, then uses Gemini to compare them and
    generate a verdict with detailed explanation.
    
    Attributes:
        config: Configuration parameters for the explanation engine
        client: Initialized Generative AI client
    """
    
    def __init__(
        self,
        api_key: str,
        config: Optional[ExplanationConfig] = None
    ) -> None:
        """
        Initialize the Explanation Engine.
        
        Args:
            api_key: Google Gemini API key.
            config: ExplanationConfig instance. If None, uses default config.
            
        Raises:
            ValueError: If API key is empty or invalid.
        """
        if not api_key or not api_key.strip():
            logger.error("API key is required")
            raise ValueError("Google API key cannot be empty")
        
        self.config = config or ExplanationConfig()
        
        try:
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(
                model_name=self.config.model_name,
                generation_config=genai.types.GenerationConfig(
                    temperature=self.config.temperature,
                    max_output_tokens=self.config.max_tokens,
                    top_p=self.config.top_p,
                    top_k=self.config.top_k
                )
            )
            logger.info(f"Initialized Gemini model: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini API: {e}")
            raise ValueError(f"Gemini API initialization failed: {e}") from e
    
    def _build_prompt(
        self,
        claim: str,
        retrieved_sources: List[Dict[str, Any]]
    ) -> str:
        """
        Build a detailed prompt for the Gemini API.
        
        Args:
            claim: The user's claim to fact-check.
            retrieved_sources: List of similar historical statements.
            
        Returns:
            Formatted prompt string for the LLM.
        """
        # Format retrieved sources for readability
        sources_text = self._format_sources(retrieved_sources)
        
        prompt = f"""You are an expert fact-checker. Your task is to evaluate the following claim against historical statements and fact-check ratings from PolitiFact.

USER'S CLAIM TO FACT-CHECK:
"{claim}"

SIMILAR HISTORICAL STATEMENTS FROM THE KNOWLEDGE BASE:
{sources_text}

FACT-CHECKING RATINGS REFERENCE:
- [TRUE]: The claim is supported by strong evidence
- [MOSTLY_TRUE]: The claim is mostly accurate but may have minor inaccuracies
- [HALF_TRUE]: The claim contains both accurate and inaccurate elements
- [MOSTLY_FALSE]: The claim is mostly inaccurate with some accurate elements
- [FALSE]: The claim is demonstrably false
- [PANTS_ON_FIRE]: The claim is a blatant lie with no truth to it

TASK:
1. Analyze the user's claim
2. Compare it against the retrieved similar statements and their official PolitiFact ratings
3. Consider the context and credibility of the speakers
4. Provide a verdict: [TRUE], [FALSE], or [UNCERTAIN]
5. Explain your reasoning with references to the retrieved sources

IMPORTANT:
- Base your verdict primarily on the retrieved sources and their ratings
- If no similar statements are found or the claim is outside the knowledge base scope, output [UNCERTAIN]
- Be concise but thorough in your explanation
- Always cite the relevant historical statement(s) that influenced your verdict
- Provide a confidence level (0.0 to 1.0) for your verdict

OUTPUT FORMAT:
[VERDICT]
Explanation: [Your detailed explanation]
Confidence: [0.0-1.0]"""
        
        return prompt
    
    def _format_sources(self, retrieved_sources: List[Dict[str, Any]]) -> str:
        """
        Format retrieved sources into a readable string.
        
        Args:
            retrieved_sources: List of retrieved statements.
            
        Returns:
            Formatted sources string.
        """
        if not retrieved_sources:
            return "No similar statements found in the knowledge base."
        
        formatted = []
        for idx, source in enumerate(retrieved_sources, 1):
            source_str = f"""
Source {idx}:
- Statement: "{source.get('text', 'N/A')}"
- Speaker: {source.get('speaker', 'Unknown')}
- Official Rating: {source.get('label', 'Unknown').upper()}
- Context: {source.get('context', 'N/A')[:200]}...
- Similarity Score: {source.get('score', 0):.2%}"""
            formatted.append(source_str)
        
        return "\n".join(formatted)
    
    def _parse_response(self, response_text: str) -> Tuple[VerdicType, str, float]:
        """
        Parse the Gemini response to extract verdict, explanation, and confidence.
        
        Args:
            response_text: Raw response from Gemini API.
            
        Returns:
            Tuple of (verdict, explanation, confidence)
            
        Raises:
            ValueError: If response format is invalid.
        """
        try:
            # Extract verdict
            verdict_line = None
            explanation_line = None
            confidence_line = None
            
            for line in response_text.split('\n'):
                if '[TRUE]' in line:
                    verdict_line = VerdicType.TRUE
                elif '[FALSE]' in line:
                    verdict_line = VerdicType.FALSE
                elif '[UNCERTAIN]' in line:
                    verdict_line = VerdicType.UNCERTAIN
                elif line.startswith('Explanation:'):
                    explanation_line = line.replace('Explanation:', '').strip()
                elif line.startswith('Confidence:'):
                    confidence_str = line.replace('Confidence:', '').strip()
                    try:
                        confidence_line = float(confidence_str)
                    except ValueError:
                        confidence_line = 0.5
            
            # Validate extracted values
            if verdict_line is None:
                logger.warning("No verdict found in response, defaulting to UNCERTAIN")
                verdict_line = VerdicType.UNCERTAIN
            
            if explanation_line is None:
                explanation_line = response_text[:500]
            
            if confidence_line is None:
                confidence_line = 0.5
            
            # Clamp confidence to [0, 1]
            confidence_line = max(0.0, min(1.0, confidence_line))
            
            logger.info(f"Parsed response: {verdict_line}, confidence: {confidence_line}")
            return verdict_line, explanation_line, confidence_line
            
        except Exception as e:
            logger.error(f"Failed to parse response: {e}")
            raise ValueError(f"Response parsing failed: {e}") from e
    
    def explain(
        self,
        claim: str,
        retrieved_sources: List[Dict[str, Any]]
    ) -> ExplanationResult:
        """
        Generate a fact-checking explanation for a claim.
        
        Args:
            claim: The user's claim to fact-check.
            retrieved_sources: List of similar historical statements.
            
        Returns:
            ExplanationResult containing verdict, explanation, and confidence.
            
        Raises:
            RuntimeError: If Gemini API call fails.
            ValueError: If response parsing fails.
            
        Example:
            >>> result = engine.explain(
            ...     "Climate change is real",
            ...     retrieved_sources=[...]
            ... )
            >>> print(f"Verdict: {result.verdict}")
            >>> print(f"Explanation: {result.explanation}")
        """
        logger.info(f"Generating explanation for claim: {claim[:100]}...")
        
        try:
            # Build prompt
            prompt = self._build_prompt(claim, retrieved_sources)
            
            # Call Gemini API
            logger.debug("Sending request to Gemini API...")
            response = self.client.generate_content(
                prompt,
                stream=False
            )
            
            if not response.text:
                logger.error("Empty response from Gemini API")
                raise RuntimeError("Gemini API returned empty response")
            
            logger.debug(f"Received response: {response.text[:200]}...")
            
            # Parse response
            verdict, explanation, confidence = self._parse_response(response.text)
            
            # Create result
            result = ExplanationResult(
                verdict=verdict,
                explanation=explanation,
                confidence=confidence,
                retrieved_sources=retrieved_sources
            )
            
            logger.info(f"Explanation generated successfully: {verdict.value}")
            return result
            
        except Exception as e:
            logger.error(f"Failed to generate explanation: {e}")
            raise RuntimeError(f"Explanation generation failed: {e}") from e
    
    def batch_explain(
        self,
        claims: List[str],
        retrieved_sources_list: List[List[Dict[str, Any]]]
    ) -> List[ExplanationResult]:
        """
        Generate explanations for multiple claims.
        
        Args:
            claims: List of claims to fact-check.
            retrieved_sources_list: List of source lists for each claim.
            
        Returns:
            List of ExplanationResult objects.
            
        Raises:
            ValueError: If input lists have different lengths.
        """
        if len(claims) != len(retrieved_sources_list):
            raise ValueError(
                f"Number of claims ({len(claims)}) must match "
                f"number of source lists ({len(retrieved_sources_list)})"
            )
        
        logger.info(f"Generating explanations for {len(claims)} claims...")
        
        results = []
        for claim, sources in zip(claims, retrieved_sources_list):
            try:
                result = self.explain(claim, sources)
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to explain claim '{claim[:50]}...': {e}")
                # Return UNCERTAIN result on failure
                results.append(
                    ExplanationResult(
                        verdict=VerdicType.UNCERTAIN,
                        explanation=f"Unable to generate explanation: {str(e)}",
                        confidence=0.0,
                        retrieved_sources=sources
                    )
                )
        
        return results


def get_explainer(
    api_key: str,
    config: Optional[ExplanationConfig] = None
) -> ExplanationEngine:
    """
    Factory function to get an ExplanationEngine instance.
    
    This is useful for Streamlit app caching with @st.cache_resource.
    
    Args:
        api_key: Google Gemini API key.
        config: ExplanationConfig instance. If None, uses default config.
        
    Returns:
        Initialized ExplanationEngine instance.
    """
    return ExplanationEngine(api_key, config)