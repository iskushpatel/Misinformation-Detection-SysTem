"""
FactChk Streamlit Application.

A RAG-based fact-checking application that retrieves similar historical claims
from the LIAR dataset and uses Gemini API to generate verdicts with explanations.
"""

import logging
import os
from typing import Optional

import streamlit as st
from dotenv import load_dotenv

from search import get_retriever, RetrievalConfig
from explain import get_explainer, ExplanationConfig
# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Streamlit page configuration
st.set_page_config(
    page_title="FactChk - AI-Powered Fact Checker",
    page_icon="✓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding: 2rem;
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 2rem;
    }
    
    /* Verdict styling */
    .verdict-true {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    
    .verdict-false {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    
    .verdict-uncertain {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    
    /* Source styling */
    .source-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        border-left: 4px solid #0066cc;
    }
    
    .source-rating {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 0.25rem;
        font-weight: bold;
        margin-top: 0.5rem;
    }
    
    .rating-true {
        background-color: #28a745;
        color: white;
    }
    
    .rating-mostly-true {
        background-color: #5cb85c;
        color: white;
    }
    
    .rating-half-true {
        background-color: #ffc107;
        color: #333;
    }
    
    .rating-mostly-false {
        background-color: #fd7e14;
        color: white;
    }
    
    .rating-false {
        background-color: #dc3545;
        color: white;
    }
    
    .rating-pants-on-fire {
        background-color: #721c24;
        color: white;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def init_retriever() -> object:
    """Initialize and cache the retriever engine."""
    try:
        config = RetrievalConfig(
            qdrant_path="qdrant_db",
            collection_name="liar_statements"
        )
        retriever = get_retriever(config)
        
        # Build database if not already done
        retriever.build_database("data/liar_train.tsv")
        
        return retriever
    except FileNotFoundError:
        st.error(
            "❌ LIAR dataset not found. Please ensure `data/liar_train.tsv` exists."
        )
        st.stop()
    except Exception as e:
        st.error(f"❌ Failed to initialize retriever: {e}")
        logger.error(f"Retriever initialization error: {e}")
        st.stop()


@st.cache_resource
def init_explainer() -> object:
    """Initialize and cache the explanation engine."""
    api_key = os.getenv("GOOGLE_API_KEY")
    
    if not api_key:
        st.error(
            "❌ GOOGLE_API_KEY environment variable not set. "
            "Please add it to your .env file."
        )
        st.stop()
    
    try:
        config = ExplanationConfig(
            model_name="gemini-2.5-flash",
            temperature=0.3,
            max_tokens=500
        )
        return get_explainer(api_key, config)
    except ValueError as e:
        st.error(f"❌ Failed to initialize Gemini API: {e}")
        logger.error(f"Explainer initialization error: {e}")
        st.stop()


def get_verdict_color(verdict: str) -> str:
    """Get CSS class for verdict color."""
    verdict_lower = verdict.lower()
    if verdict_lower == "true":
        return "verdict-true"
    elif verdict_lower == "false":
        return "verdict-false"
    else:
        return "verdict-uncertain"


def get_rating_color(label: str) -> str:
    """Get CSS class for rating color."""
    label_lower = label.lower()
    if label_lower in ["true", "t"]:
        return "rating-true"
    elif label_lower in ["mostly-true", "mostly_true"]:
        return "rating-mostly-true"
    elif label_lower in ["half-true", "half_true"]:
        return "rating-half-true"
    elif label_lower in ["mostly-false", "mostly_false"]:
        return "rating-mostly-false"
    elif label_lower in ["false", "f"]:
        return "rating-false"
    elif label_lower in ["pants-on-fire", "pants_on_fire"]:
        return "rating-pants-on-fire"
    else:
        return "rating-false"


def format_rating_display(label: str) -> str:
    """Format rating label for display."""
    label_lower = label.lower()
    mapping = {
        "t": "TRUE",
        "true": "TRUE",
        "mostly-true": "MOSTLY TRUE",
        "mostly_true": "MOSTLY TRUE",
        "half-true": "HALF TRUE",
        "half_true": "HALF TRUE",
        "mostly-false": "MOSTLY FALSE",
        "mostly_false": "MOSTLY FALSE",
        "f": "FALSE",
        "false": "FALSE",
        "pants-on-fire": "PANTS ON FIRE",
        "pants_on_fire": "PANTS ON FIRE"
    }
    return mapping.get(label_lower, label.upper())


def display_verdict(result: object) -> None:
    """Display the fact-checking verdict and explanation."""
    verdict = result.verdict.value
    explanation = result.explanation
    confidence = result.confidence
    
    # Determine verdict color
    verdict_class = get_verdict_color(verdict)
    
    # Create verdict display
    st.markdown(f"""
    <div class="{verdict_class}">
        <h2 style="margin-top: 0;">Verdict: <strong>[{verdict}]</strong></h2>
        <p><strong>Explanation:</strong></p>
        <p>{explanation}</p>
        <p><strong>Confidence:</strong> {confidence:.1%}</p>
    </div>
    """, unsafe_allow_html=True)


def display_sources(retrieved_sources: list) -> None:
    """Display retrieved sources in an expander."""
    with st.expander("🔍 View Retrieved Sources", expanded=False):
        if not retrieved_sources:
            st.info("No similar statements found in the knowledge base.")
            return
        
        st.write(f"Found {len(retrieved_sources)} similar historical claims:")
        st.markdown("---")
        
        for idx, source in enumerate(retrieved_sources, 1):
            # Extract and format data
            text = source.get('text', 'N/A')
            label = source.get('label', 'unknown')
            speaker = source.get('speaker', 'Unknown')
            context = source.get('context', 'N/A')
            score = source.get('score', 0)
            subjects = source.get('subjects', 'N/A')
            party = source.get('party', 'N/A')
            state = source.get('state', 'N/A')
            job = source.get('job', 'N/A')
            
            # Format rating for display
            rating_display = format_rating_display(label)
            rating_class = get_rating_color(label)
            
            # Create source card
            with st.container():
                st.markdown(f"""
                <div class="source-card">
                    <h4>Source {idx} (Similarity: {score:.1%})</h4>
                    <p><strong>Statement:</strong> "{text}"</p>
                    <p><strong>Speaker:</strong> {speaker}</p>
                    <p><strong>Context:</strong> {context[:200]}{'...' if len(context) > 200 else ''}</p>
                    <div class="source-rating {rating_class}">
                        {rating_display}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                # Additional metadata in columns
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.caption(f"**Job:** {job}")
                with col2:
                    st.caption(f"**Party:** {party}")
                with col3:
                    st.caption(f"**State:** {state}")
                with col4:
                    st.caption(f"**Subjects:** {subjects}")
                
                st.divider()


def main() -> None:
    """Main Streamlit application."""
    
    # Header
    st.markdown("""
    <div class="header-container">
        <h1>✓ FactChk</h1>
        <p style="font-size: 1.2rem; color: #666;">AI-Powered Fact Checker</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        "Enter a claim and FactChk will compare it against historical statements "
        "from PolitiFact's LIAR dataset to determine its veracity."
    )
    st.divider()
    
    # Initialize components
    try:
        retriever = init_retriever()
        explainer = init_explainer()
    except Exception as e:
        st.error(f"❌ Failed to initialize application: {e}")
        return
    
    # Sidebar configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        top_k = st.slider(
            "Number of sources to retrieve:",
            min_value=1,
            max_value=10,
            value=5,
            help="More sources provide more context but may be slower"
        )
        
        similarity_threshold = st.slider(
            "Similarity threshold:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Minimum similarity score for retrieved sources"
        )
        
        # Update configuration
        retriever.config.top_k = top_k
        retriever.config.similarity_threshold = similarity_threshold
        
        # Display statistics
        st.divider()
        st.subheader("📊 Database Stats")
        try:
            stats = retriever.get_statistics()
            st.metric("Total Claims", stats['points_count'])
            st.caption(f"Model: {stats['model']}")
            st.caption(f"Distance Metric: {stats['distance_metric']}")
        except Exception as e:
            st.warning(f"Could not fetch statistics: {e}")
        
        # About section
        st.divider()
        st.subheader("ℹ️ About")
        st.markdown("""
        **FactChk** uses:
        - **LIAR Dataset**: 12.8K fact-checked claims from PolitiFact
        - **Sentence Transformers**: all-MiniLM-L6-v2 for embeddings
        - **Qdrant**: Vector database for similarity search
        - **Google Gemini**: LLM for reasoning and verdicts
        
        **Disclaimer**: This tool provides assistance only and should not be 
        relied upon as the sole source of fact-checking.
        """)
    
    # Main input area
    st.subheader("🔍 Enter Your Claim")
    
    # Text input with example
    example_claims = {
        "Select an example": "",
        "Climate change is real": "Climate change is real",
        "The earth is flat": "The earth is flat",
        "Vaccines cause autism": "Vaccines cause autism"
    }
    
    selected_example = st.selectbox(
        "Or choose an example claim:",
        options=list(example_claims.keys()),
        label_visibility="collapsed"
    )
    
    claim_input = st.text_area(
        "Enter a claim to fact-check:",
        value=example_claims[selected_example],
        height=100,
        placeholder="Enter the claim you want to fact-check..."
    )
    
    # Fact-check button
    col1, col2, col3 = st.columns(3)
    with col1:
        fact_check_button = st.button(
            "🚀 Fact-Check Claim",
            use_container_width=True,
            type="primary"
        )
    
    with col2:
        clear_button = st.button(
            "🗑️ Clear",
            use_container_width=True
        )
    
    if clear_button:
        st.rerun()
    
    st.divider()
    
    # Process claim
    if fact_check_button:
        if not claim_input or not claim_input.strip():
            st.warning("⚠️ Please enter a claim to fact-check.")
            return
        
        # Show progress
        with st.spinner("🔎 Retrieving similar statements..."):
            try:
                retrieved_sources = retriever.retrieve(
                    claim_input,
                    top_k=top_k
                )
            except Exception as e:
                st.error(f"❌ Retrieval failed: {e}")
                logger.error(f"Retrieval error: {e}")
                return
        
        # Generate explanation
        with st.spinner("🤔 Generating explanation..."):
            try:
                result = explainer.explain(claim_input, retrieved_sources)
            except Exception as e:
                st.error(f"❌ Explanation generation failed: {e}")
                logger.error(f"Explanation error: {e}")
                return
        
        # Display results
        st.success("✅ Analysis complete!")
        st.divider()
        
        # Show verdict
        st.subheader("📋 Analysis Result")
        display_verdict(result)
        
        st.divider()
        
        # Show sources
        st.subheader("📚 Evidence")
        display_sources(result.retrieved_sources)
        
        # Show raw response for debugging
        with st.expander("🔧 Debug Info", expanded=False):
            st.json({
                "verdict": result.verdict.value,
                "confidence": result.confidence,
                "sources_retrieved": len(result.retrieved_sources),
                "source_scores": [
                    {"speaker": s.get('speaker'), "score": f"{s.get('score', 0):.3f}"}
                    for s in result.retrieved_sources
                ]
            })


if __name__ == "__main__":
    main()