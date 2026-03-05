

## System Overview

FactChk is a RAG-based fact-checking system that combines retrieval-augmented generation with vector similarity search to verify claims against historical fact-checked statements.

```
┌─────────────────────────────────────────────────────────────────┐
│                    FactChk System Architecture                   │
├─────────────────────────────────────────────────────────────────┤
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Streamlit User Interface                    │   │
│  │  - Input claim                                           │   │
│  │  - Configuration (top_k, similarity threshold)           │   │
│  │  - Display verdict + sources                            │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Retrieval Engine (search.py)               │   │
│  │  ┌────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Vectorize query (Sentence Transformers)       │ │   │
│  │  │ 2. Search Qdrant (cosine similarity)             │ │   │
│  │  │ 3. Return top-k results with metadata            │ │   │
│  │  └────────────────────────────────────────────────────┘ │   │
│  │                                                          │   │
│  │  Dependencies:                                          │   │
│  │  - sentence-transformers (embedding model)             │   │
│  │  - qdrant-client (vector database)                     │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │             Explanation Engine (explain.py)             │   │
│  │  ┌────────────────────────────────────────────────────┐ │   │
│  │  │ 1. Format sources into rich prompt context        │ │   │
│  │  │ 2. Send to Gemini API with comparison task       │ │   │
│  │  │ 3. Parse response for verdict + explanation      │ │   │
│  │  │ 4. Return structured result                       │ │   │
│  │  └────────────────────────────────────────────────────┘ │   │
│  │                                                          │   │
│  │  Dependencies:                                          │   │
│  │  - google-generativeai (Gemini API)                    │   │
│  └──────────────────────────────────────────────────────────┘   │
│                              │                                    │
│                              ▼                                    │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │              Data Layer                                 │   │
│  │  ┌────────────────────────────────────────────────────┐ │   │
│  │  │ Persistent Storage:                               │ │   │
│  │  │ - qdrant_db/        (vector database)            │ │   │
│  │  │                                                    │ │   │
│  │  │ Raw Data:                                         │ │   │
│  │  │ - data/liar_train.tsv  (12.8K claims)           │ │   │
│  │  └────────────────────────────────────────────────────┘ │   │
│  └──────────────────────────────────────────────────────────┘   │
│                                                                   │
└─────────────────────────────────────────────────────────────────┘
```

---

## Component Deep Dive

### 1. Retrieval Engine (`src/search.py`)

#### Purpose
Efficiently retrieve similar historical claims from a vector database.

#### Key Components

**RetrieverEngine Class**
```
RetrieverEngine
├── Configuration
│   ├── qdrant_path: "qdrant_db"
│   ├── collection_name: "liar_statements"
│   ├── embedding_model: "sentence-transformers/all-MiniLM-L6-v2"
│   ├── vector_size: 384
│   ├── top_k: 5
│   └── similarity_threshold: 0.5
│
├── Methods
│   ├── __init__()           - Initialize client & model
│   ├── build_database()     - Load LIAR dataset
│   ├── _create_collection() - Create Qdrant collection
│   ├── _vectorize_and_store() - Embed & store data
│   ├── retrieve()           - Find similar statements
│   ├── get_statistics()     - Get DB info
│   └── _check_collection_exists() - Smart rebuild skip
│
└── Dependencies
    ├── SentenceTransformer (embedding)
    ├── QdrantClient (vector DB)
    └── pandas (data loading)
```

#### Data Flow

```
LIAR TSV File
    │ (13 columns, no headers)
    ▼
pandas DataFrame
    │ (head(1000))
    ▼
Sentence Transformer
    │ (vectorize 'statement' column)
    ▼
Vector Embeddings (384-dim)
    │
    ▼
Qdrant Collection (cosine distance)
    │
    ▼
Storage: qdrant_db/
    │
    ├─ Metadata payload (per document):
    │  ├── text (statement)
    │  ├── label (TRUE/FALSE/HALF_TRUE/etc)
    │  ├── speaker
    │  ├── context
    │  ├── subjects
    │  ├── party
    │  ├── state
    │  └── job
    │
    └─ Index (for similarity search)
```

#### Retrieval Process

```
User Query: "Climate change is real"
    │
    ▼
Vectorize Query (SentenceTransformer)
    │ Output: [0.12, -0.45, 0.67, ..., 0.23] (384 dims)
    │
    ▼
Qdrant Cosine Similarity Search
    │ Compare against all stored vectors
    │ Find top-5 closest matches
    │
    ▼
Return Results with Payloads
    │
    ├─ {text: "Climate change occurring", label: "true", score: 0.95}
    ├─ {text: "Global warming is real", label: "mostly-true", score: 0.89}
    ├─ {text: "Earth temperatures rising", label: "true", score: 0.87}
    ├─ {text: "Climate caused by CO2", label: "mostly-true", score: 0.85}
    └─ {text: "Warming affects ocean levels", label: "true", score: 0.82}
```

#### Why Cosine Similarity?

- **Semantic Matching**: Captures meaning, not exact text
- **Efficient**: Fast similarity computation
- **Normalized**: Score between 0 and 1
- **Scale-Invariant**: Works regardless of sentence length
- **Industry Standard**: Used by most RAG systems

#### Performance Characteristics

| Operation | Time | Notes |
|-----------|------|-------|
| Build DB (1000 claims) | 2-3 min | One-time cost |
| Vectorize query | 50ms | Instant |
| Qdrant search | 10-50ms | Depends on DB size |
| **Total retrieval** | **100-150ms** | Very fast |

---

### 2. Explanation Engine (`src/explain.py`)

#### Purpose
Generate fact-checking verdicts by comparing claims against historical context.

#### Key Components

**ExplanationEngine Class**
```
ExplanationEngine
├── Configuration
│   ├── model_name: "gemini-pro"
│   ├── temperature: 0.3
│   ├── max_tokens: 500
│   ├── top_p: 0.8
│   └── top_k: 40
│
├── Methods
│   ├── __init__()         - Initialize Gemini API
│   ├── _build_prompt()    - Create rich context prompt
│   ├── _format_sources()  - Structure historical claims
│   ├── explain()          - Generate verdict
│   ├── batch_explain()    - Process multiple claims
│   └── _parse_response()  - Extract structured output
│
└── ExplanationResult
    ├── verdict: VerdicType (TRUE/FALSE/UNCERTAIN)
    ├── explanation: str
    ├── confidence: float (0.0-1.0)
    └── retrieved_sources: List[Dict]
```

#### Prompt Structure

The engine builds a multi-part prompt:

```
1. TASK DEFINITION
   "You are an expert fact-checker..."

2. USER'S CLAIM
   "Climate change is real"

3. SIMILAR HISTORICAL STATEMENTS
   Source 1:
   - Statement: "Climate change is occurring"
   - Speaker: Barack Obama
   - Rating: TRUE
   - Similarity: 94.5%
   
   Source 2:
   - Statement: "Global warming is a hoax"
   - Speaker: James Inhofe
   - Rating: MOSTLY_FALSE
   - Similarity: 78.3%
   
   [... more sources ...]

4. CONTEXT
   - Definition of PolitiFact ratings
   - Trust in speaker's credibility
   - Relevance of retrieved sources

5. INSTRUCTIONS
   - Compare claim to sources
   - Output format: [VERDICT], Explanation, Confidence
   - Base verdict on retrieved evidence

6. OUTPUT SPECIFICATION
   [TRUE]/[FALSE]/[UNCERTAIN]
   Explanation: ...
   Confidence: 0.0-1.0
```

#### Example Reasoning Flow

```
Input Claim: "Vaccines cause autism"

Retrieved Sources (top-3):
1. "Vaccines do not cause autism" - Dr. Andrew Wakefield retraction
   Rating: TRUE (after years of research)
   Similarity: 89%

2. "Wakefield study linking vaccines to autism was fraudulent"
   Rating: TRUE
   Similarity: 85%

3. "Vaccines have mild side effects but not autism"
   Rating: TRUE
   Similarity: 81%

Gemini Reasoning:
- All retrieved sources contradict the claim
- Official PolitiFact rating: uniformly TRUE (vaccines don't cause autism)
- Speaker credibility: Medical experts vs. discredited researcher
- Conclusion: This claim is definitively FALSE

Output:
[FALSE]
Explanation: Multiple authoritative sources confirm vaccines do not cause autism. 
The original Wakefield study claiming this link was fraudulent. Current research 
shows no correlation.
Confidence: 0.95
```

#### Verdict Types

| Verdict | Meaning | Use Case |
|---------|---------|----------|
| **TRUE** | Claim is supported by evidence | Credible sources agree |
| **FALSE** | Claim contradicts evidence | Sources disagree/false |
| **UNCERTAIN** | Insufficient evidence | Outside knowledge base |

#### Temperature Impact

```
Temperature 0.0 (Deterministic)
├─ Always same output
├─ Strict fact-based
└─ Best for: Binary verdicts

Temperature 0.3 (Current Setting)
├─ Slightly creative
├─ Consistent but nuanced
└─ Best for: Fact-checking

Temperature 1.0 (Creative)
├─ Varied outputs
├─ Interpretive
└─ Best for: Creative writing
```

#### API Cost Estimation

- **Input tokens**: ~500-1000 per claim (sources + prompt)
- **Output tokens**: ~100-200 per explanation
- **Cost**: ~$0.01-0.02 per fact-check (Gemini pricing)

---

### 3. Streamlit User Interface (`src/app.py`)

#### Purpose
Provide an interactive interface for fact-checking with transparency.

#### UI Components

```
┌─────────────────────────────────────────────────────────────┐
│  ✓ FactChk - AI-Powered Fact Checker                        │
│  Enter a claim and FactChk will compare it against...       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Sidebar (Configuration)              Main Area             │
│  ├─ top_k slider (1-10)              │                      │
│  ├─ similarity threshold              │  🔍 Enter Claim     │
│  ├─ Database stats                    │  [Text Input Field] │
│  │  - Total Claims: 1000              │                      │
│  │  - Model: all-MiniLM-L6-v2         │  [🚀 Fact-Check]   │
│  └─ About section                     │  [🗑️ Clear]        │
│                                        │                      │
│                                        ├─────────────────────┤
│                                        │                      │
│                                        │ 📋 Analysis Result  │
│                                        │ ┌─────────────────┐ │
│                                        │ │ Verdict: [TRUE] │ │
│                                        │ │ Explanation: ...│ │
│                                        │ │ Confidence: 85% │ │
│                                        │ └─────────────────┘ │
│                                        │                      │
│                                        │ 📚 Evidence         │
│                                        │ ┌─────────────────┐ │
│                                        │ │ 🔍 View Sources │ │
│                                        │ │ [Expander]      │ │
│                                        │ │                 │ │
│                                        │ │ Source 1:       │ │
│                                        │ │ Statement: ...  │ │
│                                        │ │ Speaker: ...    │ │
│                                        │ │ Rating: TRUE    │ │
│                                        │ └─────────────────┘ │
│                                        │                      │
└─────────────────────────────────────────────────────────────┘
```

#### Key Features

1. **Configuration Sidebar**
   - Adjust retrieval parameters
   - View database statistics
   - Display about information

2. **Example Claims Dropdown**
   - Quick selection for demos
   - Pre-populated claims
   - Instant testing

3. **Verdict Display**
   - Color-coded output (green=TRUE, red=FALSE, yellow=UNCERTAIN)
   - Confidence percentage
   - Detailed explanation

4. **Source Transparency Expander**
   - Shows exact retrieved claims
   - Displays speaker information
   - Shows official PolitiFact rating
   - Proves no hallucination

5. **Debug Panel**
   - Show raw response
   - Display similarity scores
   - Verify system internals


## Data Model

### LIAR Dataset Schema (TSV Format)

```
Column Index | Name    | Type   | Example Value
─────────────┼─────────┼────────┼──────────────────────────
0            | id      | str    | "0"
1            | label   | str    | "false" | "true" | "half-true"
2            | statement | str  | "As an example, we're seeing..."
3            | subjects | str   | "climate,science"
4            | speaker | str    | "Al Gore"
5            | job     | str    | "Former Vice President"
6            | state   | str    | "Tennessee"
7            | party   | str    | "Democratic Party"
8            | bt      | int    | 0 (barely_true_count)
9            | fc      | int    | 1 (false_count)
10           | ht      | int    | 0 (half_true_count)
11           | mt      | int    | 0 (mostly_true_count)
12           | pof     | int    | 0 (pants_on_fire_count)
13           | context | str    | "a floor speech."
```

### Qdrant Storage Format

```json
{
  "id": 0,
  "vector": [0.12, -0.45, 0.67, ...],  // 384 dimensions
  "payload": {
    "text": "As an example, we're seeing...",
    "label": "false",
    "speaker": "Al Gore",
    "context": "a floor speech.",
    "subjects": "climate,science",
    "party": "Democratic Party",
    "state": "Tennessee",
    "job": "Former Vice President"
  }
}
```

### API Response Format

```json
{
  "verdict": "FALSE",
  "explanation": "The claim contradicts multiple authoritative sources...",
  "confidence": 0.87,
  "retrieved_sources": [
    {
      "text": "Similar claim text",
      "label": "true",
      "speaker": "Expert Name",
      "context": "context info",
      "score": 0.95
    }
  ]
}
```

---

## Processing Pipeline

### End-to-End Flow

```
1. USER INPUT STAGE
   ├─ User enters claim in Streamlit
   ├─ Validate non-empty input
   └─ Display loading spinner

2. RETRIEVAL STAGE
   ├─ Vectorize claim → 384-dim embedding
   ├─ Query Qdrant with cosine similarity
   ├─ Filter by similarity threshold
   ├─ Return top-k results with metadata
   └─ Time: ~100-150ms

3. PROCESSING STAGE
   ├─ Format sources into structured text
   ├─ Prepare rich context prompt
   ├─ Add instructions for LLM
   └─ Validation complete

4. EXPLANATION STAGE
   ├─ Send prompt to Gemini API
   ├─ Receive response
   ├─ Parse for [VERDICT], explanation, confidence
   └─ Time: ~2-5 seconds

5. DISPLAY STAGE
   ├─ Show verdict with color coding
   ├─ Display confidence percentage
   ├─ Show full explanation
   ├─ Render sources in expander
   └─ Enable source inspection

Total E2E Time: 3-7 seconds
```

---

## Error Handling Strategy

### Graceful Degradation

```
Level 1: Input Validation
├─ Check claim not empty
├─ Validate API key present
└─ Verify dataset exists

Level 2: Component Failures
├─ Retriever failure → Return empty sources
├─ Vectorizer failure → Show error message
├─ API failure → Return UNCERTAIN verdict

Level 3: Recovery
├─ Automatic retry on transient failures
├─ Fallback to simpler prompts
├─ Cache previous results
└─ Graceful error display

Level 4: Logging
├─ All errors logged with context
├─ Tracebacks for debugging
├─ Performance metrics captured
└─ User-friendly error messages
```




---

## Performance Optimization

### Current Performance
| Operation | Time | Bottleneck |
|-----------|------|-----------|
| Vectorization | 50ms | CPU |
| Qdrant search | 50ms | Memory/Disk |
| LLM response | 3000ms | API |
| **Total** | **3.1s** | API (96%) |


---



---

## Monitoring & Logging

### Key Metrics

```
Application Level
├─ Total requests: counter
├─ Latency: histogram (p50, p95, p99)
├─ Error rate: percentage
├─ Cache hit rate: percentage
└─ API cost: total $/month

Component Level
├─ Retriever
│  ├─ Query latency
│  ├─ Sources returned
│  └─ Similarity scores
├─ Explainer
│  ├─ LLM latency
│  ├─ Token usage
│  └─ Cost per request
└─ UI
   ├─ Page load time
   ├─ User session duration
   └─ Errors/crashes
```

### Log Levels

```
DEBUG: Detailed execution (vectorization, search steps)
INFO: Key milestones (database built, API called)
WARNING: Non-critical issues (slow response)
ERROR: Failures (API error, empty results)
CRITICAL: System failure (database unreachable)
```

---

---
