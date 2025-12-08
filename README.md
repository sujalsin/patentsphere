# PatentSphere - Self-Correcting Patent Analysis System

A production-grade multi-agent RAG system with RLAIF (Reinforcement Learning from AI Feedback) self-correction for patent analysis. The system synthesizes multi-modal data (text + graph + temporal metadata) with high precision and provides verifiable citations for every claim.

## Features

- 🔍 **Hybrid Search**: Combines dense (semantic) and sparse (keyword) vectors for optimal retrieval
- 🧠 **Multi-Agent Architecture**: 4 specialized agents (Router, Extractor, Synthesizer, Critic)
- ✅ **Self-Correction**: RLAIF loop that validates and corrects responses before delivery
- 📊 **Query Expansion**: Generates 3-5 search variations to maximize recall
- ⚖️ **Litigation Analysis**: Integrated legal risk assessment
- 🚀 **Token Streaming**: Real-time response generation for better UX
- 🔗 **Citation Verification**: Prevents hallucinated citations

## Architecture

The system uses a **Cyclic State Graph** (LangGraph) that allows the system to "think, check, and correct" itself:

1. **Router** (phi4-mini): Classifies query intent (LEGAL/TECHNICAL/BOTH)
2. **Extractor** (qwen2.5:1.5b): Expands query into 3-5 search variations
3. **Retrieval**: Parallel execution of vector search (Qdrant) and metadata lookup (PostgreSQL)
4. **Synthesizer** (gemma3:4b): Generates report with citations
5. **Critic** (llama3.2:3b): Validates draft and verifies citations
6. **RLAIF Loop**: If critic fails, retry synthesis (max 3 attempts)

## Setup

### Prerequisites

- Python 3.10+
- Docker and Docker Compose
- Ollama installed and running

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Start Services

```bash
docker-compose up -d
```

This starts:
- Qdrant (vector database) on port 6333
- PostgreSQL on port 5432

### 3. Initialize Database

```bash
python db/init_db.py
```

### 4. Install Ollama Models

```bash
ollama pull phi4-mini:latest
ollama pull qwen2.5:1.5b-instruct
ollama pull gemma3:4b
ollama pull llama3.2:3b
```

Verify models are installed:
```bash
python scripts/verify_ollama.py
```

### 5. Ingest Data

```bash
# Ingest patents
python scripts/ingest_patents.py data/patents_bigquery.jsonl

# Ingest litigation data
python scripts/ingest_litigation.py data/stanford_litigation.jsonl stanford_npe
python scripts/ingest_litigation.py data/uspto_litigation.jsonl uspto_oce
```

### 6. Configure Environment

Copy `.env.example` to `.env` and adjust settings if needed (defaults should work for local development).

## Usage

### Streamlit Frontend

```bash
streamlit run app.py
```

Then open http://localhost:8501 in your browser.

### Example Queries

- **Legal**: "Has Apple sued anyone over touchscreens?"
- **Technical**: "Find prior art for transformer neural networks"
- **Both**: "What patents cover self-driving car sensors and have they been litigated?"

## Testing

Run the test suite:

```bash
python tests/test_workflow.py
```

Tests cover:
- Query routing
- Query expansion
- Citation verification
- RLAIF loop

## Project Structure

```
patentsphere/
├── config.py              # Configuration (Pydantic Settings)
├── docker-compose.yml      # Qdrant and PostgreSQL services
├── requirements.txt        # Python dependencies
├── AGENTS.md              # Agent behavior definitions
├── app.py                 # Streamlit frontend
├── db/
│   ├── qdrant_client.py   # Hybrid search client
│   ├── postgres_client.py # PostgreSQL client
│   └── init_db.py         # Database schema
├── graph/
│   ├── nodes.py           # Agent nodes
│   ├── tools.py           # Retrieval tools
│   └── graph.py           # LangGraph orchestrator
├── scripts/
│   ├── ingest_patents.py  # Patent data ingestion
│   ├── ingest_litigation.py # Litigation data ingestion
│   └── verify_ollama.py   # Model verification
└── tests/
    └── test_workflow.py    # Workflow tests
```

## Technical Details

### Hybrid Search

- **Dense Vectors**: `snowflake-arctic-embed-m` for semantic similarity
- **Sparse Vectors**: SPLADE via `fastembed` for exact keyword matching
- Combines both for optimal retrieval of patent numbers and technical terms

### Parallel Retrieval

Vector search and metadata lookup execute concurrently using `asyncio.gather()`, reducing latency by ~40%.

### Citation Format

All citations use the format: `[[PatentID]](URL)`

Example: `[[US9876543]](https://patents.google.com/patent/US9876543)`

### RLAIF Loop

The critic agent validates:
1. No hallucinated claims (all facts must be in context)
2. Correct citation format
3. All cited patent IDs exist in provided context

If validation fails, the system retries synthesis with feedback (max 3 attempts).

## Configuration

All configuration is managed via `config.py` using Pydantic Settings. Key settings:

- Database connections (PostgreSQL, Qdrant)
- Ollama model names and base URL
- Embedding models

## License

MIT

## Contributing

Contributions welcome! Please ensure all tests pass before submitting PRs.


