# PatentSphere

> **Turn a raw inventor disclosure into a complete, attorney-ready patent draft in minutes — with full traceability, verifiable citations, and a Human-Wall approval checkpoint before export.**

---

## 🚀 What's New: Disclosure-to-Strategic-Patent-Draft Pipeline

PatentSphere now ships with a **hybrid dual-mode architecture**:

| Mode | What it does |
|------|-------------|
| **Analysis Mode** (original) | Q&A over your patent database — prior art, litigation risk, claim analysis |
| **Full Draft Mode** (new) | Raw disclosure → complete patent specification in 5 steps |

### 5-Step Wizard Flow

```
Upload Raw Disclosure
        │
        ▼
Step 1 ─ Decompose      Split messy input into 7 clean sections with provenance + confidence scores
        │
        ▼
Step 2 ─ Analyze        Parallel RAG + RLAIF on every section → novelty, obviousness & litigation risk
        │
        ▼
Step 3 ─ Draft Claims   1 broad independent claim (market-protecting) + 3-5 layered dependent claims
        │                Each claim ships with plain-English strategy reasoning
        ▼
Step 4 ─ Full Spec      Complete title, abstract, background, summary, detailed description & claims
        │                ✨ Includes Prior Art Navigation Sidebar & 2x2 Competitive Positioning Map
        ▼
Step 5 ─ Human-Wall     Multi-critic review: per-section confidence scores, attorney flags, specific risk mitigation strategies
                        Export is LOCKED until you type "approve"
                        → DOCX (clickable citations) + PDF (Human-Wall Approved stamp)
```

### How to Trigger Full Draft Mode

```
# Option 1: type /draft in chat, then paste your disclosure
/draft

# Option 2: paste disclosure text directly after the command
/draft My invention is a neural network that predicts patent validity...

# Option 3: upload a PDF, TXT, or DOCX file — wizard starts automatically
```

### Demo Mode Toggle

PatentSphere selects the LLM automatically based on your environment:

| Environment | LLM Used | Notes |
|------------|----------|-------|
| `GROQ_API_KEY` set in `.env` | Groq `llama-3.1-8b-instant` | Fastest — ideal for public demos |
| No Groq key, `DEMO_MODE=true` | Ollama `llama3.2:3b` | Strong local CPU fallback |
| Default (nothing set) | Per-agent Ollama models | Privacy-first — fully local |

```bash
# .env — enable demo mode
GROQ_API_KEY=gsk_...         # Groq demo mode (fastest)
# or
DEMO_MODE=true               # Strong local CPU fallback (llama3.2:3b)
```

### Privacy-First Design

- **Default**: 100% local (Ollama + Qdrant + PostgreSQL). No data leaves your machine.
- **Demo mode**: Groq is opt-in via explicit `GROQ_API_KEY`. The key is never logged.
- **Audit log**: Every pipeline run produces an append-only audit trail tracing each output back to the original input.

### Enterprise Bridge Architecture

The system is architected for the path from MVP to enterprise:

```
Current (single-user / MVP)           Future Enterprise Bridge
────────────────────────────────      ──────────────────────────────────────────
Single Qdrant collection: patents     Per-tenant: {tenant_id}_patents
Single audit log                      Per-tenant audit log partitioning
File-based export                     S3 / Azure Blob with tenant isolation
No auth layer                         JWT → tenant_id → collection routing
```

`tenant_id` is already threaded through `AgentState`, all pipeline models, and Qdrant search calls — marked with `# ENTERPRISE BRIDGE` comments throughout the codebase. Activating full multi-tenancy requires changing one line per search call.

**Future heterogeneous source integrations** (bridge targets):
- University OTT / IP portal systems
- Corporate invention disclosure management (IDM) platforms
- USPTO, EPO, and WIPO bulk data feeds
- Law firm docketing and prosecution management systems

---

# PatentSphere - Self-Correcting Patent Analysis System

A production-grade multi-agent RAG system with RLAIF (Reinforcement Learning from AI Feedback) self-correction for patent analysis. The system synthesizes multi-modal data (text + graph + temporal metadata) with high precision and provides verifiable citations for every claim.

### Advanced Strategic Features (Wow Factor)
- **Competitive Positioning Map**: Automatically generates a 2x2 matrix placing the drafted invention against retrieved prior art competitors (e.g., Privacy vs. Clinical Integration).
- **Prior Art Navigation**: Actively maps specific drafted claim limitations to identified gaps in the prior art.
- **Resilient Human-Wall Review**: Unlike basic RLAIF critics that collapse on high-risk inventions, PatentSphere's reviewer forces structured claim-drafting strategies and specific prior art distinctions, avoiding "manual review" dead ends.

## Core Capabilities

- Hybrid Search: Combines dense (semantic) and sparse (keyword) vectors for optimal retrieval
- Multi-Agent Architecture: 4 specialized agents (Router, Extractor, Synthesizer, Critic)
- Self-Correction: RLAIF loop that validates and corrects responses before delivery
- Query Expansion: Generates 3-5 search variations to maximize recall
- Litigation Analysis: Integrated legal risk assessment
- Token Streaming: Real-time response generation for better UX
- Citation Verification: Prevents hallucinated citations

## Architecture

The system uses a Cyclic State Graph (LangGraph) that allows the system to "think, check, and correct" itself:

1. Router (phi4-mini): Classifies query intent (LEGAL/TECHNICAL/BOTH)
2. Extractor (qwen2.5:1.5b): Expands query into 3-5 search variations
3. Retrieval: Parallel execution of vector search (Qdrant) and metadata lookup (PostgreSQL)
4. Synthesizer (gemma3:4b): Generates report with citations
5. Critic (llama3.2:3b): Validates draft and verifies citations
6. RLAIF Loop: If critic fails, retry synthesis (max 3 attempts)

## Prerequisites

- Python 3.10 or higher
- Docker and Docker Compose
- Ollama installed and running locally
- At least 8GB RAM (16GB recommended for running all models)

## Quick Start

### Option 1: Docker Setup (Recommended)

This is the easiest way to get started. The Docker setup handles all dependencies automatically.

#### Step 1: Install Ollama

Download and install Ollama from https://ollama.ai

Start the Ollama service:
```bash
ollama serve
```

Keep this running in a separate terminal.

#### Step 2: Pull Required Models

In a new terminal, pull all required Ollama models:
```bash
ollama pull phi4-mini:latest
ollama pull qwen2.5:1.5b-instruct
ollama pull gemma3:4b
ollama pull llama3.2:3b
```

Verify models are installed:
```bash
ollama list
```

#### Step 3: Start Database Services

From the project root directory, start PostgreSQL and Qdrant:
```bash
docker compose up -d
```

This starts:
- Qdrant (vector database) on port 6333
- PostgreSQL on port 5432

Verify services are running:
```bash
docker compose ps
```

#### Step 4: Prepare Data Directory

Create a data directory for your patent and litigation files:
```bash
mkdir -p data
```

Place your data files in the `data/` directory. Supported formats:
- Patent data: JSONL files (e.g., `patents_bigquery.jsonl`, `patents_2015.jsonl`)
- Litigation data: JSONL files (e.g., `stanford_litigation.jsonl`, `uspto_litigation.jsonl`)

#### Step 5: Ingest Data

Run the ingestion service to load data into the databases:
```bash
docker compose -f docker/docker-compose.yml run --rm --no-deps ingest
```

The `--no-deps` flag prevents starting duplicate postgres/qdrant containers since they're already running from step 3.

This will:
- Initialize the database schema
- Ingest litigation data (if files are present in `data/`)
- Ingest patent data (if files are present in `data/`)

#### Step 6: Start the Application

Start the Chainlit application:
```bash
docker compose -f docker/docker-compose.yml up app
```

Or run it in detached mode:
```bash
docker compose -f docker/docker-compose.yml up -d app
```

The application will be available at http://localhost:8000

### Option 2: Local Development Setup

For development or if you prefer running without Docker.

#### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

#### Step 2: Install and Start Ollama

Download and install Ollama from https://ollama.ai

Start the Ollama service:
```bash
ollama serve
```

#### Step 3: Pull Required Models

```bash
ollama pull phi4-mini:latest
ollama pull qwen2.5:1.5b-instruct
ollama pull gemma3:4b
ollama pull llama3.2:3b
```

Verify models:
```bash
python scripts/verify_ollama.py
```

#### Step 4: Start Database Services

Start PostgreSQL and Qdrant using Docker:
```bash
docker compose up -d
```

#### Step 5: Initialize Database

```bash
python db/init_db.py
```

#### Step 6: Ingest Data

```bash
# Ingest patents
python scripts/ingest_patents.py data/patents_bigquery.jsonl

# Ingest litigation data
python scripts/ingest_litigation.py data/stanford_litigation.jsonl stanford
python scripts/ingest_litigation.py data/uspto_litigation.jsonl uspto_oce
```

#### Step 7: Start the Application

```bash
chainlit run app.py
```

The application will be available at http://localhost:8000

## Usage

### Accessing the Application

Once the application is running, open your browser and navigate to:
- Docker: http://localhost:8000
- Local: http://localhost:8000

### Example Queries

Legal Query:
```
Has Apple sued anyone over touchscreens?
```

Technical Query:
```
Find prior art for transformer neural networks
```

Combined Query:
```
What patents cover self-driving car sensors and have they been litigated?
```

### Understanding the Interface

The Chainlit interface provides:
- Real-time agent status updates showing which agent is processing
- Streaming responses as they are generated
- Side panel with retrieved patent documents and litigation cases
- Clickable citations that link to patent details

## Docker Compose Files

This project uses two docker-compose files:

1. `docker-compose.yml` (root): Contains only database services (PostgreSQL and Qdrant)
   - Use this to start/stop database services independently
   - Services: `postgres`, `qdrant`

2. `docker/docker-compose.yml`: Contains application services
   - Use this to run the application or ingestion
   - Services: `app`, `ingest`
   - Also includes postgres and qdrant, but these conflict with the root compose file

When using both files:
- Start databases with: `docker compose up -d` (uses root file)
- Run ingestion with: `docker compose -f docker/docker-compose.yml run --rm --no-deps ingest`
- Start app with: `docker compose -f docker/docker-compose.yml up app`

The `--no-deps` flag prevents starting duplicate services.

## Configuration

Configuration is managed via environment variables and `config.py`. Default settings work for local development.

### Database Configuration

Default PostgreSQL settings (root docker-compose.yml):
- Host: localhost
- Port: 5432
- User: patentuser
- Password: patentpass
- Database: patentsphere

Default Qdrant settings:
- Host: localhost
- Port: 6333
- Collection: patents

### Ollama Configuration

Default Ollama settings:
- Base URL: http://localhost:11434
- Router Model: phi4-mini:latest
- Extractor Model: qwen2.5:1.5b-instruct
- Synthesizer Model: gemma3:4b
- Critic Model: llama3.2:3b

### Docker Environment Variables

The `docker/docker-compose.yml` uses different database credentials:
- User: patents
- Password: patents
- Database: patents

If you need to use the docker compose app service, ensure your data was ingested using the same credentials, or update the environment variables in `docker/docker-compose.yml` to match your root compose file.

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
├── docker/
│   ├── docker-compose.yml  # Application and ingestion services
│   ├── Dockerfile          # Application container definition
│   ├── entrypoint.sh       # Container entrypoint script
│   └── ingest.sh           # Data ingestion script
├── requirements.txt        # Python dependencies
├── AGENTS.md              # Agent behavior definitions
├── app.py                 # Chainlit frontend
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

- Dense Vectors: `snowflake-arctic-embed-m` for semantic similarity (or `sentence-transformers/all-MiniLM-L6-v2` for 384-dim collections)
- Sparse Vectors: SPLADE via `fastembed` for exact keyword matching
- Combines both for optimal retrieval of patent numbers and technical terms

### Parallel Retrieval

Vector search and metadata lookup execute concurrently using `asyncio.gather()`, reducing latency by approximately 40%.

### Citation Format

All citations use the format: `[[PatentID]](URL)`

Example: `[[US9876543]](https://patents.google.com/patent/US9876543)`

### RLAIF Loop

The critic agent validates:
1. No hallucinated claims (all facts must be in context)
2. Correct citation format
3. All cited patent IDs exist in provided context

If validation fails, the system retries synthesis with feedback (max 3 attempts).

## Troubleshooting

### Port Already in Use

If you see "port is already allocated" errors:
- Check what's using the port: `docker ps --filter "publish=5432"` or `docker ps --filter "publish=6333"`
- Stop conflicting containers: `docker compose down` or `docker stop <container-name>`

### Ollama Connection Issues

If the application can't connect to Ollama:
- Ensure Ollama is running: `ollama serve`
- Check Ollama is accessible: `curl http://localhost:11434/api/tags`
- Verify models are installed: `ollama list`

### Database Connection Issues

If you can't connect to databases:
- Verify services are running: `docker compose ps`
- Check service logs: `docker compose logs postgres` or `docker compose logs qdrant`
- Ensure you're using the correct credentials (check config.py vs docker-compose.yml)

### Ingestion Errors

If ingestion fails:
- Ensure data files exist in the `data/` directory
- Check file formats match expected JSONL structure
- Verify database services are running and accessible
- Check logs: `docker compose -f docker/docker-compose.yml logs ingest`

## License

MIT

## Contributing

Contributions welcome! Please ensure all tests pass before submitting PRs.
