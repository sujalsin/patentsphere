# PatentSphere

PatentSphere is a multi-agent RAG system for patent intelligence.  
This repository currently contains:

- Configuration / safeguards for local vs. GCP deployments
- Preprocessing pipeline for patent JSONL data (chunking + embeddings)
- Project proposal and two-week execution plan

## Environment Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` pins the PyTorch + sentence-transformer stack used by the preprocessing script.  
On Apple Silicon, install the CPU wheel via `pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cpu`.

Verify you're in the safe `local_dev` profile before proceeding:

```bash
python scripts/check_profile.py
```

### Fetching a 10K Patent Subset (BigQuery)

1. Ensure `config/config.yaml` has your `GCP_PROJECT_ID` and `GCP_CREDENTIALS_PATH`.
1. Explicitly allow BigQuery exports:

```bash
export GCP_ALLOW_BIGQUERY_EXPORTS=true
```

1. Run a dry run (default) to confirm cost:

```bash
python scripts/fetch_bigquery_subset.py --limit 10000
```

1. When ready, execute the query (charges apply, still within guardrails):

```bash
python scripts/fetch_bigquery_subset.py \
  --limit 10000 \
  --output data/patents_10k.jsonl \
  --execute
```

The script refuses to run unless the `gcp.usage_policy.allow_bigquery_exports` flag is enabled and the env var above is set, preventing accidental credit usage.

## Preprocessing Pipeline

The script `scripts/process_patents.py` converts Google Patents JSONL files into normalized chunks and (optionally) generates embeddings using PyTorch-backed `sentence-transformers`.

### CLI Flags

| Flag | Description |
|------|-------------|
| `--input` | Path to JSONL (defaults to profile dataset, e.g. `data/patents_50k.jsonl`). |
| `--output-dir` | Destination for `chunks.jsonl`, `embeddings.pt`, `stats.json`. |
| `--max-patents` | Safety cap on the number of patents processed. |
| `--generate-embeddings` | Enable PyTorch embedding generation. |
| `--embedding-model` | SentenceTransformer model name. |
| `--embedding-batch-size` | Batch size for inference. |
| `--device {auto,cpu,cuda}` | Device selector (`auto` picks CUDA when available). |
| `--embedding-num-workers` | Passed to `SentenceTransformer.encode` for CPU multi-processing. |
| `--use-pyspark` | Use PySpark for distributed parallel embedding generation (faster for large datasets). |
| `--spark-partitions` | Number of Spark partitions (default: auto based on CPU count). |

### Example Usage

```bash
# Dry-run parsing only
python scripts/process_patents.py --dry-run

# Full local processing + embeddings (CPU fallback, single-process)
python scripts/process_patents.py \
  --output-dir data/processed \
  --generate-embeddings \
  --device auto \
  --embedding-num-workers 4

# Parallel processing with PySpark (faster for 10K+ chunks)
pip install pyspark  # Install PySpark first
python scripts/process_patents.py \
  --output-dir data/processed \
  --generate-embeddings \
  --device cpu \
  --use-pyspark \
  --spark-partitions 8
```

The embeddings step supports two modes:  

- **Standard mode**: Uses PyTorch tensors with `sentence-transformers` built-in multiprocessing (`--embedding-num-workers`)  
- **PySpark mode**: Distributes chunks across Spark partitions for maximum parallelism (`--use-pyspark`). Recommended for datasets with 10K+ chunks.

Outputs:

- `chunks.jsonl` – metadata payloads for Postgres + Qdrant
- `embeddings.pt` – torch tensor saved via `torch.save` (matches chunk order)
- `stats.json` – processing counts/timestamp

## Local Ingestion

Start the local databases:

```bash
docker-compose up -d postgres qdrant
```

Then load the processed data:

```bash
python scripts/ingest_local.py \
  --chunks data/processed/chunks.jsonl \
  --embeddings data/processed/embeddings.pt
```

The script ensures Postgres has a `patent_chunks` table and upserts the same payloads into Qdrant (vector DB).

## Testing

```bash
python -m pytest tests
```

Tests cover the chunking helpers plus a CLI dry-run smoke test using `tests/data/sample_patents.jsonl`.

## FastAPI Prototype (Phase 2)

The `app/` package now exposes a baseline FastAPI server with guarded endpoints:

```bash
source .venv/bin/activate
uvicorn app.main:app --reload
```

Endpoints (GET):

- `/health` – basic status probe (profile-aware)
- `/whoami` – asserts `local_dev` and shows dataset + agent limits
- `/retrieve?query=...` – runs `CitationMapperAgent`
- `/query?query=...` – runs the async orchestrator (Claims, Citation, Litigation, Synthesis)

All endpoints inherit the runtime guard: if the profile isn’t `local_dev`, they raise until we explicitly enable cloud usage.

### Claims Analyzer LLM Setup

The `ClaimsAnalyzerAgent` now calls an Ollama-hosted Mistral model to extract CPC codes and technical features.

1. Install [Ollama](https://ollama.ai) locally and pull the model listed in `config/config.yaml` (default `mistral:7b`):
   ```bash
   ollama pull mistral:7b
   ```
2. Ensure the Ollama service is running (default `http://localhost:11434`). Override via `export OLLAMA_HOST=http://host.docker.internal:11434` when the API runs inside Docker.
3. Optional: map alternate models per profile by editing `llm.agent_models.claims_analyzer` and the corresponding block under `llm.models`.

Each ClaimsAnalyzer run logs its output (CPC predictions, feature bullets, assumptions) to the `query_claims_analysis` table in Postgres for downstream agents (Synthesis, Critic, RL). If the LLM is unavailable, the agent falls back to a keyword-based heuristic and still records the attempt with `used_fallback=true`.

### Synthesis Agent LLM Setup

The `SynthesisAgent` consumes all upstream agent outputs and generates an executive summary/action plan via the `llm.agent_models.synthesis` model (default `llama3:8b` in `config/config.yaml`). Configuration steps mirror the claims agent:

1. `ollama pull llama3:8b` (or whichever model name you assign under `llm.models`).
2. Ensure the service is reachable at the host configured in `OLLAMA_HOST`.
3. Run `/query`—the response now includes:
   - `executive_summary`
   - `action_items` with priority/recommendation/rationale
   - Structured `citations`, `risk_score`, and `notes`

When the LLM fails, the agent falls back to a claims-only summary and flags `source=heuristic` so the frontend and Critic can detect degraded runs.

## Streamlit Control Room

The Streamlit companion app (`streamlit_app/app.py`) provides a richer console for querying the orchestrator, visualizing retrieved chunks, and drilling into agent logs.

```bash
streamlit run streamlit_app/app.py
```

Features:

- Query box + run history (stored locally in the Streamlit session)
- Agent status cards with latency + error surfacing
- Rich synthesis view (executive summary, action items, citations)
- Retrieval browser with expandable chunk previews
- Raw payload + log inspector per agent

The Streamlit process runs in the same environment as FastAPI and reuses the in-process orchestrator, so make sure Postgres/Qdrant/Ollama are reachable before launching the UI.

## Litigation Data Setup

### BigQuery Credentials

1. Create a GCP service account with BigQuery access:
   - Go to [GCP Console](https://console.cloud.google.com/iam-admin/serviceaccounts)
   - Create a new service account
   - Grant roles: "BigQuery Data Viewer" and "BigQuery Job User"
   - Download JSON key file
   - Save as `patentsphere.json` in project root

2. Set environment variable (optional):
   ```bash
   export GCP_CREDENTIALS_PATH=./patentsphere.json
   export GCP_PROJECT_ID=your-project-id
   ```

### Research USPTO Litigation Schema

Explore the USPTO OCE litigation dataset structure:

```bash
python scripts/research_litigation_schema.py --dataset uspto_oce_litigation
```

This will show available tables, schemas, and cost estimates.

### Fetch Litigation Data

1. Run dry-run to estimate costs:
   ```bash
   export GCP_ALLOW_BIGQUERY_EXPORTS=true
   python scripts/fetch_litigation_data.py --use-db-patents
   ```

2. Execute the query (when ready):
   ```bash
   python scripts/fetch_litigation_data.py \
     --use-db-patents \
     --output data/litigation_data.jsonl \
     --execute
   ```

3. Load litigation data into PostgreSQL:
   ```bash
   python scripts/ingest_local.py \
     --chunks data/processed/chunks.jsonl \
     --embeddings data/processed/embeddings.pt \
     --citations data/citations.jsonl \
     --litigation data/litigation_data.jsonl
   ```

## End-to-End Testing

Run comprehensive end-to-end tests:

```bash
# Start services
docker compose up -d postgres qdrant

# Run test suite
python scripts/test_fastapi_endtoend.py

# Or test specific components
python scripts/test_fastapi_endtoend.py --skip-docker  # If services already running
python scripts/test_fastapi_endtoend.py --skip-api    # Test orchestrator only
```

The test suite verifies:
- Docker services (PostgreSQL, Qdrant)
- Database connectivity and data loading
- All FastAPI endpoints (`/health`, `/whoami`, `/retrieve`, `/query`)
- All agents (ClaimsAnalyzer, CitationMapper, LitigationScout, Synthesis, CriticAgent)
- Response times and error handling

## Reinforcement Learning (RLAIF)

PatentSphere uses Reinforcement Learning from AI Feedback (RLAIF) to optimize retrieval depth and quality through the `AdaptiveRetrievalAgent`.

### AdaptiveRetrievalAgent

The `AdaptiveRetrievalAgent` uses Q-learning to adaptively determine retrieval depth based on query type and context. It replaces the standard `CitationMapperAgent` when enabled.

**State Space:**
- Query type (from ClaimsAnalyzerAgent)
- Retrieval depth (current iteration)
- Cumulative reward
- Chunk quality (average similarity score)

**Actions:**
- `RETRIEVE`: Initial retrieval
- `RETRIEVE_MORE`: Additional retrieval iteration
- `STOP`: End retrieval

**Configuration:**
Enable in `config/config.yaml`:
```yaml
adaptive_retrieval:
  enabled: true
  policy_path: "models/policy.pkl"
  exploration_rate: 0.1
```

### Training the RL Policy

Train the Q-learning policy using CriticAgent rewards:

```bash
# Train on 1000 synthetic queries
python scripts/train_rl_policy.py --episodes 1000 --save-frequency 100

# Options:
# --episodes: Number of training episodes (default: 1000)
# --batch-size: Batch size for updates (default: 32)
# --save-frequency: Save checkpoint every N episodes (default: 100)
# --validation-split: Fraction for validation set (default: 0.2)
```

The training script:
1. Replays logged adaptive runs stored in Postgres (`adaptive_retrieval_events` joined with `rl_experiences`) to warm up the policy.
2. Generates synthetic queries
3. Runs queries through the orchestrator
4. Collects CriticAgent rewards
5. Updates Q-values using Q-learning
6. Saves policy checkpoints

Telemetry and rewards are persisted automatically:
- Each AdaptiveRetrievalAgent run writes per-iteration telemetry to `adaptive_retrieval_events` (state vector, action, chunk coverage, exploration rate).
- CriticAgent scores are recorded in `rl_experiences` alongside sanitized chunk metadata and the telemetry `run_id`, enabling offline replay via `train_rl_policy.py`.

Policy is saved to `models/policy.pkl` and automatically loaded on agent initialization.

### RLAIF Evaluation

Evaluate the trained policy against baseline:

```bash
# Run RLAIF evaluation on 100 queries
python evaluation/rlaif_runner.py --queries evaluation/data/baseline_queries.json --limit 100

# Compare baseline vs RLAIF
python evaluation/compare_baseline_rlaif.py \
  --baseline evaluation/baseline_scores.json \
  --rlaif evaluation/rlaif_scores.json \
  --output evaluation/comparison_report.json
```

The comparison script performs:
- Paired t-test for statistical significance
- Effect size calculation (Cohen's d)
- Confidence intervals
- Improvement percentage calculations

### Human Evaluation

Generate evaluation pairs for human grading:

```bash
# Generate evaluation pairs (baseline + RLAIF)
python evaluation/human_eval_runner.py \
  --queries evaluation/human_eval_selected.json \
  --output-dir evaluation/human_eval

# Aggregate human scores
python evaluation/human_eval_runner.py \
  --scores evaluation/human_eval/scores.json
```

## Next Steps

The next milestone is wiring the processed chunks and embeddings into Qdrant + PostgreSQL (Week 1 Days 2–3) while honoring the cost guardrails defined in `config/config.yaml`. See `docs/proposal.md` for the full two-week plan.
