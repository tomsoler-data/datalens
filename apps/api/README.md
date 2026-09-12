# DataLens API

Backend service for the DataLens local-first analytics and AI engineering platform.

The API is built with FastAPI and coordinates deterministic analytics, data preparation, document ingestion and RAG, local LLM workflows, reporting, classical machine learning, deep learning, monitoring, observability, and model lifecycle governance.

The backend is intentionally designed so that:

- analytical computation remains deterministic where possible;
- trusted workflow state remains server-owned;
- LLM access is local-first and privacy-constrained;
- model artifacts are persisted and loaded through controlled paths;
- evaluation evidence is explicit;
- observability avoids raw payload logging;
- model promotion is separate from application deployment.

---

## 1. Runtime

The standard backend runtime uses:

```text
Python 3.13
FastAPI
Pydantic
Uvicorn
SQLite
pandas
NumPy
SciPy
scikit-learn
Ollama Python client
```

The container image is based on:

```text
python:3.13-slim
```

The service listens on:

```text
http://127.0.0.1:8000
```

Interactive FastAPI documentation is available at:

```text
http://127.0.0.1:8000/docs
```

---

## 2. Main application

The FastAPI composition root is:

```text
app/main.py
```

It assembles routers for major product capabilities including:

- analysis;
- requested resolution;
- document ingestion;
- preparation;
- reporting;
- model training;
- Model Lab;
- ML monitoring;
- performance monitoring;
- model health;
- monitoring alerts.

Runtime tracing is installed as middleware.

---

## 3. Backend domains

The backend is organized by capability.

```text
app/
|
|-- adaptation/
|-- ai/
|-- analysis/
|-- api/
|-- cleaning/
|-- core/
|-- dashboard/
|-- deep_learning/
|-- discovery/
|-- evals/
|-- evaluation/
|-- evidence/
|-- execution/
|-- ingestion/
|-- ml/
|-- model_lifecycle/
|-- models/
|-- observability/
|-- persistence/
|-- planning/
|-- preparation/
|-- profiling/
|-- ranking/
|-- relationships/
|-- reporting/
|-- security/
|-- semantics/
|-- statistics/
`-- visualization/
```

These directories are not independent microservices.

They are capability boundaries inside one FastAPI backend.

---

## 4. Data preparation

Preparation is implemented as a server-owned multi-stage workflow.

Representative stages:

```text
IMPORT
UNDERSTAND
QUALITY
CLEAN
TRANSFORM
COMBINE
VALIDATE
```

The backend owns:

- workflow identity;
- root dataset scope;
- workflow revision;
- stage state;
- output dataset scope;
- validated handoff.

The browser receives a controlled read model rather than submitting complete authoritative workflow state.

This prevents client-side state from becoming trusted preparation authority.

---

## 5. Deterministic analytics

The analysis stack is designed so that numerical results are computed by deterministic Python execution rather than generated directly by an LLM.

Typical responsibilities include:

- descriptive statistics;
- grouped metrics;
- rankings;
- comparisons;
- categorical analysis;
- correlations;
- time-series analysis;
- chart payload generation;
- reporting evidence.

The preferred execution chain is:

```text
request
  |
intent / planning
  |
validated plan
  |
deterministic execution
  |
structured evidence
  |
visualization / explanation / report
```

---

## 6. Planning and execution

Planning and execution are separate concerns.

The planning layer resolves analytical intent and requested operations.

The execution layer performs controlled computation.

This separation allows:

- validation before execution;
- deterministic tests;
- clearer observability;
- safer LLM integration;
- reproducible results.

---

## 7. Document ingestion and RAG

DataLens supports document ingestion for:

```text
.pdf
.docx
.txt
.md
```

The ingestion layer extracts and normalizes text, creates source-aware chunks, and produces structured manifests.

The RAG pipeline separates:

```text
ingestion
retrieval
relevance
context
explanation
```

Relevant modules include:

```text
app/rag.py
app/rag_retrieval.py
app/rag_relevance.py
app/rag_context.py
app/rag_explanation.py
```

Retrieved content is not automatically accepted as authoritative context.

---

## 8. Local LLM integration

Local model access is resolved through the Ollama runtime.

Primary configuration:

```text
DATALENS_OLLAMA_HOST
```

Local default:

```text
http://localhost:11434
```

Docker can explicitly enable the host Ollama bridge through:

```text
DATALENS_LLM_DOCKER_BRIDGE_ENABLED=1
```

The backend validates the model-service destination before network I/O.

---

## 9. LLM privacy boundary

Model-visible payloads are classified.

Allowed classes include:

```text
metadata_only
deterministic_evidence
semantic_value_sample
document_content
```

Raw tabular rows are forbidden:

```text
tabular_raw_rows
```

When actual values are required for semantic interpretation, the semantic sample budget is bounded.

Current maximum:

```text
5 values
```

---

## 10. Classical machine learning

The `app/ml` domain contains the classical ML stack.

Responsibilities include:

- estimator contracts;
- preprocessing;
- model training;
- holdout strategies;
- diagnostics;
- tuning;
- comparison;
- artifact persistence;
- prediction;
- monitoring;
- drift analysis;
- performance monitoring;
- model health;
- monitoring alerts.

The backend supports more than a single train/predict endpoint.

The model lifecycle continues after training.

---

## 11. Model artifact store

Persisted ML artifacts are registered through controlled artifact-store logic.

Primary optional configuration:

```text
DATALENS_ML_MODEL_ARTIFACT_STORE_PATH
```

Default relative location:

```text
var/ml/model_artifacts.json
```

The artifact layer separates:

```text
training
artifact registration
trusted loading
prediction
evaluation
monitoring
```

Clients should not supply arbitrary model filesystem paths.

---

## 12. Model Lab API

The Model Lab router is exposed under:

```text
/model-lab
```

The service supports trusted operations such as:

- listing models;
- retrieving model details;
- evaluating a model;
- executing predictions.

Public API contracts intentionally avoid exposing:

- raw model bytes;
- estimator objects;
- internal filesystem paths;
- complete internal training contracts.

Cross-workflow artifact existence is not intentionally disclosed.

---

## 13. Deep learning

The `app/deep_learning` domain contains dedicated neural workflows.

Implemented areas include:

### Tabular neural models

- datasets;
- tensors;
- network definitions;
- training;
- evaluation;
- bundles;
- trusted loading;
- execution.

### Autoencoder anomaly detection

- dataset preparation;
- autoencoder network;
- training;
- evaluation;
- threshold selection;
- bundle creation;
- trusted loading;
- Model Lab execution.

### Neural time-series models

- MLP;
- RNN;
- LSTM;
- scaling;
- worker execution;
- bundles;
- inference;
- model comparison;
- artifact registration.

---

## 14. Deep-learning runtime

Neural time-series execution can use an isolated Python runtime.

Optional configuration:

```text
DATALENS_DL_PYTHON_PATH
```

When no override is provided, the backend resolves a project-local `.venv-dl` environment.

Windows default pattern:

```text
apps/api/.venv-dl/Scripts/python.exe
```

POSIX default pattern:

```text
apps/api/.venv-dl/bin/python
```

Do not point this variable to an untrusted executable.

---

## 15. LLM adaptation

The `app/adaptation` domain contains experimental LLM adaptation infrastructure.

Implemented concepts include:

- adaptation contracts;
- QLoRA preparation;
- quantized base-model loading;
- adapter artifacts;
- frozen authorities;
- provenance;
- evaluation protocols;
- protected holdouts;
- scoring;
- official evaluation artifacts.

The adaptation subsystem is not the production default model path.

---

## 16. Official QLoRA lifecycle state

The latest official QLoRA v0.4 independent evaluation completed successfully as an evaluation process, but the candidate did not satisfy all preregistered promotion gates.

Official state:

```text
evaluation_status: completed
promotion_decision: not_promoted
```

The protected Hospital benchmark is:

```text
CONSUMED / NEVER REPLAY
```

It must not be replayed for the same candidate.

---

## 17. Model lifecycle governance

The `app/model_lifecycle` domain provides metadata-level model governance.

It includes:

- artifact contracts;
- fingerprints;
- provenance contracts;
- identity registry;
- trusted artifact resolution;
- evaluation-decision contracts;
- evaluation-decision registry;
- adaptation projections.

The lifecycle layer separates:

```text
artifact identity
provenance
evaluation evidence
promotion decision
```

---

## 18. Lifecycle configuration

Identity/provenance registry:

```text
DATALENS_MODEL_LIFECYCLE_REGISTRY_PATH
```

Default:

```text
var/model_lifecycle/registry.sqlite3
```

Evaluation decision registry:

```text
DATALENS_MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_PATH
```

Default:

```text
var/model_lifecycle/evaluation_decisions.sqlite3
```

These registries contain governance metadata and should not be treated as model binaries.

---

## 19. Observability

Backend observability includes:

```text
app/observability/request_context.py
app/observability/runtime_trace.py
app/observability/ai_trace.py
app/observability/trace_store.py
app/observability/trace_explorer.py
```

The runtime layer provides request correlation through:

```text
X-DataLens-Request-ID
```

---

## 20. Runtime trace privacy

Runtime traces intentionally avoid storing:

- request bodies;
- response bodies;
- query strings;
- incoming headers;
- client IP addresses;
- raw dataset rows;
- uploaded file contents;
- document chunks;
- exception messages;
- filesystem paths.

Runtime trace configuration:

```text
DATALENS_RUNTIME_TRACE_ENABLED
DATALENS_RUNTIME_TRACE_PATH
```

---

## 21. AI trace configuration

AI traces are enabled by default unless explicitly disabled.

Configuration:

```text
DATALENS_AI_TRACE_ENABLED
DATALENS_AI_TRACE_PATH
```

Accepted disabled values include:

```text
0
false
no
off
```

The observability goal is correlation and diagnostics without uncontrolled raw-data persistence.

---

## 22. Security modules

The dedicated security domain currently includes:

```text
app/security/llm_egress.py
app/security/llm_payload.py
app/security/semantic_value_sample.py
```

These modules enforce:

- local-only LLM egress;
- redirect rejection;
- proxy bypass;
- payload classification;
- raw-row prohibition;
- bounded semantic samples.

Security-related behavior also exists across API, observability, Model Lab, evaluation, and reporting layers.

See the root `SECURITY.md` for the full threat model.

---

## 23. Persistence

The backend uses several local persistence forms.

### SQLite

Used for state that benefits from transactional persistence and lookup.

### Filesystem artifacts

Used for:

- preparation artifacts;
- analysis/report artifacts;
- model artifacts;
- adaptation evidence;
- governance evidence.

### JSONL

Used for trace-oriented append storage.

The Docker API container persists runtime state beneath:

```text
/app/var
```

---

## 24. Docker runtime

Build the API image from the repository root through Docker Compose:

```bash
docker compose build api
```

Or directly:

```bash
docker build -t datalens-api:v0.1 apps/api
```

Run the full application:

```bash
docker compose up --build
```

The API container:

- runs as non-root;
- uses UID/GID `10001`;
- exposes port `8000`;
- includes a `/health` healthcheck;
- persists runtime state through the Compose volume.

---

## 25. Local development setup

From the repository root:

```bash
cd apps/api
```

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it using the appropriate command for your shell.

### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

### POSIX shell

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 26. Run the API locally

From `apps/api` with the virtual environment activated:

```bash
python -m uvicorn app.main:app --reload
```

Default development URL:

```text
http://127.0.0.1:8000
```

Health endpoint:

```text
http://127.0.0.1:8000/health
```

OpenAPI / Swagger UI:

```text
http://127.0.0.1:8000/docs
```

---

## 27. Ollama for local development

Start or install Ollama separately.

DataLens defaults to:

```text
http://localhost:11434
```

A custom local Ollama endpoint may be configured with:

```powershell
$env:DATALENS_OLLAMA_HOST = "http://localhost:11434"
```

Remote model endpoints are intentionally rejected by the local-first egress guard.

---

## 28. Environment reference

The repository root contains:

```text
.env.example
```

It documents the major runtime variables.

Do not place secrets or protected benchmark material in `.env.example`.

---

## 29. Tests

The backend has a broad automated test suite under:

```text
tests/
```

The final capstone gap audit counted:

```text
422 backend test files
```

Test areas include:

- preparation;
- deterministic analysis;
- planning;
- execution;
- RAG;
- reporting;
- ML;
- deep learning;
- monitoring;
- observability;
- security;
- adaptation;
- model lifecycle;
- CI/runtime contracts.

---

## 30. Running tests

The repository contains a mixture of focused executable test modules and pytest-compatible tests.

For pytest-based suites, run from `apps/api`:

```bash
python -m pytest
```

For a focused test module:

```bash
python -m pytest tests/path/to/test_file.py
```

Some historical and acceptance modules are also executed directly through:

```bash
python -m tests.package.test_module
```

Follow the relevant CI workflow or test file when a subsystem defines a specific execution contract.

---

## 31. CI gates

Backend behavior is validated through the repository workflows:

```text
.github/workflows/datalens-evals.yml
.github/workflows/datalens-runtime.yml
.github/workflows/datalens-publish.yml
.github/workflows/datalens-release.yml
```

These gates cover different concerns such as:

- evaluation regressions;
- runtime behavior;
- container builds;
- image publishing;
- releases.

Passing CI does not automatically promote an experimental model.

---

## 32. Error handling

Public API errors are intentionally separated from internal failures.

Examples include:

- sanitized Model Lab artifact errors;
- hidden cross-workflow model existence;
- controlled prediction failures;
- controlled evaluation failures;
- preparation AI error privacy;
- report error privacy.

The backend should not expose internal filesystem details merely because an internal exception contains them.

---

## 33. CORS

The current FastAPI CORS policy allows the local frontend origins:

```text
http://localhost:3000
http://127.0.0.1:3000
```

Credentials are disabled.

The current policy is designed for the local-first application topology.

---

## 34. Development invariants

When modifying the backend, preserve the following principles.

### Deterministic evidence

Do not replace deterministic analytics with generated estimates.

### Server-owned authority

Do not allow clients to manufacture trusted preparation or evaluation state.

### Local-first LLM transport

Do not bypass the local-only egress guard for convenience.

### Payload minimization

Do not send raw tabular rows to the model.

### Trusted model loading

Do not load arbitrary client-supplied model paths.

### Protected evaluation

Do not replay consumed independent benchmarks.

### Observability privacy

Do not store raw request or response payloads in runtime traces.

---

## 35. Protected evaluation invariant

The Hospital benchmark used by the QLoRA v0.4 independent evaluation is frozen evidence.

Current invariant:

```text
Hospital benchmark: CONSUMED / NEVER REPLAY
```

Do not:

- rerun it for the same candidate;
- use it for tuning;
- expose gold labels to the model;
- change promotion thresholds after observing results;
- mutate official result artifacts in place.

---

## 36. Current backend status

At the current capstone consolidation point:

```text
FastAPI application              IMPLEMENTED
Preparation workflow             IMPLEMENTED
Deterministic analytics          IMPLEMENTED
Planning / execution             IMPLEMENTED
Document ingestion               IMPLEMENTED
RAG                              IMPLEMENTED
Local LLM integration            IMPLEMENTED
Classical ML                     IMPLEMENTED
Deep learning                    IMPLEMENTED
Model Lab                        IMPLEMENTED
Monitoring                       IMPLEMENTED
Observability                    IMPLEMENTED
Security boundaries              IMPLEMENTED
QLoRA adaptation                 IMPLEMENTED
Independent evaluation           IMPLEMENTED
Model lifecycle governance       IMPLEMENTED
Docker runtime                   IMPLEMENTED
Backend automated testing        IMPLEMENTED
```

The backend foundations and capstone documentation baseline are implemented and integrated. Future backend work can focus on product evolution, additional hardening, new governed experiments, and maintenance rather than missing documentation or lifecycle foundations.
