# DataLens

**Local-first data analysis and AI engineering platform.**

DataLens is an end-to-end capstone project designed to demonstrate the engineering of a production-oriented AI system rather than a standalone notebook or chatbot.

It combines deterministic data analysis, controlled data preparation, machine learning, deep learning, retrieval-augmented generation, local LLM integration, evaluation, observability, security boundaries, model lifecycle governance, and containerized delivery behind a unified product interface.

The project is intentionally built around one principle:

> AI assists the analytical workflow, but deterministic computation, explicit contracts, evidence, validation, and reproducibility remain authoritative.

---

## Why DataLens exists

Many AI demonstrations stop at model inference.

DataLens explores the larger engineering problem:

- How should tabular data be inspected before analysis?
- How can an AI system understand analytical intent without inventing computations?
- How can deterministic statistical execution remain separate from LLM interpretation?
- How should retrieved context be filtered and grounded?
- How can classical ML and neural models be trained, evaluated, persisted, monitored, and compared?
- How should model artifacts and provenance be governed?
- How can experimental model adaptation be evaluated without contaminating protected benchmarks?
- How can traces remain useful without leaking private payloads?
- How can the complete application be tested, containerized, published, and released?

The result is a deliberately layered system in which AI capabilities are surrounded by contracts, evidence, validation gates, and explicit lifecycle decisions.

---

## Core capabilities

### Deterministic analytics

DataLens includes a deterministic analytical engine for structured datasets.

The analytical workflow covers areas such as:

- descriptive statistics;
- grouped analysis;
- categorical analysis;
- correlations and relationships;
- ranking and prioritization;
- time-series analysis;
- visualization payload generation;
- report construction.

The LLM does not replace these computations. Analytical results are produced by controlled execution paths.

### Data preparation

The preparation workflow follows the logic of a real data-analysis process:

- import;
- quality review;
- semantic review;
- preparation planning;
- context enrichment;
- ambiguity resolution;
- cleaning;
- transformations;
- dataset combination;
- validation;
- readiness decisions;
- validated handoff to analysis.

Preparation state is explicitly represented and validated rather than hidden inside a single AI prompt.

### Semantic AI and local LLM integration

DataLens can use a local LLM for controlled semantic tasks and grounded explanations.

The system includes:

- structured AI contracts;
- semantic review;
- analytical request interpretation;
- controlled planning;
- tool-oriented orchestration;
- guarded model payloads;
- local Ollama integration;
- explicit AI egress boundaries.

The architecture is local-first and does not require sending dataset contents to a hosted LLM service.

### Retrieval-Augmented Generation

The RAG subsystem separates retrieval from acceptance and explanation.

It includes dedicated components for:

- retrieval;
- contextualization;
- relevance decisions;
- grounded explanation;
- evidence-aware behavior.

Retrieved material is not automatically treated as authoritative context.

### Classical machine learning

The Model Lab supports controlled machine-learning workflows including:

- regression;
- classification;
- preprocessing contracts;
- holdout strategies;
- group-aware evaluation;
- time-aware evaluation;
- purged group/time holdouts;
- model comparison;
- tuning and selection evidence;
- artifact persistence;
- model loading;
- prediction;
- monitoring;
- drift evaluation;
- performance monitoring;
- model-health evaluation;
- monitoring alerts.

Model artifacts are registered and consumed through controlled storage and loading paths.

### Deep learning

DataLens also contains dedicated neural-model workflows rather than treating deep learning as a label around an LLM.

Implemented areas include:

- tabular neural models;
- autoencoder-based anomaly detection;
- autoencoder training and evaluation;
- neural model bundles;
- trusted model loading;
- time-series MLP models;
- recurrent neural networks;
- LSTM models;
- neural time-series execution;
- neural model comparison;
- production artifact registration.

These workflows integrate with the same broader Model Lab and artifact-governance principles used by the classical ML stack.

### LLM adaptation

DataLens includes an experimental local LLM adaptation pipeline built around a frozen base model and PEFT/QLoRA adapter artifacts.

The adaptation work demonstrates:

- dataset and contract preparation;
- quantized base-model loading;
- adapter training;
- frozen training authorities;
- artifact hashing;
- provenance;
- protected evaluation;
- label-blind model boundaries;
- preregistered promotion gates;
- independent post-training evaluation.

The latest official QLoRA v0.4 independent evaluation completed successfully as an experiment, but the candidate was **not promoted**.

The adapted candidate achieved a material improvement over the base model, while still failing the preregistered absolute promotion criteria.

Official lifecycle state:

```text
evaluation_status: completed
promotion_decision: not_promoted
```

The protected Hospital benchmark was consumed exactly once and must never be replayed for that candidate.

This distinction is intentional: an experiment can produce useful learning without being declared production-ready.

### Model lifecycle governance

DataLens implements explicit model identity and provenance instead of relying only on filenames.

Lifecycle capabilities include:

- deterministic artifact identity;
- provenance identity;
- artifact fingerprints;
- trusted artifact resolution;
- metadata registries;
- evaluation-decision contracts;
- promotion decisions;
- portable governance evidence.

The lifecycle layer is designed to separate:

1. what a model artifact is;
2. where it came from;
3. how it was evaluated;
4. whether it is eligible for promotion.

### Observability

Runtime and AI behavior can be inspected through structured traces.

The observability layer includes:

- request context;
- runtime traces;
- AI traces;
- trace persistence;
- trace exploration.

Privacy boundaries are tested so observability does not become an uncontrolled raw-data logging channel.

### Security and privacy boundaries

DataLens includes explicit controls around AI payloads and runtime exposure.

Existing controls include:

- local-first LLM execution;
- restricted frontend CORS;
- AI egress contracts;
- controlled LLM payload construction;
- semantic value sampling boundaries;
- sanitized public errors;
- private-detail suppression in traces and APIs;
- trusted artifact loading;
- fail-closed validation in critical governance paths.

Security is treated as a system property rather than a single middleware component.

### Reporting

The reporting pipeline turns deterministic analytical outputs into user-facing findings and reports.

It includes controlled report selection, structured findings, visual payloads, and PDF-oriented reporting flows while preserving the distinction between computed evidence and generated explanation.

---

## Architecture at a glance

DataLens is organized as a monorepo.

```text
datalens/
|
|-- apps/
|   |-- api/                    Python / FastAPI backend
|   |   |-- app/
|   |   |   |-- analysis/
|   |   |   |-- preparation/
|   |   |   |-- planning/
|   |   |   |-- execution/
|   |   |   |-- evals/
|   |   |   |-- evaluation/
|   |   |   |-- ml/
|   |   |   |-- deep_learning/
|   |   |   |-- adaptation/
|   |   |   |-- model_lifecycle/
|   |   |   |-- observability/
|   |   |   |-- security/
|   |   |   `-- reporting/
|   |   |
|   |   |-- artifacts/
|   |   |-- tests/
|   |   |-- Dockerfile
|   |   `-- requirements.txt
|   |
|   `-- web/                    Next.js frontend
|       |-- src/
|       |-- public/
|       |-- Dockerfile
|       `-- package.json
|
|-- .github/
|   `-- workflows/
|       |-- datalens-evals.yml
|       |-- datalens-runtime.yml
|       |-- datalens-publish.yml
|       `-- datalens-release.yml
|
|-- compose.yaml
`-- compose.registry.yaml
```

The main product surfaces are the analytical workspace, preparation workflow, reporting workflow, and Model Lab.

---

## Technology stack

### Backend

- Python 3.13
- FastAPI
- Pydantic
- pandas
- NumPy
- SciPy
- scikit-learn
- Ollama Python client
- SQLite-backed persistence where appropriate
- ReportLab / python-docx / pypdf for document and reporting workflows

### Frontend

- Next.js 16
- React 19
- TypeScript
- App Router

### AI / ML

- deterministic statistical execution;
- classical scikit-learn workflows;
- dedicated deep-learning execution paths;
- local LLM inference through Ollama;
- PEFT/QLoRA adaptation experiments;
- RAG and semantic relevance controls;
- evaluation and regression gates.

### Delivery

- Docker
- Docker Compose
- GitHub Actions
- container publishing workflows
- release workflows
- runtime and evaluation CI gates

---

## Quick start with Docker

Docker Compose is the recommended way to run the standard local application stack.

### Prerequisites

- Docker with Docker Compose support
- a local Ollama runtime for LLM-backed features

By default, the containers expose only loopback interfaces.

```text
Web: http://127.0.0.1:3000
API: http://127.0.0.1:8000
```

FastAPI interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

### Start the stack

```bash
docker compose up --build
```

The standard Compose configuration:

- builds the API image from `apps/api/Dockerfile`;
- builds the web image from `apps/web/Dockerfile`;
- persists backend runtime state in a Docker volume;
- connects the API container to the host Ollama runtime;
- configures the frontend to call the local API.

### Stop the stack

```bash
docker compose down
```

Runtime state is stored in the named API volume and is therefore not deleted by a normal `docker compose down`.

---

## Local development

### Backend

```bash
cd apps/api
python -m venv .venv
```

Activate the virtual environment for your platform, then install the backend dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the API:

```bash
python -m uvicorn app.main:app --reload
```

### Frontend

```bash
cd apps/web
npm ci
npm run dev
```

The frontend development server is available at:

```text
http://localhost:3000
```

The public API base URL can be configured through:

```text
NEXT_PUBLIC_DATALENS_API_URL
```

---

## Configuration

DataLens uses environment variables for runtime paths and optional execution bridges.

Important examples include:

```text
DATALENS_OLLAMA_HOST
DATALENS_LLM_DOCKER_BRIDGE_ENABLED
DATALENS_ML_MODEL_ARTIFACT_STORE_PATH
DATALENS_MODEL_LIFECYCLE_REGISTRY_PATH
DATALENS_MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_PATH
DATALENS_AI_TRACE_ENABLED
DATALENS_DL_PYTHON_PATH
NEXT_PUBLIC_DATALENS_API_URL
```

A versioned environment reference is available in [`.env.example`](.env.example).

Do not commit secrets, local model files, runtime databases, or private datasets.

---

## Evaluation philosophy

DataLens treats evaluation as part of the product architecture.

The project uses several kinds of evidence:

- deterministic unit and integration tests;
- analytical regression tests;
- AI evaluation scenarios;
- model holdout evaluation;
- ML monitoring evidence;
- model-health checks;
- protected post-training benchmarks;
- CI evaluation gates.

For protected model evaluation, results are not treated as an invitation to repeatedly tune against the same holdout.

The independent Hospital benchmark used by the QLoRA v0.4 experiment is single-use evidence.

```text
CONSUMED / NEVER REPLAY
```

Any future adapted candidate that requires independent validation must use new independent benchmark evidence.

---

## CI/CD

The repository contains four main GitHub Actions workflows:

```text
datalens-evals.yml
datalens-runtime.yml
datalens-publish.yml
datalens-release.yml
```

Together they cover evaluation gates, runtime validation, container build/publishing flows, and controlled releases.

The release model is intentionally separated from experimental model-promotion decisions.

Passing application CI does not automatically promote an experimental ML or LLM candidate.

---

## Engineering principles

DataLens follows several recurring design rules.

### Deterministic computation before generated explanation

Numerical answers should come from deterministic execution whenever possible.

### Structured contracts over free-form coupling

Boundaries between preparation, analysis, ML, AI, monitoring, and reporting use explicit contracts.

### Server-owned evidence

Clients do not get to invent trusted evaluation or provenance evidence.

### Fail closed on ambiguous authority

Critical lifecycle and artifact-validation paths reject inconsistent identity, provenance, or evidence.

### Local-first AI

Local model execution is preferred for privacy-sensitive analytical workflows.

### Protected evaluation is evidence, not training data

Independent holdouts are not repeatedly consumed after their results are known.

### Observability without raw-data leakage

Traces should explain system behavior without becoming a second uncontrolled dataset.

---

## What this project demonstrates

DataLens is intended as an AI Engineering capstone demonstrating the integration of:

- backend engineering;
- deterministic analytics;
- data preparation;
- statistical reasoning;
- ML engineering;
- deep learning;
- LLM integration;
- RAG;
- structured outputs;
- orchestration;
- model adaptation;
- evaluation engineering;
- model lifecycle governance;
- observability;
- security;
- Docker;
- CI/CD;
- product-facing frontend integration.

The focus is not on maximizing the number of AI features.

The focus is on showing how AI capabilities can be made testable, inspectable, reproducible, and governable inside a real application.

---

## Documentation

The capstone documentation is split by concern:

- [Architecture](ARCHITECTURE.md) — system boundaries, runtime topology, analytical, AI, ML/DL, persistence, lifecycle, and CI/CD architecture.
- [Security](SECURITY.md) — current threat model, implemented privacy/security controls, trust assumptions, and known hardening gaps.
- [AI Engineering Case Study](AI_ENGINEERING_CASE_STUDY.md) — recruiter-facing explanation of the engineering problem, decisions, evidence, and capstone outcomes.
- [Demo Walkthrough](DEMO_WALKTHROUGH.md) — repeatable 3-minute, 10-minute, and technical demo paths.
- [Backend README](apps/api/README.md) — FastAPI backend domains, runtime, local development, testing, Model Lab, observability, and lifecycle notes.
- [Frontend README](apps/web/README.md) — Next.js workspace, preparation, analysis, Model Lab, monitoring, observability, and frontend development notes.
- [Environment reference](.env.example) — documented runtime configuration without secrets.

---

## Project status

DataLens is an actively developed capstone project.

The core analytical, AI, ML, deep-learning, evaluation, observability, lifecycle, security, frontend, container, and CI/CD foundations are implemented.

The capstone documentation set includes:

- project-level documentation;
- architecture documentation;
- reproducible environment configuration;
- a documented security threat model;
- a recruiter-facing AI Engineering case study;
- a repeatable demonstration walkthrough;
- backend-specific runtime and development documentation;
- frontend-specific workspace and development documentation.

The capstone documentation consolidation, coherence audit, CI validation, and repository integration are complete for this baseline.

Future work can focus on product evolution, optional security hardening, portfolio presentation, and new independently governed experiments rather than missing documentation foundations.

No protected benchmark is replayed as part of documentation maintenance.
