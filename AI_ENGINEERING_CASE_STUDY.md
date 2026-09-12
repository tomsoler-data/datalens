# DataLens — AI Engineering Case Study

## Executive summary

DataLens is a local-first data analysis and AI engineering platform built to demonstrate how analytical software, machine learning, deep learning, retrieval-augmented generation, local LLMs, evaluation, observability, security, and model lifecycle governance can be integrated into one coherent application.

The project started from a data-analysis workflow and progressively evolved into a capstone focused on a harder engineering question:

> How can AI be added to a real analytical product without allowing generated text to become the source of truth?

The resulting architecture keeps deterministic computation, validated state, evaluation evidence, model identity, and lifecycle decisions under explicit server-side control while using AI for semantic interpretation, contextual assistance, grounded explanation, and model adaptation.

DataLens is not presented as a production SaaS platform. It is a technical capstone demonstrating production-oriented AI engineering patterns inside a local-first system.

---

# 1. Problem

A conventional data-analysis application can compute metrics, transform datasets, and generate charts.

An AI-enabled data-analysis application introduces new failure modes:

- an LLM can invent numerical answers;
- analytical intent can be misunderstood;
- retrieved context can be irrelevant;
- raw data can leak into model payloads or traces;
- trained models can lose provenance;
- evaluation evidence can be contaminated;
- a model can be treated as production-ready simply because training succeeded;
- observability can expose sensitive data;
- frontend state can accidentally become trusted backend state;
- experimentation can become impossible to reproduce.

The goal of DataLens was therefore not merely to “add AI”.

The goal was to build a system where AI capabilities are:

- constrained;
- testable;
- observable;
- reproducible;
- grounded;
- governed.

---

# 2. Product concept

DataLens combines four major workflows.

## 2.1 Data preparation

A user can move through a preparation workflow that reflects the work of a data analyst:

```text
IMPORT
UNDERSTAND
QUALITY
CLEAN
TRANSFORM
COMBINE
VALIDATE
```

The workflow is persisted and server-owned.

The browser does not define authoritative preparation state.

---

## 2.2 Deterministic analysis

Analytical requests are converted into validated execution paths.

The application can compute structured analytical outputs such as:

- descriptive statistics;
- grouped metrics;
- comparisons;
- rankings;
- correlations;
- categorical analysis;
- time-series analysis;
- chart-ready structures;
- report findings.

The core rule is:

```text
LLM interpretation != numerical authority
```

Numerical evidence is produced through deterministic code.

---

## 2.3 Model Lab

The Model Lab supports model-oriented workflows including:

- training;
- evaluation;
- model comparison;
- persistence;
- prediction;
- monitoring;
- drift analysis;
- performance evaluation;
- model-health evaluation;
- alerts.

The backend supports both classical ML and dedicated deep-learning workflows.

---

## 2.4 AI-assisted reasoning

Local AI capabilities support:

- semantic review;
- request interpretation;
- planning assistance;
- document understanding;
- RAG;
- relevance decisions;
- grounded explanations;
- model adaptation experiments.

The AI layer is deliberately surrounded by deterministic and contract-based boundaries.

---

# 3. System architecture

DataLens is a monorepo.

```text
datalens/
|
|-- apps/
|   |-- api/       FastAPI backend
|   `-- web/       Next.js frontend
|
|-- .github/
|   `-- workflows/
|
|-- compose.yaml
|-- compose.registry.yaml
|-- README.md
|-- ARCHITECTURE.md
|-- SECURITY.md
`-- .env.example
```

The standard local topology is:

```text
Browser
   |
   v
Next.js
   |
   v
FastAPI
   |
   +--> Preparation
   +--> Analytics
   +--> RAG
   +--> Reporting
   +--> Model Lab
   +--> Monitoring
   +--> Observability
   |
   v
Local Ollama
```

Runtime state is persisted locally.

The standard Docker configuration binds the application to loopback interfaces.

---

# 4. Key engineering decision: deterministic authority

The most important architectural choice in DataLens is the separation between:

```text
semantic interpretation
```

and:

```text
analytical truth
```

An LLM may help interpret a question such as:

> Which customer segment generated the strongest revenue growth?

But the actual values must come from deterministic execution.

A representative flow is:

```text
User request
    |
    v
Intent interpretation
    |
    v
Validated analysis plan
    |
    v
Python execution
    |
    v
Structured evidence
    |
    v
Explanation / visualization / report
```

This choice improves:

- reproducibility;
- debugging;
- evaluation;
- numerical consistency;
- trust.

It also creates a clear boundary for testing.

---

# 5. Data preparation as a state machine

Data preparation is not implemented as a single “clean my data” AI prompt.

It is represented as a workflow with explicit stages and state.

The backend owns:

- root dataset scope;
- workflow revision;
- stage state;
- analytical output scope;
- validation status.

This allows the system to reject stale decisions and avoid client-manufactured readiness.

A simplified preparation lifecycle is:

```text
Import
  |
Understand
  |
Quality
  |
  +--> Clean
  |
  +--> Transform
  |
  +--> Combine
  |
Validate
  |
Ready for analysis
```

Not every workflow requires every optional stage, but validation remains the trusted handoff into downstream analysis.

---

# 6. Structured AI instead of free-form coupling

DataLens uses explicit contracts across AI boundaries.

Rather than allowing generated prose to directly mutate system state, the system prefers:

- typed requests;
- typed results;
- validation;
- versioned rules;
- server-owned reconstruction;
- controlled execution.

This is especially important for:

- semantic decisions;
- preparation decisions;
- analytical planning;
- model evaluation;
- model lifecycle decisions.

The architecture therefore treats LLM output as input to a controlled system, not as the system itself.

---

# 7. Retrieval-Augmented Generation

DataLens contains a local RAG pipeline for documentary context.

The RAG architecture separates:

```text
ingestion
retrieval
relevance
context construction
explanation
```

Supported ingestion includes:

- PDF;
- DOCX;
- TXT;
- Markdown.

This separation matters because retrieved content is not automatically considered authoritative.

The system can reject or constrain context before it reaches the explanatory layer.

---

# 8. Privacy-aware local LLM design

DataLens uses local LLM execution through Ollama.

The system includes an explicit egress guard.

Allowed destinations are restricted to:

- `localhost`;
- loopback IP addresses;
- the explicit Docker Ollama bridge when enabled.

The system rejects:

- arbitrary remote hosts;
- redirects;
- inherited HTTP proxy routing;
- credentials embedded in model URLs;
- unexpected Docker bridge destinations.

The model boundary is therefore designed to remain local.

---

# 9. Model-visible data minimization

Local execution does not mean the model automatically receives the complete dataset.

DataLens classifies model-visible payloads.

Allowed categories include:

```text
metadata_only
deterministic_evidence
semantic_value_sample
document_content
```

Raw tabular rows are explicitly forbidden.

```text
tabular_raw_rows -> rejected
```

When semantic interpretation genuinely requires example values, the current semantic sample budget is limited to:

```text
5 values
```

This demonstrates a privacy principle that is often missing from local AI prototypes:

> Local execution should still minimize unnecessary data exposure.

---

# 10. Classical machine learning

The Model Lab includes classical ML capabilities around:

- regression;
- classification;
- preprocessing;
- holdout evaluation;
- group holdouts;
- time-based holdouts;
- purged group/time holdouts;
- tuning;
- model comparison;
- selection evidence;
- prediction.

Model training does not end at an in-memory estimator.

The project includes model artifact persistence and trusted loading.

This supports the broader lifecycle:

```text
training
  |
evaluation
  |
artifact persistence
  |
trusted loading
  |
prediction
  |
monitoring
```

---

# 11. Deep learning

DataLens contains dedicated deep-learning implementations rather than relying only on LLM fine-tuning.

Implemented areas include:

## Tabular neural models

- tensor preparation;
- neural-network definitions;
- training;
- evaluation;
- execution;
- bundles;
- trusted loading.

## Autoencoder anomaly detection

- autoencoder network;
- training;
- reconstruction evaluation;
- threshold selection;
- production artifact creation;
- trusted loading;
- Model Lab integration.

## Neural time-series models

- MLP;
- RNN;
- LSTM;
- scaling;
- worker execution;
- bundles;
- inference;
- model comparison;
- artifact registration.

This provides explicit deep-learning evidence inside the capstone beyond the LLM adaptation work.

---

# 12. Model artifacts and provenance

A model file is not enough to explain:

- what the model is;
- what trained it;
- which workflow owns it;
- which dataset context produced it;
- how it was evaluated;
- whether it was selected.

DataLens therefore tracks model artifact and provenance evidence.

The model lifecycle architecture separates:

```text
artifact identity
provenance
evaluation
promotion
```

This reduces reliance on filenames as implicit model identity.

---

# 13. QLoRA adaptation experiment

DataLens includes an experimental local LLM adaptation workflow.

The experiment uses:

- a frozen base model;
- quantized loading;
- PEFT/QLoRA;
- adapter artifacts;
- frozen authorities;
- artifact fingerprints;
- training provenance;
- protected evaluation.

The experiment demonstrates adaptation engineering rather than simply calling a pre-trained model.

---

# 14. Independent protected evaluation

One of the most important parts of the capstone is the independent Hospital evaluation.

The evaluation was designed as a protected single-use benchmark.

Key rules included:

- preregistered protocol;
- protected benchmark;
- label-blind model input;
- protected gold hidden before generation;
- immutable scoring logic;
- frozen thresholds;
- exactly one official consumption;
- no post-result replay.

The official adapted-model result was:

```text
17 / 30 correct
accuracy: 0.566667
macro score: 0.566667
strict JSON: 30 / 30
```

The base model result was:

```text
0 / 30 correct
accuracy: 0.0
macro score: 0.0
strict JSON: 0 / 30
```

The adapter therefore produced a material improvement.

However, the candidate did not satisfy all preregistered absolute promotion gates.

Official lifecycle state:

```text
evaluation_status: completed
promotion_decision: not_promoted
```

This is a deliberate engineering outcome.

The experiment improved the model, but DataLens did not reinterpret that improvement as production readiness.

---

# 15. Why the QLoRA model was not promoted

The adapted model passed some relation categories and failed others.

Strong areas included:

- same-process / different-stage classification;
- unrelated classification;
- strict JSON output;
- non-regression versus the base model.

Weak areas remained, especially:

- uncertain relations;
- same-metric / different-state relations;
- absolute aggregate thresholds.

The promotion policy therefore remained fail-closed.

This demonstrates a core AI Engineering principle:

> Evaluation thresholds should not be rewritten after observing results.

---

# 16. Single-use benchmark discipline

The Hospital benchmark is now:

```text
CONSUMED / NEVER REPLAY
```

It must not be reused to:

- tune the same candidate;
- adjust thresholds;
- test another iteration of the same tuning cycle;
- optimize prompts after seeing results.

A future candidate requiring independent evidence must use a new independent benchmark.

This protects the meaning of the evaluation.

---

# 17. Lifecycle governance

After the official evaluation, DataLens records lifecycle evidence separately from the model artifact.

The lifecycle layer captures:

- artifact identity;
- provenance identity;
- evaluation identity;
- source receipt;
- evaluation status;
- promotion eligibility;
- promotion decision.

The current QLoRA v0.4 decision is:

```text
completed / not_promoted
```

This wording is important.

`completed` means the official evaluation lifecycle completed successfully.

`not_promoted` means the model did not meet promotion requirements.

The system does not call the evaluation itself a failure.

---

# 18. Observability

DataLens contains structured runtime and AI traces.

The observability stack includes:

- request correlation;
- runtime traces;
- AI traces;
- trace persistence;
- trace explorer.

A request can be correlated through:

```text
X-DataLens-Request-ID
```

The observability layer focuses on system behavior rather than raw payload storage.

---

# 19. Privacy-constrained traces

Runtime traces deliberately exclude high-risk data.

They do not persist:

- request bodies;
- response bodies;
- query strings;
- request headers;
- client IP addresses;
- raw dataset rows;
- uploaded file contents;
- document chunks;
- raw exception messages;
- filesystem paths.

The goal is:

```text
debuggable system
without
secondary raw-data logging
```

---

# 20. Security architecture

Security is distributed across the application.

Current protections include:

- local-only LLM egress;
- redirect rejection;
- proxy bypass for local LLM traffic;
- payload classification;
- raw-row prohibition;
- bounded semantic samples;
- local frontend CORS;
- sanitized public errors;
- cross-workflow Model Lab isolation;
- trusted artifact loading;
- lifecycle identity validation;
- protected evaluation boundaries;
- non-root containers.

DataLens is therefore best described as:

```text
security-aware and privacy-constrained
```

not:

```text
fully hardened multi-tenant production platform
```

---

# 21. Known security limits

The project does not claim complete coverage for:

- prompt-injection detection;
- jailbreak-specific defenses;
- general automatic PII detection;
- authentication and authorization for multiple users;
- comprehensive rate limiting.

These are documented hardening opportunities rather than hidden omissions.

The local-first deployment model reduces exposure but does not eliminate the need for security boundaries.

---

# 22. Docker and reproducibility

The standard application runtime is containerized.

The repository includes:

```text
apps/api/Dockerfile
apps/web/Dockerfile
compose.yaml
compose.registry.yaml
```

The API container uses:

```text
python:3.13-slim
```

The web container uses:

```text
node:22-alpine
```

Both runtime images use a non-root `datalens` user.

The standard stack exposes:

```text
Web: 127.0.0.1:3000
API: 127.0.0.1:8000
```

A versioned `.env.example` documents major runtime configuration variables.

---

# 23. CI/CD

DataLens uses four main GitHub Actions workflows:

```text
datalens-evals.yml
datalens-runtime.yml
datalens-publish.yml
datalens-release.yml
```

They cover different responsibilities.

## Evaluation gate

Protects AI/evaluation behavior from regression.

## Runtime gate

Validates runtime and container behavior.

## Publish workflow

Builds and publishes application images.

## Release workflow

Controls release authority and versioned delivery.

The project keeps application release authority separate from model promotion authority.

---

# 24. Testing strategy

The capstone has a large automated test footprint.

At the final gap-discovery audit, the repository contained:

```text
Backend test files:  422
Frontend test files: 175
```

This includes dedicated tests around:

- analytics;
- preparation;
- planning;
- RAG;
- evaluation;
- classical ML;
- deep learning;
- Model Lab;
- monitoring;
- observability;
- security;
- adaptation;
- model lifecycle;
- Docker / CI behavior.

The goal is not test-count maximization.

The value is that important architectural boundaries are represented as executable evidence.

---

# 25. Example end-to-end DataLens path

A representative product flow is:

```text
1. User imports a dataset
2. DataLens creates preparation state
3. Dataset quality is reviewed
4. Cleaning / transformation / combination is performed if needed
5. Dataset is validated
6. User asks an analytical question
7. Intent is interpreted
8. Analysis plan is validated
9. Deterministic Python execution computes evidence
10. Structured results are created
11. Charts / explanations are generated
12. Report findings are assembled
13. Runtime and AI traces provide observability
```

A model-oriented workflow can continue:

```text
14. User opens Model Lab
15. A model is trained
16. Holdout evaluation is performed
17. Artifact and provenance are persisted
18. Trusted model is loaded
19. Predictions are generated
20. Monitoring evaluates drift / performance / health
```

---

# 26. Engineering challenges

Several parts of DataLens required more than adding isolated features.

## Challenge 1 — AI vs deterministic authority

The architecture had to prevent LLM interpretation from becoming numerical truth.

Solution:

- explicit execution layer;
- structured plans;
- deterministic computation;
- server-owned evidence.

---

## Challenge 2 — persistent preparation state

Preparation required revision-aware state and controlled transitions rather than stateless endpoints.

Solution:

- server-owned workflow sessions;
- workflow revision;
- validated output handoff;
- persistence.

---

## Challenge 3 — private but useful observability

Tracing needed to be useful without leaking datasets or raw document content.

Solution:

- metadata-only runtime traces;
- explicit privacy contract;
- safe request correlation;
- sanitized failures.

---

## Challenge 4 — model persistence and trust

Training a model was not enough.

Solution:

- artifact registration;
- provenance;
- trusted loaders;
- workflow isolation;
- evaluation summaries;
- monitoring.

---

## Challenge 5 — honest model adaptation evaluation

A tuned model could not be declared successful based only on training loss or reused test data.

Solution:

- independent holdout;
- preregistered thresholds;
- frozen scoring;
- single-use benchmark;
- lifecycle decision.

---

# 27. What DataLens demonstrates as AI Engineering

The project demonstrates practical experience across several AI Engineering domains.

## Software engineering

- FastAPI backend architecture;
- Next.js frontend;
- typed contracts;
- persistence;
- error handling;
- tests;
- containerization.

## Data engineering / analytics

- dataset preparation;
- validation;
- transformations;
- deterministic analytics;
- time-series handling;
- reporting.

## Machine learning engineering

- training contracts;
- classical estimators;
- evaluation;
- model comparison;
- tuning;
- artifact persistence;
- inference;
- monitoring.

## Deep learning

- autoencoders;
- tabular neural models;
- MLP;
- RNN;
- LSTM;
- training;
- bundles;
- trusted loading.

## LLM engineering

- local inference;
- semantic workflows;
- structured outputs;
- orchestration;
- RAG;
- relevance;
- model-visible payload boundaries.

## Adaptation

- PEFT/QLoRA;
- quantized loading;
- adapter artifacts;
- experimental provenance;
- independent evaluation.

## Evaluation engineering

- regression gates;
- protected benchmarks;
- preregistered thresholds;
- non-regression checks;
- immutable evidence.

## LLMOps / MLOps

- artifact stores;
- lifecycle registries;
- promotion decisions;
- monitoring;
- CI gates;
- Docker;
- publishing;
- release workflows.

## Security / privacy

- local-only egress;
- payload minimization;
- raw-row prohibition;
- trace privacy;
- trusted artifacts;
- public error sanitization.

---

# 28. What DataLens does not try to prove

The capstone does not claim that:

- every model should be deployed;
- every AI decision is correct;
- local models eliminate security risk;
- QLoRA automatically improves every semantic task;
- the product is a finished enterprise SaaS system;
- the current architecture provides multi-tenant IAM;
- every security attack class is fully mitigated.

Instead, the project demonstrates engineering discipline around uncertainty.

---

# 29. Most important result

The most important result of DataLens is not a single model score.

It is the architectural progression from:

```text
"ask an LLM to analyze data"
```

to:

```text
validated data
    +
deterministic execution
    +
structured AI
    +
controlled RAG
    +
trained models
    +
evaluation evidence
    +
observability
    +
security boundaries
    +
lifecycle governance
```

That progression is the central AI Engineering story of the project.

---

# 30. Current status

The main technical foundations are implemented.

Current status:

```text
Data preparation                 IMPLEMENTED
Deterministic analytics          IMPLEMENTED
Semantic AI                      IMPLEMENTED
RAG                              IMPLEMENTED
Reporting                        IMPLEMENTED
Classical ML                     IMPLEMENTED
Deep learning                    IMPLEMENTED
Model Lab                        IMPLEMENTED
Monitoring                       IMPLEMENTED
Observability                    IMPLEMENTED
Security boundaries              IMPLEMENTED
QLoRA adaptation                 IMPLEMENTED
Independent evaluation           IMPLEMENTED
Model lifecycle governance       IMPLEMENTED
Docker                           IMPLEMENTED
CI/CD                            IMPLEMENTED

Project README                   IMPLEMENTED
Architecture documentation      IMPLEMENTED
Environment reference           IMPLEMENTED
Security threat model           IMPLEMENTED
AI Engineering case study       THIS DOCUMENT
Backend README                   IMPLEMENTED
Demo walkthrough                IMPLEMENTED
Final documentation audit       IN PROGRESS
```

---

# 31. Recruiter summary

DataLens demonstrates an end-to-end AI engineering system rather than a collection of notebooks.

The project shows the ability to:

- design backend AI architecture;
- keep deterministic analytics authoritative;
- integrate a local LLM safely;
- build RAG with explicit relevance boundaries;
- train and persist ML/DL models;
- build evaluation pipelines;
- adapt an LLM with QLoRA;
- enforce independent benchmark discipline;
- implement model lifecycle governance;
- instrument AI systems with privacy-aware observability;
- containerize and validate runtime behavior;
- build CI/CD and release gates;
- expose capabilities through a real frontend.

The project also records negative or incomplete outcomes honestly.

The QLoRA candidate improved substantially over the base model but was not promoted because the preregistered thresholds were not fully satisfied.

That result is part of the evidence that DataLens is built around engineering controls rather than optimistic demo claims.

---

# 32. Final principle

DataLens can be summarized by one engineering principle:

> AI should increase capability without removing authority, evidence, reproducibility, or accountability from the system around it.
