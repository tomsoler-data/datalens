# DataLens Security

This document describes the current security and privacy model of DataLens.

DataLens is a local-first analytics and AI engineering platform. Its security model is designed around explicit trust boundaries, fail-closed validation, minimal model-visible data, server-owned authority, privacy-constrained observability, and controlled artifact access.

This document describes the protections currently implemented in the repository and identifies remaining hardening work.

---

## 1. Security goals

The main security goals are:

- keep LLM traffic local by default;
- prevent accidental model egress to remote hosts;
- minimize dataset-derived content sent to a model;
- forbid raw tabular rows from model-visible payloads;
- keep trusted workflow state server-owned;
- isolate model artifacts across workflows;
- avoid leaking internal paths and exception details through public APIs;
- prevent observability from becoming a raw-data store;
- preserve protected evaluation boundaries;
- maintain explicit model provenance and lifecycle authority;
- run application containers without root privileges;
- fail closed when authority, provenance, or protected evidence is ambiguous.

---

## 2. Threat model scope

This threat model focuses on the current local-first application architecture.

Primary assets include:

- uploaded datasets;
- uploaded documents;
- preparation workflow state;
- analysis artifacts;
- model artifacts;
- deep-learning bundles;
- model provenance;
- lifecycle registries;
- evaluation evidence;
- protected benchmark material;
- local traces;
- runtime configuration;
- local Ollama access.

Primary trust boundaries include:

```text
Browser
  |
  v
FastAPI boundary
  |
  +--> deterministic analytics
  +--> preparation workflow
  +--> document / RAG pipeline
  +--> Model Lab
  +--> reporting
  +--> observability
  |
  v
Local model boundary
  |
  v
Ollama
```

Additional boundaries exist around:

```text
Model artifact storage
Lifecycle registries
Protected evaluation material
Docker runtime
CI/CD release authority
```

---

## 3. Trust assumptions

DataLens currently assumes:

- the machine running DataLens is trusted by the user;
- the local filesystem is not already compromised;
- the local Docker daemon is trusted;
- the local Ollama runtime is trusted to execute configured models;
- GitHub repository permissions and CI credentials are managed outside the application;
- users with direct filesystem access can access local runtime state;
- DataLens is not currently designed as a multi-tenant hostile-cloud service.

These assumptions matter because local-first reduces remote exposure but does not protect against a compromised host.

---

## 4. Local-first LLM egress

DataLens implements an explicit local-only LLM transport policy.

The egress guard allows:

- `localhost`;
- literal loopback IP addresses;
- `host.docker.internal` only when the Docker bridge is explicitly enabled;
- Docker Ollama bridge access only on port `11434`.

The guard rejects:

- arbitrary DNS hostnames;
- LAN addresses;
- non-loopback IP addresses;
- HTTPS remote destinations;
- URL credentials;
- query strings;
- fragments;
- malformed ports;
- unexpected Docker bridge ports.

The Docker bridge requires explicit opt-in through:

```text
DATALENS_LLM_DOCKER_BRIDGE_ENABLED=1
```

The primary Ollama endpoint is configured through:

```text
DATALENS_OLLAMA_HOST
```

Default local development value:

```text
http://localhost:11434
```

---

## 5. Redirect and proxy protection

LLM egress does not inherit normal HTTP proxy behavior.

The application builds a dedicated opener with proxy handling disabled for local model I/O.

HTTP redirects are rejected before a second network request is performed.

This prevents a nominally local URL from silently redirecting the request to another destination.

The rule is intentionally strict:

```text
local request
    |
    v
destination validation
    |
    v
no proxy inheritance
    |
    v
no redirects
    |
    v
local model I/O
```

---

## 6. Model-visible payload classification

Model-visible content is explicitly classified before transport.

Current payload classes include:

```text
metadata_only
deterministic_evidence
semantic_value_sample
document_content
tabular_raw_rows
```

Allowed classes are:

```text
metadata_only
deterministic_evidence
semantic_value_sample
document_content
```

The following class is explicitly forbidden:

```text
tabular_raw_rows
```

Unknown or invalid payload classifications fail closed.

---

## 7. Raw tabular row prohibition

Complete or row-shaped tabular records are not authorized as model-visible payloads.

This means local-first does not imply unrestricted model access to all dataset contents.

The privacy boundary is stronger:

```text
dataset
  |
  +--> metadata -------------------------- allowed
  +--> deterministic summaries ---------- allowed
  +--> bounded semantic values ---------- allowed
  +--> complete raw rows ---------------- forbidden
```

This reduces unnecessary disclosure even to a local model.

---

## 8. Semantic value sampling

Some semantic tasks cannot be resolved from column names or schema metadata alone.

For those cases, DataLens supports a bounded semantic value sample.

The current privacy budget is:

```text
maximum values per semantic sample: 5
```

This mechanism is intentionally distinct from raw-row access.

A semantic sample must be:

- an explicit sequence;
- field-scoped;
- bounded;
- used only when semantic interpretation requires actual values.

Oversized samples fail before model transport.

---

## 9. Document content

Document content is an allowed model-visible payload class because document understanding and RAG explicitly require access to uploaded text.

This is different from tabular raw-row access.

Document ingestion currently supports controlled extraction from:

- PDF;
- DOCX;
- TXT;
- Markdown.

Document handling still remains subject to application-level ingestion limits and downstream RAG relevance rules.

---

## 10. RAG security boundary

Retrieved content is not automatically trusted.

The architecture separates:

```text
ingestion
retrieval
relevance
context acceptance
grounded explanation
```

This matters because document retrieval alone does not prove that content is relevant or safe to use as authoritative context.

Current security value:

- retrieved material does not automatically become accepted evidence;
- context construction is a separate step;
- explanation is downstream from relevance decisions.

---

## 11. Prompt injection status

DataLens currently has strong local egress and payload privacy controls, but it does **not** currently expose a dedicated, explicit prompt-injection or jailbreak defense subsystem under those names.

This is an important distinction.

Existing controls reduce prompt-injection impact indirectly by:

- limiting model-visible data;
- separating deterministic execution from generated text;
- keeping authoritative workflow state server-owned;
- separating retrieval from relevance;
- validating structured outputs;
- preventing the model from directly owning lifecycle decisions.

However, these controls should not be described as a complete prompt-injection defense.

### Remaining hardening opportunity

Future hardening may include:

- explicit prompt-injection detection;
- instruction-source separation;
- untrusted-document labeling;
- stronger RAG content isolation;
- tool-call authorization policies;
- instruction precedence validation;
- adversarial prompt-injection evals;
- jailbreak-focused regression tests.

This is a documented future hardening area, not a hidden claim of existing coverage.

---

## 12. Browser trust boundary

The browser is not treated as an authoritative source for trusted backend state.

Examples of server-owned authority include:

- preparation workflow state;
- workflow revision;
- validated preparation output;
- model artifact authority;
- evaluation evidence;
- model lifecycle decisions;
- trusted model paths.

The client can request operations, but it does not manufacture trusted evidence.

---

## 13. Preparation workflow security

Preparation state is server-owned.

The frontend receives a read-only representation rather than being allowed to submit the complete authoritative workflow state.

This helps prevent a client from fabricating:

- readiness;
- successful validation;
- output dataset authority;
- completed preparation stages.

Revision-aware state also helps reject stale decisions evaluated against an older workflow revision.

---

## 14. Model Lab isolation

Model Lab exposes trusted model operations without exposing arbitrary model filesystem access.

Public contracts intentionally avoid exposing sensitive implementation details such as:

- raw model bytes;
- estimator objects;
- internal filesystem paths;
- complete internal training contracts.

A model identifier belonging to another workflow is not revealed as a cross-workflow existence leak.

The public API intentionally maps workflow mismatch to the same external shape as a missing model.

This reduces cross-workflow information disclosure.

---

## 15. Trusted artifact loading

Persisted model artifacts are not intended to be loaded through arbitrary client-supplied paths.

Trusted loading and artifact registration separate:

```text
artifact identity
artifact persistence
trusted resolution
model loading
prediction / evaluation
```

This boundary reduces direct path injection and accidental cross-workflow access.

---

## 16. Lifecycle registry security

Model lifecycle registries store governance metadata rather than owning model bytes.

The identity/provenance registry binds:

- artifact identity;
- artifact family;
- experiment identity;
- source contract;
- source contract hash;
- provenance.

Inconsistent artifact/provenance bindings fail validation.

The lifecycle layer therefore provides an integrity boundary above raw artifact filenames.

---

## 17. Evaluation decision security

Evaluation decision state is stored separately from model identity.

This supports explicit separation between:

```text
model exists
model was evaluated
evaluation completed
model was promoted
```

A completed evaluation does not imply promotion.

For the current QLoRA v0.4 candidate:

```text
status: completed
promotion_decision: not_promoted
```

---

## 18. Protected benchmark security

The independent Hospital benchmark is protected evaluation evidence.

The core security rules are:

- consumed exactly once;
- never replayed for the same candidate;
- protected gold is not model-visible before generation;
- scoring happens after generation;
- official artifacts are immutable evidence;
- thresholds are not changed after results are known;
- the benchmark is not converted into tuning data.

Current status:

```text
CONSUMED / NEVER REPLAY
```

Any future tuned candidate requiring independent validation must use new independent benchmark evidence.

---

## 19. Observability privacy

Observability is explicitly privacy-constrained.

Runtime traces are designed to record controlled metadata such as:

- request identifier;
- timestamp;
- method;
- server-owned route template;
- status;
- duration;
- validated workflow identifier when available;
- high-level failure kind.

Runtime traces deliberately exclude:

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

The goal is:

```text
useful correlation
without
raw payload persistence
```

---

## 20. Public error sanitization

Internal exceptions and public API errors are separate concerns.

Public APIs should not expose:

- private filesystem paths;
- hidden model locations;
- internal exception details;
- secret workflow information;
- cross-workflow model existence.

This pattern is tested in multiple API and ML workflows.

---

## 21. CORS policy

The FastAPI application uses an explicit local frontend allowlist.

Current allowed origins include:

```text
http://localhost:3000
http://127.0.0.1:3000
```

Credentials are not enabled.

Allowed methods and headers are explicitly constrained.

This is appropriate for the current local-first deployment model.

---

## 22. Container security

The standard API and web images run as non-root users.

Current runtime user:

```text
datalens
UID 10001
GID 10001
```

The API image:

- uses `python:3.13-slim`;
- runs as non-root;
- exposes a healthcheck;
- stores runtime state under `/app/var`.

The web image:

- uses a multi-stage Node build;
- runs as non-root;
- exposes a healthcheck;
- disables Next.js telemetry.

This reduces unnecessary container privilege.

---

## 23. Network exposure

The standard Docker Compose configuration binds the application to loopback interfaces:

```text
127.0.0.1:3000
127.0.0.1:8000
```

This prevents the standard stack from automatically listening on all host network interfaces.

Changing these bindings changes the security posture and should be treated as an explicit deployment decision.

---

## 24. Secrets

The repository contains tests and contracts that refer to secret-handling behavior, but `.env.example` intentionally contains no credentials or tokens.

Do not commit:

- API tokens;
- bearer tokens;
- private SSH keys;
- cloud credentials;
- private model repository credentials;
- private datasets;
- production secrets;
- protected benchmark gold material.

A deployment that introduces external services should use an appropriate secret-management mechanism rather than storing credentials in Git.

---

## 25. Personally identifiable information

The current codebase strongly emphasizes privacy boundaries, raw-row prohibition, sanitization, and controlled traces.

However, there is not currently a dedicated central PII-classification subsystem explicitly identified as such.

This means DataLens should not claim automatic PII discovery or redaction for arbitrary datasets.

Current protection should be described as:

- data minimization;
- payload classification;
- trace minimization;
- local-first model transport;
- raw-row prohibition.

Future hardening may add explicit PII detection or classification.

---

## 26. File and document upload risk

Uploaded documents and datasets should be treated as untrusted input.

Existing architectural mitigations include:

- supported-file restrictions;
- file-count and byte limits;
- structured ingestion;
- deterministic parsing paths;
- downstream relevance controls;
- controlled public error surfaces.

Remaining risk includes malformed or adversarial content that is syntactically valid but semantically hostile to downstream AI interpretation.

---

## 27. Denial-of-service considerations

Local-first does not eliminate resource-exhaustion risk.

Potential resource pressure includes:

- large datasets;
- large documents;
- expensive model training;
- deep-learning workloads;
- long model generation;
- large monitoring histories;
- repeated report generation.

Existing mitigations include selected size limits, bounded document ingestion, worker timeouts, and controlled execution paths.

A hardened shared deployment would require stronger quotas and rate limiting.

---

## 28. Deep-learning worker boundary

Neural time-series execution can use an isolated Python runtime.

The executable may be overridden through:

```text
DATALENS_DL_PYTHON_PATH
```

The default resolves to a project-local `.venv-dl`.

This is a local execution boundary and should not be configured to an untrusted executable.

---

## 29. Runtime configuration risk

Environment variables can alter runtime paths and model endpoints.

Configuration is therefore part of the security model.

High-impact variables include:

```text
DATALENS_OLLAMA_HOST
DATALENS_LLM_DOCKER_BRIDGE_ENABLED
DATALENS_DL_PYTHON_PATH
DATALENS_ML_MODEL_ARTIFACT_STORE_PATH
DATALENS_MODEL_LIFECYCLE_REGISTRY_PATH
DATALENS_MODEL_LIFECYCLE_EVALUATION_DECISION_REGISTRY_PATH
DATALENS_AI_TRACE_PATH
DATALENS_RUNTIME_TRACE_PATH
```

Deployment configuration should be controlled by the operator rather than user-supplied through the application UI.

---

## 30. CI/CD security boundary

CI/CD validates and publishes application artifacts, but CI success is not equivalent to model promotion.

The project deliberately separates:

```text
application release authority
```

from:

```text
model lifecycle promotion authority
```

This prevents a successful application build from silently redefining model-governance state.

---

## 31. Security test coverage

The repository contains dedicated security-related tests covering areas such as:

- LLM egress;
- LLM redirect rejection;
- payload privacy;
- transport invariants;
- semantic value sample boundaries;
- AI trace privacy;
- HTTP error privacy;
- document upload guards;
- offline evaluation boundaries;
- local frontend CORS;
- preparation AI error privacy;
- report error privacy.

Security is therefore tested across system boundaries rather than only inside `app/security`.

---

## 32. Known gaps

The following areas remain explicit hardening opportunities.

### 32.1 Prompt injection

No dedicated prompt-injection defense subsystem is currently claimed.

### 32.2 Jailbreak testing

No dedicated jailbreak-specific evaluation suite is currently claimed.

### 32.3 Central PII classification

No general automatic PII detector or redaction engine is currently claimed.

### 32.4 Authentication and authorization

The current local-first architecture does not represent a full multi-user identity and access-management system.

A hosted multi-user deployment would require authentication, authorization, tenancy, and stronger access isolation.

### 32.5 Rate limiting

The current local product is not documented as providing a comprehensive rate-limiting layer.

### 32.6 Host compromise

Local-first cannot protect data from an already compromised host operating system or privileged local user.

---

## 33. Recommended future hardening

Future security work should prioritize:

1. explicit prompt-injection and jailbreak evaluation;
2. instruction-source labeling for RAG;
3. tool authorization policies;
4. PII classification where required;
5. dependency and image vulnerability scanning;
6. SBOM generation;
7. container capability hardening;
8. stronger filesystem mount restrictions;
9. authentication/authorization before any shared deployment;
10. rate limiting and workload quotas for networked deployments.

These are hardening improvements, not prerequisites for the current local capstone architecture.

---

## 34. Security invariants

The following invariants summarize the intended current posture.

### LLM transport

```text
REMOTE MODEL EGRESS: FORBIDDEN BY DEFAULT
```

### Raw tabular model visibility

```text
RAW TABULAR ROWS: FORBIDDEN
```

### Semantic sample budget

```text
MAX SEMANTIC SAMPLE VALUES: 5
```

### Runtime traces

```text
RAW REQUEST / RESPONSE PAYLOAD STORAGE: FORBIDDEN
```

### Protected Hospital benchmark

```text
CONSUMED / NEVER REPLAY
```

### QLoRA v0.4 lifecycle

```text
completed / not_promoted
```

### Client authority

```text
BROWSER STATE != TRUSTED SERVER AUTHORITY
```

---

## 35. Reporting security issues

For this capstone repository, security findings should be documented privately before publishing exploit details if they could expose local data, protected evaluation material, credentials, or artifact authority.

Do not include real secrets, private datasets, or protected benchmark gold data in public issues.

---

## 36. Security status

Current security posture:

```text
Local-only LLM egress             IMPLEMENTED
Redirect rejection                IMPLEMENTED
Proxy bypass for local LLM I/O    IMPLEMENTED
Payload classification            IMPLEMENTED
Raw tabular row prohibition       IMPLEMENTED
Bounded semantic sampling         IMPLEMENTED
Privacy-constrained traces        IMPLEMENTED
Public error sanitization         IMPLEMENTED
Model artifact isolation          IMPLEMENTED
Lifecycle integrity checks        IMPLEMENTED
Protected evaluation boundary     IMPLEMENTED
Non-root containers               IMPLEMENTED
Loopback default exposure         IMPLEMENTED

Prompt-injection hardening         PARTIAL / FUTURE HARDENING
Jailbreak-specific evaluation      FUTURE HARDENING
General PII classification        FUTURE HARDENING
Multi-user authN/authZ             OUT OF CURRENT LOCAL SCOPE
Rate limiting                      FUTURE HARDENING
```

DataLens should therefore be described as **security-aware and privacy-constrained**, not as a fully hardened multi-tenant production platform.
