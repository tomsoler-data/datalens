# DataLens — Demo Walkthrough

This walkthrough is designed to demonstrate DataLens as an end-to-end AI Engineering capstone.

The goal is not to show every screen or every subsystem.

The goal is to demonstrate the engineering story clearly:

> Data enters the system, is prepared and validated, analyzed through deterministic execution, enriched by controlled AI, used for model training and evaluation, observed through traces and monitoring, and governed through explicit lifecycle evidence.

---

# 1. Demo objective

A successful demo should prove six things.

1. DataLens is a real application, not a notebook.
2. AI assists the system without replacing deterministic computation.
3. Data preparation and analysis are explicit workflows.
4. ML and deep learning are integrated into a product surface.
5. Evaluation, monitoring, observability, and security are first-class concerns.
6. Experimental model adaptation is governed honestly through lifecycle evidence.

---

# 2. Recommended demo length

A concise recruiter-facing demonstration should take approximately:

```text
8 to 12 minutes
```

A longer technical walkthrough can take:

```text
20 to 30 minutes
```

The short version should prioritize the system architecture and key transitions rather than opening every implementation detail.

---

# 3. Demo structure

Recommended order:

```text
1. Product overview
2. Dataset import
3. Preparation workflow
4. Deterministic analysis
5. Report generation
6. Model Lab
7. Monitoring
8. Observability
9. Deep learning evidence
10. QLoRA adaptation / lifecycle
11. Architecture / security summary
12. Closing
```

This order tells a coherent story from raw data to governed AI.

---

# 4. Before the demo

Before recording or presenting, verify:

```text
Web UI reachable
API reachable
Ollama reachable if AI features are required
Repository clean
Expected branch / commit checked
Required dataset available
No private data visible
No secret paths or credentials visible
No protected benchmark replay
```

Do not use the protected Hospital benchmark during the demo.

Its status remains:

```text
CONSUMED / NEVER REPLAY
```

The demo should show the frozen result or lifecycle receipt only if needed.

---

# 5. Start the application

Recommended standard runtime:

```bash
docker compose up --build
```

Expected local endpoints:

```text
Web: http://127.0.0.1:3000
API: http://127.0.0.1:8000
API docs: http://127.0.0.1:8000/docs
```

If using direct local development, start the API and frontend separately.

---

# 6. Opening narrative

Suggested introduction:

> DataLens is a local-first data analysis and AI engineering platform.
> The project started as a data-analysis workflow and evolved into a capstone designed to demonstrate how deterministic analytics, local LLMs, RAG, machine learning, deep learning, evaluation, observability, security, and model lifecycle governance can work together inside one application.

Then explain the main design principle:

> The LLM can interpret, explain, and assist, but trusted data state, numerical results, evaluation evidence, and model lifecycle decisions remain controlled by deterministic or server-owned logic.

---

# 7. Show the main product surfaces

Briefly show the major areas of the interface.

Recommended product surfaces:

```text
Preparation
Analysis
Reporting
Model Lab
Monitoring / Observability
```

Do not stay too long on navigation.

The purpose is to establish that the project is one integrated product.

---

# 8. Dataset import

Import a safe demonstration dataset.

Use a dataset that contains enough structure to support:

- numerical variables;
- categorical variables;
- optionally time;
- optionally group/entity identifiers.

The dataset should be small enough for a predictable live demonstration.

Avoid private or personal data.

---

# 9. Preparation workflow

Show that import is not immediately treated as analysis-ready.

Walk through the preparation stages.

Representative progression:

```text
IMPORT
UNDERSTAND
QUALITY
CLEAN
TRANSFORM
COMBINE
VALIDATE
```

Explain:

> Preparation state is server-owned. The frontend does not decide that a dataset is validated just because a button was clicked.

Show one or two meaningful preparation actions.

Examples:

- missing-value review;
- duplicate review;
- semantic review;
- cleaning decision;
- transformation;
- dataset combination;
- final validation.

---

# 10. Explain the server-owned preparation model

This is an important AI Engineering point.

Suggested explanation:

> DataLens keeps the preparation workflow state on the server, including workflow revision, selected root datasets, derived outputs, and validation state. This prevents the browser from manufacturing trusted readiness.

If useful, mention revision-aware decisions.

Avoid diving into implementation unless the audience is technical.

---

# 11. Validate the analytical handoff

Finish the preparation flow by reaching a validated dataset.

The important transition is:

```text
prepared data
    |
    v
validated analytical input
```

Explain:

> Validation is the handoff boundary between preparation and trusted downstream analysis.

---

# 12. Submit an analytical request

Use a request that requires deterministic computation.

Examples:

```text
Which segment has the highest revenue?
```

```text
How did revenue evolve over time?
```

```text
Which categories perform above the overall average?
```

```text
Which groups are the top performers?
```

Choose a request supported by the demo dataset.

---

# 13. Show AI interpretation

If the request passes through semantic planning, show that DataLens interprets the analytical intent.

Explain:

> The AI layer can help understand what the user wants, but it does not calculate the final answer itself.

This is the key transition from semantic AI to deterministic execution.

---

# 14. Show deterministic execution

Display the resulting analytical output.

Explain:

> Once the request is understood, the numerical answer comes from deterministic Python execution.

If possible, show:

- computed metric;
- grouped result;
- chart;
- structured finding.

Emphasize:

```text
generated explanation != numerical source of truth
```

---

# 15. Show visualization

Show the chart generated from the same structured analytical result.

This demonstrates consistency between:

```text
computed groups
reported groups
chart groups
```

Avoid presenting the visualization as a separate AI generation step.

---

# 16. Show grounded explanation

If the application produces an explanation, explain the evidence boundary.

Suggested wording:

> The explanation is downstream from computed evidence. DataLens can use AI to make the result easier to understand, but the underlying metrics already exist independently of the prose.

This is an important distinction from chatbot-only analytics.

---

# 17. Demonstrate RAG

If using the long demo, upload one safe document.

Supported examples:

```text
PDF
DOCX
TXT
Markdown
```

Show that DataLens:

```text
ingests
chunks
retrieves
checks relevance
constructs context
explains
```

Explain:

> Retrieval alone does not mean the chunk is accepted as authoritative context.

This demonstrates the retrieval/relevance separation.

---

# 18. Reporting

Generate or preview a report from the analytical workflow.

Show that the report combines:

- deterministic findings;
- visual evidence;
- structured selection;
- explanatory text.

Explain:

> Reporting is downstream from analysis. Generated prose does not redefine the evidence.

If PDF output is available, show it briefly.

---

# 19. Transition to Model Lab

Suggested transition:

> DataLens does not stop at descriptive analytics. The same validated data workflow can continue into model development.

Open Model Lab.

---

# 20. Classical ML demonstration

Use one simple model workflow.

Possible tasks:

```text
regression
classification
```

Show:

- selected target;
- training;
- evaluation;
- model result;
- comparison if available.

Do not attempt to demonstrate every estimator.

The point is to show the full model workflow.

---

# 21. Explain holdout discipline

If the UI exposes split information, mention:

- holdout;
- group holdout;
- time holdout;
- purged group/time holdout.

Suggested explanation:

> Evaluation strategy depends on the data structure. DataLens supports group-aware and time-aware splits to reduce leakage risk.

This is stronger evidence than simply showing a train/test score.

---

# 22. Show model artifact persistence

After training, show that the model becomes a persisted artifact.

Explain:

> Training produces a registered artifact that can later be loaded, evaluated, monitored, and used for prediction.

The key lifecycle is:

```text
train
evaluate
persist
load
predict
monitor
```

---

# 23. Show model detail

Open the trusted model detail view.

If available, show:

- problem type;
- metrics;
- selection evidence;
- preprocessing;
- evaluation summary;
- model identity.

Avoid exposing internal filesystem paths.

---

# 24. Show prediction

Run one safe prediction example.

Explain:

> Prediction uses a trusted persisted model, not an arbitrary client-supplied model path.

This demonstrates the trusted artifact boundary.

---

# 25. Show monitoring

Open monitoring or model health.

Demonstrate one or more of:

```text
drift
performance
model health
monitoring history
alerts
```

Explain:

> DataLens treats model deployment as an ongoing lifecycle rather than a one-time training event.

---

# 26. Show observability

Open the observability or trace explorer surface.

Show one request or AI trace.

Explain:

> Traces are designed to explain system behavior without storing raw request payloads, dataset rows, uploaded document content, or filesystem paths.

If visible, show the request correlation ID.

---

# 27. Mention privacy-aware tracing

Summarize the runtime trace policy.

Runtime traces intentionally exclude:

```text
request bodies
response bodies
query strings
headers
client IPs
raw dataset rows
uploaded file contents
document chunks
raw exception messages
filesystem paths
```

This demonstrates that observability has its own privacy contract.

---

# 28. Deep learning evidence

For a short demo, do not retrain a deep model live unless runtime is predictable.

Instead, show existing Model Lab evidence or backend capability.

Explain that DataLens contains dedicated deep-learning workflows for:

```text
tabular neural models
autoencoders
MLP time-series models
RNN
LSTM
```

If a stable UI path exists, show one trained deep-learning artifact or result.

If not, show the documentation / architecture section briefly rather than forcing an unstable live flow.

---

# 29. Autoencoder example

If demonstrating anomaly detection:

Explain the workflow:

```text
validated data
  |
autoencoder training
  |
reconstruction error
  |
threshold
  |
anomaly decision
  |
persisted artifact
```

The point is to demonstrate a real neural workflow beyond LLM usage.

---

# 30. Time-series deep learning example

If available, show one neural time-series model.

Possible architectures:

```text
MLP
RNN
LSTM
```

Explain:

> These models use a dedicated deep-learning execution path and are registered into the broader artifact system.

This demonstrates integration rather than isolated experimentation.

---

# 31. Transition to model adaptation

Suggested transition:

> The last part of the capstone explores model adaptation itself. Rather than only consuming a local LLM, DataLens includes a controlled QLoRA experiment.

Do not run training live during the standard demo.

Use frozen evidence.

---

# 32. Show QLoRA adaptation evidence

Show the adaptation documentation or frozen artifacts.

Explain:

- frozen base model;
- quantized loading;
- adapter training;
- provenance;
- protected independent evaluation.

Keep the explanation focused on the engineering process.

---

# 33. Show official evaluation result

Present the official adapted result:

```text
Adapted:
17 / 30 correct
accuracy: 0.566667
macro: 0.566667
strict JSON: 30 / 30

Base:
0 / 30 correct
accuracy: 0.0
macro: 0.0
strict JSON: 0 / 30
```

Then immediately explain:

> The adapted model improved substantially, but improvement was not enough to satisfy every preregistered absolute promotion threshold.

---

# 34. Show lifecycle decision

Present:

```text
evaluation_status: completed
promotion_decision: not_promoted
```

Explain:

> The evaluation succeeded as an experiment. The model was not promoted.

This is one of the strongest AI Engineering signals in the project because it shows governance discipline.

---

# 35. Explain benchmark discipline

Show only frozen evidence.

Do not execute the protected benchmark.

Explain:

```text
Hospital benchmark
CONSUMED / NEVER REPLAY
```

Suggested wording:

> Once the official result was observed, the benchmark stopped being valid as fresh independent evidence for the same candidate. Future validation requires new independent data.

This demonstrates evaluation maturity.

---

# 36. Show architecture documentation

Open:

```text
README.md
ARCHITECTURE.md
SECURITY.md
AI_ENGINEERING_CASE_STUDY.md
```

Briefly explain that the final capstone phase consolidates the implementation into auditable documentation.

Do not spend too long reading documents live.

---

# 37. Show local-first security

Summarize the local LLM security model.

Key points:

```text
local-only model egress
redirect rejection
proxy bypass
raw tabular rows forbidden
semantic sample limited to 5 values
local frontend CORS
sanitized public errors
trusted model loading
```

If useful, show `.env.example` to demonstrate reproducible configuration.

---

# 38. Mention security limits

A strong demo should include limits.

Suggested wording:

> DataLens is security-aware, but I do not claim that the current local capstone provides complete prompt-injection defense, general PII detection, or multi-user authentication. Those are explicit hardening areas.

This increases credibility.

---

# 39. Show CI/CD

Open the GitHub Actions workflow list or repository workflow directory.

Show:

```text
datalens-evals.yml
datalens-runtime.yml
datalens-publish.yml
datalens-release.yml
```

Explain the separation:

```text
evaluation
runtime validation
publishing
release
```

Then add:

> Application release and model promotion are separate authorities.

---

# 40. Short architecture summary

Close the technical walkthrough with this chain:

```text
Data
  |
Preparation
  |
Validation
  |
Planning
  |
Deterministic execution
  |
Evidence
  |
Explanation / report
  |
Model training
  |
Evaluation
  |
Artifact persistence
  |
Monitoring
  |
Lifecycle governance
```

---

# 41. Recruiter-facing closing statement

Suggested closing:

> DataLens is the project I use to connect the different skills required for AI Engineering. It includes the application layer, deterministic analytics, local LLM integration, RAG, ML, deep learning, adaptation, evaluation, observability, security, Docker, CI/CD, and model lifecycle governance.
>
> The most important design choice is that AI increases capability without replacing evidence or authority.

---

# 42. Three-minute ultra-short demo

If time is extremely limited, use this version.

## Minute 1

Show:

```text
Preparation
Analysis
Reporting
```

Explain deterministic authority.

## Minute 2

Show:

```text
Model Lab
Monitoring
Observability
```

Explain artifact persistence and model lifecycle.

## Minute 3

Show:

```text
QLoRA evaluation
completed / not_promoted
CONSUMED / NEVER REPLAY
```

Explain independent benchmark discipline.

Close with the architecture principle.

---

# 43. Ten-minute recommended demo

Suggested timing:

```text
00:00 - 01:00   Product / architecture
01:00 - 03:00   Preparation
03:00 - 05:00   Analysis / report
05:00 - 07:00   Model Lab / monitoring
07:00 - 08:00   Observability / security
08:00 - 09:30   QLoRA / lifecycle
09:30 - 10:00   Closing
```

---

# 44. Twenty-minute technical demo

Suggested timing:

```text
00:00 - 02:00   Architecture overview
02:00 - 05:00   Preparation workflow
05:00 - 08:00   Analytical planning / execution
08:00 - 10:00   RAG / reporting
10:00 - 13:00   Classical ML / Model Lab
13:00 - 15:00   Deep learning
15:00 - 17:00   Monitoring / observability
17:00 - 19:00   QLoRA / lifecycle
19:00 - 20:00   Security / CI / conclusion
```

---

# 45. What not to do during the demo

Avoid:

- retraining the QLoRA model live;
- replaying the Hospital benchmark;
- changing lifecycle thresholds;
- showing secret or private paths unnecessarily;
- exposing protected gold labels;
- using a large unpredictable dataset;
- spending several minutes scrolling code;
- claiming unsupported production readiness;
- claiming prompt-injection protection that is not implemented;
- describing `not_promoted` as a failed evaluation.

---

# 46. Demo dataset criteria

A good demo dataset should be:

- non-sensitive;
- small;
- predictable;
- understandable;
- rich enough for grouped analysis;
- rich enough for at least one ML task.

Useful columns may include:

```text
date
customer_id
category
segment
quantity
revenue
target
```

The exact dataset is less important than the reliability of the flow.

---

# 47. Demo reliability strategy

For a recorded or live presentation:

1. use frozen known-good input;
2. avoid network dependencies;
3. start Ollama before the demo;
4. verify Docker healthchecks;
5. pre-warm local models if needed;
6. avoid expensive live training;
7. prefer persisted artifacts for ML/DL evidence;
8. keep one fallback analytical request;
9. keep one fallback report;
10. keep the QLoRA result frozen and read-only.

---

# 48. Evidence to keep visible

Useful evidence during the demo includes:

```text
validated preparation state
analytical result
chart
report
model metric
artifact/model detail
monitoring signal
trace record
lifecycle decision
```

These provide stronger proof than code scrolling alone.

---

# 49. Portfolio screenshots

Recommended screenshots for the portfolio:

1. Preparation workflow
2. Analytical result with visualization
3. Report view
4. Model Lab model detail
5. Monitoring / health panel
6. Observability / trace explorer
7. Architecture diagram
8. QLoRA lifecycle evidence

Avoid screenshots containing private local paths or protected evaluation data.

---

# 50. Portfolio video structure

Recommended short video sequence:

```text
Title
  |
Preparation
  |
Analysis
  |
Report
  |
Model Lab
  |
Monitoring
  |
Observability
  |
QLoRA lifecycle
  |
Architecture summary
```

Keep transitions fast.

The video should communicate the system, not every implementation detail.

---

# 51. Interview deep-dive topics

If an interviewer asks for more detail, good topics include:

## Why deterministic analytics?

Because LLMs are not reliable numerical execution engines.

## Why local LLM?

Privacy, controllability, reproducibility, and experimentation.

## Why explicit payload classes?

To make model-visible data policy enforceable and testable.

## Why lifecycle registries?

Because model filenames do not encode trustworthy provenance or evaluation state.

## Why single-use holdout?

To preserve independent evaluation evidence.

## Why non-root containers?

To reduce runtime privilege.

## Why separate CI release from model promotion?

Because application deployment and model suitability are different decisions.

---

# 52. Expected questions and concise answers

## “Why not just use an LLM agent for everything?”

Because deterministic computation, persistent state, provenance, and protected evaluation should not depend on generated text.

## “Why Ollama?”

It supports local-first model execution and makes privacy boundaries easier to control.

## “Why was the QLoRA model not promoted?”

It improved significantly, but it did not satisfy every preregistered absolute promotion threshold.

## “Why not rerun the benchmark after improving the model?”

Because once results are observed, the same benchmark is no longer fresh independent evidence.

## “Does DataLens prevent prompt injection?”

It has strong surrounding controls, but a dedicated complete prompt-injection defense is not currently claimed.

## “Is this production ready?”

It demonstrates production-oriented engineering patterns, but the current project remains a local-first capstone rather than a fully hardened multi-tenant SaaS platform.

---

# 53. Demo evidence map

| Capability | Demo evidence |
|---|---|
| Data preparation | Preparation workflow |
| Deterministic analytics | Computed result |
| Structured AI | Planning / interpretation |
| RAG | Document retrieval / relevance |
| Reporting | Report output |
| Classical ML | Model Lab |
| Deep learning | Neural artifact/result |
| Model persistence | Trusted model detail |
| Monitoring | Drift/performance/health |
| Observability | Trace Explorer |
| Privacy | Trace / payload policy |
| QLoRA | Frozen adaptation result |
| Evaluation | Official protected result |
| Lifecycle | completed / not_promoted |
| Docker | Local stack |
| CI/CD | GitHub Actions workflows |

---

# 54. Final demo checklist

Before presenting:

```text
[ ] Repository branch verified
[ ] Working tree state understood
[ ] Web running
[ ] API healthy
[ ] Ollama ready if required
[ ] Demo dataset ready
[ ] Preparation flow tested
[ ] Analytical request tested
[ ] Report tested
[ ] Model Lab artifact available
[ ] Monitoring view available
[ ] Observability view available
[ ] Deep-learning evidence available
[ ] QLoRA result frozen
[ ] Hospital benchmark NOT executed
[ ] No secrets visible
[ ] No private data visible
[ ] Closing statement prepared
```

---

# 55. Final message

The strongest DataLens demonstration is not:

```text
“Here is a model that produces an answer.”
```

It is:

```text
“Here is a system that knows which parts may be generated,
which parts must be computed,
which evidence is trusted,
which artifacts are governed,
which results are observable,
and when a model should not be promoted.”
```

That is the core AI Engineering story of DataLens.
