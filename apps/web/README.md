# DataLens Web

DataLens Web is the Next.js frontend for the DataLens local-first AI Engineering capstone.

It provides the product-facing workspace for dataset import, preparation, deterministic analysis, document-assisted workflows, reporting, Model Lab, monitoring, and observability while the FastAPI backend remains the authority for execution, trusted state, and lifecycle evidence.

## Stack

- Next.js 16
- React 19
- TypeScript
- CSS Modules
- ESLint
- Node.js 22 in the containerized runtime

The current package versions are defined in `package.json`.

## Frontend responsibilities

The web application is responsible for interaction and presentation.

Its major areas include:

- dataset workspace and import controls;
- multi-step data preparation;
- analysis execution and requested-analysis follow-up;
- findings, quality information, and blocked-analysis visibility;
- business-document and RAG-facing presentation;
- report synthesis actions;
- Model Lab training and prediction workflows;
- model monitoring history;
- model observability;
- workflow history and product-facing state.

The frontend does not replace backend analytical authority.

Numerical results, preparation state, model artifacts, lifecycle decisions, and protected evaluation evidence remain backend-owned.

## Main source areas

```text
apps/web/
├── src/
│   ├── app/
│   │   ├── WorkspaceClient.tsx
│   │   ├── DatasetImportControls.tsx
│   │   ├── DatasetWorkspaceSection.tsx
│   │   ├── BusinessDocumentsSection.tsx
│   │   └── ...
│   └── components/
│       ├── analysis/
│       ├── modelLab/
│       ├── preparation/
│       └── workspace/
├── package.json
├── Dockerfile
└── README.md
```

### Preparation

The preparation UI follows the DataLens preparation workflow rather than a generic upload-and-run pattern.

Current frontend components include dedicated views for:

- understanding;
- cleaning plans;
- transformations;
- dataset combination;
- finalization;
- resolved stages;
- preparation history and navigation.

These views present server-owned preparation state and decisions.

### Analysis

The analysis area contains dedicated UI for:

- analysis execution;
- requested analyses;
- follow-up history;
- findings;
- quality information;
- blocked analyses;
- document-request summaries;
- RAG/report summaries;
- audit disclosures.

The frontend presents deterministic and AI-assisted outputs while preserving their different roles.

### Model Lab

Model Lab provides the frontend surface for model-oriented workflows.

The current component set includes API/types for training and prediction together with UI for:

- model monitoring history;
- model observability;
- artifact- and workflow-oriented model interactions.

Deep-learning workflows are exposed through the broader Model Lab product model rather than a separate top-level deep-learning application.

## Backend connection

The browser connects to the DataLens API through:

```text
NEXT_PUBLIC_DATALENS_API_URL
```

For the standard local setup:

```text
http://127.0.0.1:8000
```

The repository-level `.env.example` documents the supported environment reference.

## Local development

From the repository root:

```bash
cd apps/web
npm install
npm run dev
```

Then open:

```text
http://localhost:3000
```

The API should normally be available at:

```text
http://127.0.0.1:8000
```

## Quality checks

Run frontend linting with:

```bash
npm run lint
```

Create a production build with:

```bash
npm run build
```

Run the production server after building with:

```bash
npm run start
```

## Docker

The frontend has its own `Dockerfile` and is also part of the repository-level Docker Compose stack.

The containerized web runtime is exposed on the loopback interface by the root compose configuration.

For the full local stack, use the repository-level instructions in the root README.

## Engineering boundaries

### Deterministic authority

The UI should not invent numerical answers that belong to deterministic analytical execution.

### Server-owned preparation state

Preparation state and validated outputs are backend-owned. The browser presents and requests transitions; it is not the source of truth.

### Model lifecycle evidence

Model identity, evaluation status, promotion decisions, and protected benchmark evidence are backend lifecycle concerns.

The official QLoRA lifecycle state remains:

```text
evaluation_status: completed
promotion_decision: not_promoted
```

The protected Hospital benchmark remains:

```text
CONSUMED / NEVER REPLAY
```

The frontend must not provide a path that replays or silently mutates that evidence.

### Privacy

Security and LLM-egress enforcement live in the backend. Frontend changes should preserve those boundaries and avoid introducing direct external model calls that bypass DataLens controls.

## Related documentation

From `apps/web/`:

- [Project README](../../README.md)
- [Architecture](../../ARCHITECTURE.md)
- [Security](../../SECURITY.md)
- [AI Engineering case study](../../AI_ENGINEERING_CASE_STUDY.md)
- [Demo walkthrough](../../DEMO_WALKTHROUGH.md)
- [Backend README](../api/README.md)
- [Environment reference](../../.env.example)

## Status

The frontend is an implemented part of the DataLens capstone, not a default Next.js starter.

The current product surface demonstrates integration across the Data Analyst workflow and AI Engineering layers: preparation, analysis, document context, reporting, Model Lab, monitoring, and observability.

Future work can focus on UX refinement, responsive polish, clearer deep-learning visibility, and portfolio presentation without changing the core authority model described above.
