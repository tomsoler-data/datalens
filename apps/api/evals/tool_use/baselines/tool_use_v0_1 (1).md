# DataLens tool-use baseline v0.1

Frozen from local run:

`20260817T103955_435556`

## Scope

- 6 controlled synthetic cases
- 3 repetitions per case
- 18 total runs
- 5 executable native analytical families
- 1 explicit-column fidelity guardrail case

## Baseline

- Overall pass rate: 100%
- Planner family accuracy: 100%
- Planner binding accuracy: 100%
- Planner first-pass rate: 83.3%
- Planner first-pass rate on executable cases: 100%
- Planner retry rate: 16.7%
- Execution-recovery retry rate: 0%
- Guardrail retry rate: 100%
- Native tool selection accuracy: 100%
- Native argument accuracy: 100%
- Native first-pass rate: 100%
- Execution success rate: 100%
- Chart type accuracy: 100%
- Guardrail accuracy: 100%
- Median latency: 4.02 s

The remaining retry belongs to the intentionally invalid guardrail case. It ends in blocked/unresolved abstention with no native tool execution.

## Important interpretation

This baseline proves behavior only on the controlled synthetic benchmark used here. It is **not** evidence that DataLens is generally 100% reliable.

The next benchmark expands across a second domain so the architecture is tested beyond one HR-shaped fixture.
