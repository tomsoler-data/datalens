# DataLens Evals

DataLens uses versioned evaluation pipelines to protect analytical quality and detect regressions.

## Discovery and Prioritization

The current evaluation stack covers:

- analytical discovery expectations;
- prioritization decisions;
- analytical guardrails;
- deterministic execution;
- prioritization reason-code coverage;
- frozen regression baselines.

## Run the eval suite

From `apps/api`:

```powershell
python -m app.evals.suite_runner
```

Generate the machine-readable report:

```powershell
python -m app.evals.suite_runner `
    --json-output .\evals_report.json
```

## Run the regression gate

```powershell
python -m app.evals.regression_gate `
    --baseline .\app\evals\baselines\discovery_prioritization_v0_1.json `
    --report .\evals_report.json
```

The regression gate returns a non-zero exit code when the frozen quality contract is violated.

## Current contracts

- Analysis Benchmark: `analysis_benchmark_v0.1`
- Eval Suite: `eval_suite_v0.2`
- Eval Coverage: `eval_coverage_v0.1`
- Eval Regression Gate: `eval_regression_gate_v0.1`
- CI Evals Gate: `datalens_ci_evals_gate_v0.1`

## CI

GitHub Actions runs the evaluation suite and regression gate automatically for backend changes.

The required status check for the protected `main` branch is:

`Discovery & Prioritization Evals`
