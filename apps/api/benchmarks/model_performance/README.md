# DataLens — Local Model Performance Benchmark v0.1

Ce benchmark est séparé des baselines de qualité `tool_use_*` déjà gelées.

Objectif : mesurer, avec les mêmes cas contrôlés, le compromis **qualité / latence**
du couple de modèles locaux utilisé par DataLens.

Cas couverts :

1. quantitative association
2. categorical association
3. group comparison
4. distribution
5. time series
6. guardrail sur une colonne explicitement absente

Par défaut :

- Planner : `gemma3:4b`
- Tool calling : `qwen2.5:1.5b-instruct`
- 3 répétitions
- 1 warmup
- les traces produit `ai_traces.jsonl` sont désactivées pendant le benchmark

Le benchmark écrit dans un dossier horodaté :

- `metadata.json`
- `results.csv`
- `summary.json`
- `raw/*.json`

## Installation

Créer :

`C:\Users\tomas\datalens\apps\api\benchmarks\model_performance`

Y placer :

- `run_model_performance_benchmark.py`
- `cases_model_performance_v0_1.json`
- `datalens_hr_benchmark.csv`

## Validation syntaxique

Depuis `apps\api` :

```powershell
python -m py_compile .\benchmarks\model_performance\run_model_performance_benchmark.py
```

## Baseline actuelle

```powershell
python .\benchmarks\model_performance\run_model_performance_benchmark.py
```

Cela exécute 6 cas × 3 répétitions = 18 exécutions mesurées, après 1 warmup.

## Plus tard : comparer une autre paire

```powershell
python .\benchmarks\model_performance\run_model_performance_benchmark.py `
    --planner-model "AUTRE_MODELE_PLANNER" `
    --tool-model "AUTRE_MODELE_TOOL"
```

On ne retient jamais un modèle uniquement parce qu'il est plus rapide.
Le choix doit préserver les garde-fous, les bindings exacts et le tool calling validé.
