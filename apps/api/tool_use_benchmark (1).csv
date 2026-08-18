# DataLens — Tool-use evals v0.1

Ce premier harness évalue le pipeline local complet :

1. Gemma propose un contrat analytique structuré.
2. Python valide la famille, les colonnes, les rôles et les garde-fous.
3. Qwen choisit un outil natif parmi le catalogue autorisé.
4. Python valide le nom de l’outil et ses arguments.
5. Le moteur déterministe exécute le calcul.

## Cas inclus

- quantitative_association
- categorical_association
- group_comparison
- distribution
- time_series
- guardrail de fidélité objectif → colonne (`Year` absent)

Le fixture est synthétique et déterministe. Il ne représente aucune donnée réelle.

## Installation dans le repo

Copier le dossier `evals` fourni dans :

`C:\Users\tomas\datalens\apps\api\evals`

Structure attendue :

```text
apps/api/
├─ app/
├─ evals/
│  └─ tool_use/
│     ├─ cases.json
│     ├─ run_tool_use_evals.py
│     └─ fixtures/
│        └─ tool_use_benchmark.csv
```

## Lancer une première passe

Depuis `C:\Users\tomas\datalens\apps\api` :

```powershell
python .\evals\tool_use\run_tool_use_evals.py
```

## Répéter chaque cas 3 fois

```powershell
python .\evals\tool_use\run_tool_use_evals.py --repeat 3
```

C’est plus utile qu’un seul run pour mesurer la stabilité des petits modèles locaux.

## Lancer un seul cas

```powershell
python .\evals\tool_use\run_tool_use_evals.py --case time_salary_snapshot
```

## Utilisation CI / régression stricte

```powershell
python .\evals\tool_use\run_tool_use_evals.py --repeat 3 --fail-on-regression
```

Cette option retourne un code de sortie non nul si au moins un run échoue.

## Résultats

Le runner crée :

```text
evals/tool_use/results/
├─ results.csv
├─ summary.json
└─ raw/
   ├─ quantitative_salary_training__r01.json
   └─ ...
```

`results.csv` permet de comparer les versions de modèles et de prompts.

`summary.json` contient les métriques agrégées.

Les JSON bruts conservent la trace complète pour diagnostiquer une régression.

## Métriques v0.1

- overall_pass_rate
- planner_outcome_accuracy
- planner_family_accuracy
- planner_binding_accuracy
- planner_first_pass_rate
- planner_retry_rate
- planner_retry_recovery_rate
- planner_normalization_rate
- native_tool_selection_accuracy
- native_argument_accuracy
- native_first_pass_rate
- native_retry_rate
- native_retry_recovery_rate
- execution_success_rate
- chart_type_accuracy
- guardrail_accuracy
- latence moyenne / médiane

Une réussite finale ne masque donc pas un retry : la fiabilité au premier essai est mesurée séparément.
