# Insurance Next Best Action

This repository adapts the banking-style next best action brief into an insurance-domain starter implementation.

It includes:

- an insurance project brief
- a synthetic propensity dataset generator with dated customer-offer records, missing values, skew, and outliers
- EDA and feature-engineering workflow
- imputation benchmarking across mean, median, mode, and KNN strategies
- model comparison across logistic regression, decision tree, random forest, extra trees, and histogram gradient boosting
- time-aware validation and probability calibration for ranked propensity scores
- deterministic eligibility rules and ranking service
- feedback capture and retraining trigger for closed-loop learning
- recommendation responses with business-driver explanations
- generated data dictionary and monitoring/governance artifacts
- a small FastAPI surface

## Quick start

```powershell
.\.venv\Scripts\python -m pip install -e .[dev]
.\.venv\Scripts\python -m pytest
.\.venv\Scripts\python -m insurance_nba.run_workflow
.\.venv\Scripts\python -m uvicorn insurance_nba.api:app --reload
```

The API exposes:

- `GET /health`
- `GET /metrics/model`
- `GET /recommendations/{customer_id}?top_n=3`
- `POST /feedback/{customer_id}`
- `POST /retrain`

## Project structure

- `docs/insurance_nba_project_brief.md`
- `src/insurance_nba/`
- `data/insurance_propensity_dataset.csv`
- `reports/eda_summary.md`
- `reports/model_comparison.csv`
- `reports/production_recommendation.md`
- `reports/data_dictionary.md`
- `reports/monitoring_governance.md`
- `tests/`

## Workflow Summary

The ML workflow is designed around an insurance propensity problem at the customer-offer level.

- Data collection: generate a project dataset with customer, policy, claims, engagement, channel, and offer attributes.
- Validation anchor: stamp each customer-offer record with an observation date for ordered holdout evaluation.
- EDA: summarize missingness, skewness, response balance, and offer-level conversion behavior.
- Missing values: compare mean, median, mode, and KNN imputers.
- Outliers: cap numeric features with IQR bounds rather than dropping records.
- Skewness: apply Yeo-Johnson after capping because the dataset contains zero-heavy and right-skewed fields.
- Feature engineering: add claim severity, premium-to-income ratio, service engagement ratio, policy density, and renewal urgency.
- Feature selection: use tree-based selection before model fitting.
- Model evaluation: compare multiple classifiers on a time-aware holdout using calibrated probabilities and report ROC-AUC, PR-AUC, accuracy, precision, recall, and F1.
- Feedback loop: capture realized outcomes and append them to retraining data before refreshing the production model.
- Explanation layer: return concise business-driver reasons alongside each ranked recommendation.

If you have a Kaggle or enterprise dataset later, replace the generated CSV and rerun the same workflow.
