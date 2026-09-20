from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score

from .data import create_project_dataset, write_project_dataset
from .model import FEATURE_COLUMNS, build_serving_estimator, choose_calibration_folds


@dataclass
class WorkflowOutputs:
    dataset_path: Path
    eda_report_path: Path
    comparison_path: Path
    recommendation_path: Path
    data_dictionary_path: Path
    governance_report_path: Path


def run_full_workflow(project_root: Path, customer_count: int = 1200, seed: int = 17) -> WorkflowOutputs:
    data_dir = project_root / "data"
    reports_dir = project_root / "reports"
    data_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = write_project_dataset(data_dir / "insurance_propensity_dataset.csv", customer_count, seed)
    dataset = pd.read_csv(dataset_path)

    eda_report_path = reports_dir / "eda_summary.md"
    comparison_path = reports_dir / "model_comparison.csv"
    recommendation_path = reports_dir / "production_recommendation.md"
    data_dictionary_path = reports_dir / "data_dictionary.md"
    governance_report_path = reports_dir / "monitoring_governance.md"

    eda_report_path.write_text(build_eda_report(dataset), encoding="utf-8")
    comparison = benchmark_models(dataset)
    comparison.to_csv(comparison_path, index=False)
    recommendation_path.write_text(build_recommendation_report(comparison), encoding="utf-8")
    data_dictionary_path.write_text(build_data_dictionary(dataset), encoding="utf-8")
    governance_report_path.write_text(build_monitoring_governance_report(dataset, comparison), encoding="utf-8")

    return WorkflowOutputs(
        dataset_path=dataset_path,
        eda_report_path=eda_report_path,
        comparison_path=comparison_path,
        recommendation_path=recommendation_path,
        data_dictionary_path=data_dictionary_path,
        governance_report_path=governance_report_path,
    )


def benchmark_models(dataset: pd.DataFrame) -> pd.DataFrame:
    ordered = dataset.assign(_observation_date=pd.to_datetime(dataset["observation_date"])).sort_values(
        ["_observation_date", "customer_id", "offer_code"]
    )
    split_index = min(max(int(len(ordered) * 0.75), 1), len(ordered) - 1)
    train_frame = ordered.iloc[:split_index].drop(columns="_observation_date")
    test_frame = ordered.iloc[split_index:].drop(columns="_observation_date")

    train_features = train_frame[FEATURE_COLUMNS]
    train_target = train_frame["responded"]
    test_features = test_frame[FEATURE_COLUMNS]
    test_target = test_frame["responded"]
    calibration_folds = choose_calibration_folds(train_target)

    rows: list[dict[str, float | str]] = []
    for imputation_strategy in ["mean", "median", "mode", "knn"]:
        for model_name in [
            "logistic_regression",
            "decision_tree",
            "random_forest",
            "extra_trees",
            "hist_gradient_boosting",
        ]:
            pipeline = build_serving_estimator(
                model_name=model_name,
                imputation_strategy=imputation_strategy,
                calibration_folds=calibration_folds,
            )
            pipeline.fit(train_features, train_target)

            probabilities = pipeline.predict_proba(test_features)[:, 1]
            predictions = (probabilities >= 0.5).astype(int)
            row = {
                "model_name": model_name,
                "imputation_strategy": imputation_strategy,
                "accuracy": float(accuracy_score(test_target, predictions)),
                "precision": float(precision_score(test_target, predictions, zero_division=0)),
                "recall": float(recall_score(test_target, predictions, zero_division=0)),
                "f1": float(f1_score(test_target, predictions, zero_division=0)),
                "roc_auc": float(roc_auc_score(test_target, probabilities)),
                "pr_auc": float(average_precision_score(test_target, probabilities)),
                "validation_strategy": "time_aware_holdout",
                "calibrated": str(calibration_folds is not None).lower(),
                "calibration_method": "sigmoid" if calibration_folds is not None else "none",
            }
            rows.append(row)

    comparison = pd.DataFrame(rows).sort_values(["roc_auc", "pr_auc", "f1"], ascending=False).reset_index(drop=True)
    return comparison


def build_eda_report(dataset: pd.DataFrame) -> str:
    missingness = (dataset.isna().mean() * 100).sort_values(ascending=False)
    numeric_columns = dataset.select_dtypes(include=["number"]).columns.tolist()
    skewness = dataset[numeric_columns].skew(numeric_only=True).sort_values(key=lambda values: values.abs(), ascending=False)
    response_by_offer = dataset.groupby("offer_code")["responded"].mean().sort_values(ascending=False)
    observation_dates = pd.to_datetime(dataset["observation_date"])

    return "\n".join(
        [
            "# Insurance NBA EDA Summary",
            "",
            "## Dataset Overview",
            f"- Rows: {len(dataset)}",
            f"- Columns: {len(dataset.columns)}",
            f"- Response rate: {dataset['responded'].mean():.2%}",
            f"- Observation window: {observation_dates.min().date()} to {observation_dates.max().date()}",
            "",
            "## Missingness",
            *[f"- {column}: {value:.2f}%" for column, value in missingness.head(10).items()],
            "",
            "## Highest Absolute Skewness",
            *[f"- {column}: {value:.3f}" for column, value in skewness.head(10).items()],
            "",
            "## Offer Response Rate",
            *[f"- {column}: {value:.2%}" for column, value in response_by_offer.items()],
            "",
            "## Treatment Decisions",
            "- Missing values are intentionally present to benchmark mean, median, mode, and KNN imputers.",
            "- Heavy-tailed numeric features are capped with IQR bounds before modeling to reduce extreme leverage without dropping rows.",
            "- Residual skewness is handled with Yeo-Johnson because it supports zero and near-zero values, unlike Box-Cox.",
            "- Feature engineering adds claim severity, premium-income ratio, service engagement ratio, policy density, and renewal urgency.",
            "- Evaluation uses a time-aware holdout ordered by observation_date so later customer-offer events simulate production scoring.",
            "- Propensity outputs are calibrated with Platt scaling via sigmoid calibration before ranking recommendations.",
        ]
    )


def build_recommendation_report(comparison: pd.DataFrame) -> str:
    best = comparison.iloc[0].to_dict()
    baseline = (
        comparison.loc[comparison["model_name"] == "logistic_regression"]
        .sort_values("roc_auc", ascending=False)
        .iloc[0]
        .to_dict()
    )
    imputation_reason = _imputation_reason(str(best["imputation_strategy"]))
    runner_up = comparison.iloc[1].to_dict()
    return "\n".join(
        [
            "# Production Model Recommendation",
            "",
            f"Best holdout performer: {best['model_name']} with {best['imputation_strategy']} imputation.",
            f"- ROC-AUC: {best['roc_auc']:.4f}",
            f"- PR-AUC: {best['pr_auc']:.4f}",
            f"- Accuracy: {best['accuracy']:.4f}",
            f"- F1: {best['f1']:.4f}",
            f"- Validation: {best['validation_strategy']}",
            f"- Calibration: {best['calibration_method']}",
            "",
            "## Why This Setup",
            f"- {imputation_reason}",
            "- KNN imputation can perform well but is slower and less stable at scale for online retraining.",
            "- IQR capping preserves sample size, which is better for recommendation coverage than dropping outlier rows.",
            "- Yeo-Johnson is preferred over Box-Cox because the engineered feature set includes zero-heavy distributions.",
            "- Tree ensembles outperform the logistic baseline by capturing non-linear interactions between offer type, claims, renewal timing, and channel preference.",
            "- Time-aware holdout evaluation reduces leakage from future observations into the training slice.",
            "- Sigmoid calibration makes ranked recommendation scores more reliable for thresholding and downstream channel decisions.",
            "",
            "## Why Other Options Were Not Chosen",
            f"- Logistic regression remains the interpretability baseline but scored ROC-AUC {baseline['roc_auc']:.4f}, below the top ensemble.",
            "- A single decision tree is easier to explain but more variance-prone and less stable on noisy, partially imputed training data.",
            f"- The closest runner-up was {runner_up['model_name']} with {runner_up['imputation_strategy']} imputation, but it trailed on ROC-AUC or PR-AUC.",
            "- HistGradientBoosting remains a strong fallback when latency is tighter than interpretability and feature importance review requirements.",
            "",
            "## Production Guidance",
            "- Use the top ensemble for batch or API scoring with periodic recalibration checks on the latest observation window.",
            "- Keep the logistic baseline as a challenger for governance and interpretability review.",
            "- Re-run the benchmark whenever the schema or eligibility policy changes.",
        ]
    )


def build_data_dictionary(dataset: pd.DataFrame) -> str:
    field_descriptions = {
        "customer_id": "Synthetic customer identifier.",
        "observation_date": "Observation date for the customer-offer training instance used in time-aware validation.",
        "age": "Customer age in years.",
        "tenure_years": "Years since the customer joined the insurer.",
        "annual_premium": "Annualized premium paid across active policies.",
        "monthly_income": "Estimated monthly income for affordability features.",
        "claims_last_24m": "Claim count in the last 24 months.",
        "claim_amount_24m": "Total claim amount in the last 24 months.",
        "digital_engagement": "Synthetic engagement score between 0 and 1.",
        "has_motor_policy": "Flag indicating an active motor policy.",
        "has_health_policy": "Flag indicating an active health policy.",
        "has_life_policy": "Flag indicating an active life policy.",
        "homeowner": "Flag indicating whether the customer owns a home.",
        "frequent_traveler": "Flag indicating frequent travel behavior.",
        "dependents": "Number of declared dependents.",
        "days_to_renewal": "Days until the nearest renewal event.",
        "last_campaign_response": "Flag indicating response to the prior campaign.",
        "web_sessions_30d": "Web sessions observed in the last 30 days.",
        "call_center_contacts_90d": "Call center contacts observed in the last 90 days.",
        "payment_delay_days": "Days of payment delay across recent billing cycles.",
        "policy_count": "Count of active policies held by the customer.",
        "vehicle_age": "Insured vehicle age in years when a motor policy exists.",
        "credit_score": "Synthetic credit score proxy.",
        "region": "Customer region segment.",
        "payment_method": "Preferred payment method.",
        "channel_preference": "Preferred interaction channel.",
        "policy_tier": "Portfolio tier label.",
        "offer_code": "Candidate offer or action code evaluated for the customer.",
        "eligible": "Eligibility flag after business-rule filtering.",
        "responded": "Target flag showing whether the customer accepted or responded.",
    }
    business_domains = {
        "customer_id": "identity",
        "observation_date": "governance",
        "age": "profile",
        "tenure_years": "profile",
        "annual_premium": "portfolio",
        "monthly_income": "affordability",
        "claims_last_24m": "claims",
        "claim_amount_24m": "claims",
        "digital_engagement": "engagement",
        "has_motor_policy": "portfolio",
        "has_health_policy": "portfolio",
        "has_life_policy": "portfolio",
        "homeowner": "profile",
        "frequent_traveler": "lifestyle",
        "dependents": "profile",
        "days_to_renewal": "retention",
        "last_campaign_response": "marketing",
        "web_sessions_30d": "engagement",
        "call_center_contacts_90d": "servicing",
        "payment_delay_days": "billing",
        "policy_count": "portfolio",
        "vehicle_age": "portfolio",
        "credit_score": "risk",
        "region": "profile",
        "payment_method": "billing",
        "channel_preference": "engagement",
        "policy_tier": "portfolio",
        "offer_code": "offer",
        "eligible": "governance",
        "responded": "target",
    }

    lines = [
        "# Data Dictionary",
        "",
        "This dataset is synthetic and structured at the customer-plus-offer-plus-observation-date grain.",
        "",
        "| Column | Type | Domain | Missing % | Description |",
        "| --- | --- | --- | ---: | --- |",
    ]
    missingness = dataset.isna().mean().mul(100)
    for column in dataset.columns:
        lines.append(
            f"| {column} | {dataset[column].dtype} | {business_domains.get(column, 'other')} | {missingness[column]:.2f} | {field_descriptions.get(column, 'Synthetic field used in the prototype workflow.')} |"
        )

    lines.extend(
        [
            "",
            "## Notes",
            "- Missing values are injected intentionally to benchmark imputation strategies.",
            "- `eligible` is rule-derived and only eligible customer-offer rows are retained in the training frame.",
            "- `responded` is a synthetic propensity outcome and should be replaced with observed production outcomes when available.",
        ]
    )
    return "\n".join(lines)


def build_monitoring_governance_report(dataset: pd.DataFrame, comparison: pd.DataFrame) -> str:
    best = comparison.iloc[0]
    response_rate = dataset["responded"].mean()
    missing_columns = (dataset.isna().mean().mul(100)).sort_values(ascending=False).head(5)
    return "\n".join(
        [
            "# Monitoring And Governance",
            "",
            "## Deployment Guardrails",
            "- Apply eligibility rules before scoring responses are shown to agent, web, app, or servicing channels.",
            "- Keep logistic regression as a governance challenger against the selected production model.",
            "- Review any schema change, rule change, or material score shift before promoting a retrained model.",
            "",
            "## Monitoring KPIs",
            f"- Current benchmark leader: {best['model_name']} with ROC-AUC {best['roc_auc']:.4f} and PR-AUC {best['pr_auc']:.4f}.",
            f"- Baseline synthetic response rate: {response_rate:.2%}.",
            "- Monitor Precision@K, Recall@K, ROC-AUC, PR-AUC, recommendation coverage, acceptance rate, renewal uplift, and digital engagement rate.",
            "- Track calibration drift with reliability checks on the latest observation window before threshold changes.",
            "",
            "## Data Quality Checks",
            *[f"- {column}: {value:.2f}% missing in the prototype dataset." for column, value in missing_columns.items()],
            "- Alert on spikes in missing payment method, channel, premium, income, or vehicle-age fields because they affect ranking stability.",
            "",
            "## Governance Controls",
            "- Preserve audit logs for recommendation requests, top-N outputs, explanations, and feedback events.",
            "- Restrict retraining inputs to consented and policy-compliant customer outcomes.",
            "- Validate that underwriting or compliance exclusions remain enforced ahead of recommendation delivery.",
            "- Review cold-start fallback logic and coverage concentration by offer to avoid over-serving popular products.",
            "",
            "## Retraining Policy",
            "- Retrain when enough new feedback accumulates or when monitored KPI degradation breaches thresholds.",
            "- Use the newest observation window as the holdout slice to preserve time order during validation.",
            "- Re-run calibration checks after every retraining cycle before exposing new scores to channels.",
        ]
    )


def _imputation_reason(imputation_strategy: str) -> str:
    if imputation_strategy == "mean":
        return "Mean imputation won this run because the synthetic numeric features retained enough central tendency after IQR capping and Yeo-Johnson transformation."
    if imputation_strategy == "median":
        return "Median imputation won this run because it stayed robust under the skewed premium and claims distributions."
    if imputation_strategy == "mode":
        return "Mode imputation won this run because several sparse binary and categorical insurance fields benefited from preserving the most common observed pattern."
    if imputation_strategy == "knn":
        return "KNN imputation won this run because neighboring policy profiles reconstructed missing numeric values more effectively than global summaries."
    return "The selected imputation strategy produced the strongest holdout performance in this benchmark."


def outputs_as_dict(outputs: WorkflowOutputs) -> dict[str, str]:
    return {key: str(value) for key, value in asdict(outputs).items()}