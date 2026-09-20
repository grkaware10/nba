from pathlib import Path

import pandas as pd

from insurance_nba.data import create_project_dataset
from insurance_nba.workflow import (
    benchmark_models,
    build_data_dictionary,
    build_eda_report,
    build_monitoring_governance_report,
    run_full_workflow,
)


def test_generated_dataset_contains_missing_values() -> None:
    dataset = create_project_dataset(customer_count=180, seed=21)
    assert dataset.isna().sum().sum() > 0
    assert {"monthly_income", "annual_premium", "offer_code", "responded", "observation_date"}.issubset(dataset.columns)
    assert pd.to_datetime(dataset["observation_date"], errors="coerce").notna().all()


def test_benchmark_returns_all_requested_models() -> None:
    dataset = create_project_dataset(customer_count=250, seed=22)
    comparison = benchmark_models(dataset)
    assert set(comparison["model_name"]) == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
    }
    assert set(comparison["imputation_strategy"]) == {"mean", "median", "mode", "knn"}
    assert set(comparison["validation_strategy"]) == {"time_aware_holdout"}
    assert set(comparison["calibration_method"]) == {"sigmoid"}
    assert set(comparison["calibrated"]) == {"true"}
    assert ((comparison["f1"] >= comparison[["precision", "recall"]].min(axis=1)) & (comparison["f1"] <= comparison[["precision", "recall"]].max(axis=1))).all()


def test_eda_report_mentions_skew_and_imputation() -> None:
    dataset = create_project_dataset(customer_count=160, seed=23)
    report = build_eda_report(dataset)
    assert "Yeo-Johnson" in report
    assert "KNN" in report
    assert "time-aware holdout" in report
    assert "calibrated" in report


def test_generated_documentation_mentions_schema_and_governance() -> None:
    dataset = create_project_dataset(customer_count=140, seed=24)
    comparison = benchmark_models(dataset)
    dictionary_report = build_data_dictionary(dataset)
    governance_report = build_monitoring_governance_report(dataset, comparison)

    assert "customer-plus-offer-plus-observation-date" in dictionary_report
    assert "| observation_date |" in dictionary_report
    assert "Monitoring KPIs" in governance_report
    assert "audit logs" in governance_report


def test_full_workflow_generates_all_expected_artifacts(tmp_path: Path) -> None:
    outputs = run_full_workflow(tmp_path, customer_count=120, seed=25)

    assert outputs.dataset_path.exists()
    assert outputs.eda_report_path.exists()
    assert outputs.comparison_path.exists()
    assert outputs.recommendation_path.exists()
    assert outputs.data_dictionary_path.exists()
    assert outputs.governance_report_path.exists()