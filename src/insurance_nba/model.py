from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, PowerTransformer, StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURE_COLUMNS = [
    "age",
    "tenure_years",
    "annual_premium",
    "monthly_income",
    "claims_last_24m",
    "claim_amount_24m",
    "digital_engagement",
    "has_motor_policy",
    "has_health_policy",
    "has_life_policy",
    "homeowner",
    "frequent_traveler",
    "dependents",
    "days_to_renewal",
    "last_campaign_response",
    "web_sessions_30d",
    "call_center_contacts_90d",
    "payment_delay_days",
    "policy_count",
    "vehicle_age",
    "credit_score",
    "region",
    "payment_method",
    "channel_preference",
    "policy_tier",
    "offer_code",
]

NUMERIC_COLUMNS = [
    "age",
    "tenure_years",
    "annual_premium",
    "monthly_income",
    "claims_last_24m",
    "claim_amount_24m",
    "digital_engagement",
    "has_motor_policy",
    "has_health_policy",
    "has_life_policy",
    "homeowner",
    "frequent_traveler",
    "dependents",
    "days_to_renewal",
    "last_campaign_response",
    "web_sessions_30d",
    "call_center_contacts_90d",
    "payment_delay_days",
    "policy_count",
    "vehicle_age",
    "credit_score",
]

CATEGORICAL_COLUMNS = ["region", "payment_method", "channel_preference", "policy_tier", "offer_code"]

PRODUCTION_CONFIG = {"model_name": "random_forest", "imputation_strategy": "mode"}


class FeatureEngineeringTransformer:
    def fit(self, frame: pd.DataFrame, target: pd.Series | None = None) -> "FeatureEngineeringTransformer":
        return self

    def transform(self, frame: pd.DataFrame) -> pd.DataFrame:
        engineered = frame.copy()
        engineered["premium_income_ratio"] = engineered["annual_premium"] / (
            engineered["monthly_income"] * 12
        )
        engineered["claim_severity"] = engineered["claim_amount_24m"] / engineered["claims_last_24m"].replace(0, 1)
        engineered["engagement_to_service_ratio"] = engineered["web_sessions_30d"] / (
            engineered["call_center_contacts_90d"] + 1
        )
        engineered["policy_density"] = engineered["policy_count"] / (engineered["tenure_years"] + 1)
        engineered["renewal_urgency"] = 365 - engineered["days_to_renewal"]
        engineered["age_bucket"] = pd.cut(
            engineered["age"],
            bins=[18, 30, 40, 50, 60, 80],
            labels=["18_30", "31_40", "41_50", "51_60", "61_plus"],
            include_lowest=True,
        ).astype("object")
        return engineered


ENGINEERED_NUMERIC_COLUMNS = NUMERIC_COLUMNS + [
    "premium_income_ratio",
    "claim_severity",
    "engagement_to_service_ratio",
    "policy_density",
    "renewal_urgency",
]

ENGINEERED_CATEGORICAL_COLUMNS = CATEGORICAL_COLUMNS + ["age_bucket"]


class IQRCapper:
    def fit(self, values: pd.DataFrame, target: pd.Series | None = None) -> "IQRCapper":
        frame = pd.DataFrame(values).astype(float)
        quartile_1 = frame.quantile(0.25)
        quartile_3 = frame.quantile(0.75)
        iqr = quartile_3 - quartile_1
        self.lower_bounds_ = quartile_1 - 1.5 * iqr
        self.upper_bounds_ = quartile_3 + 1.5 * iqr
        return self

    def transform(self, values: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(values).astype(float)
        clipped = frame.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        return clipped.to_numpy()


class SkewnessHandler:
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold

    def fit(self, values: np.ndarray, target: pd.Series | None = None) -> "SkewnessHandler":
        frame = pd.DataFrame(values).astype(float)
        skewness = frame.skew(numeric_only=True)
        self.columns_ = [index for index, value in skewness.items() if abs(value) >= self.threshold]
        self.transformer_ = PowerTransformer(method="yeo-johnson", standardize=False)
        if self.columns_:
            self.transformer_.fit(frame[self.columns_])
        return self

    def transform(self, values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values).astype(float)
        if self.columns_:
            frame[self.columns_] = self.transformer_.transform(frame[self.columns_])
        return frame.to_numpy()


@dataclass
class TrainingArtifacts:
    pipeline: Any
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    pr_auc: float
    model_name: str
    imputation_strategy: str
    calibrated: bool
    calibration_method: str | None


def _build_numeric_imputer(imputation_strategy: str):
    if imputation_strategy == "mean":
        return SimpleImputer(strategy="mean")
    if imputation_strategy == "median":
        return SimpleImputer(strategy="median")
    if imputation_strategy == "mode":
        return SimpleImputer(strategy="most_frequent")
    if imputation_strategy == "knn":
        return KNNImputer(n_neighbors=5)
    raise ValueError(f"Unsupported imputation strategy: {imputation_strategy}")


def _build_model_registry() -> dict[str, Callable[[], object]]:
    return {
        "logistic_regression": lambda: LogisticRegression(max_iter=800, class_weight="balanced"),
        "decision_tree": lambda: DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, random_state=42),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced_subsample",
        ),
        "extra_trees": lambda: ExtraTreesClassifier(
            n_estimators=250,
            max_depth=10,
            min_samples_leaf=8,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced",
        ),
        "hist_gradient_boosting": lambda: HistGradientBoostingClassifier(
            max_depth=8,
            learning_rate=0.08,
            max_iter=250,
            random_state=42,
        ),
    }


def build_training_pipeline(model_name: str, imputation_strategy: str) -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", _build_numeric_imputer(imputation_strategy)),
            ("iqr", IQRCapper()),
            ("skew", SkewnessHandler()),
            ("scale", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, ENGINEERED_NUMERIC_COLUMNS),
            ("categorical", categorical_pipeline, ENGINEERED_CATEGORICAL_COLUMNS),
        ]
    )
    selector = SelectFromModel(
        estimator=ExtraTreesClassifier(n_estimators=120, random_state=42, n_jobs=-1),
        threshold="median",
    )

    return Pipeline(
        steps=[
            ("feature_engineering", FeatureEngineeringTransformer()),
            ("preprocessor", preprocessor),
            ("feature_selection", selector),
            ("model", _build_model_registry()[model_name]()),
        ]
    )


def choose_calibration_folds(target: pd.Series, max_folds: int = 3) -> int | None:
    class_counts = target.value_counts()
    if class_counts.empty:
        return None

    min_class_count = int(class_counts.min())
    if min_class_count < 2:
        return None

    return min(max_folds, min_class_count)


def build_serving_estimator(
    model_name: str,
    imputation_strategy: str,
    calibration_folds: int | None,
):
    pipeline = build_training_pipeline(model_name, imputation_strategy)
    if calibration_folds is None:
        return pipeline

    return CalibratedClassifierCV(
        estimator=pipeline,
        method="sigmoid",
        cv=calibration_folds,
        ensemble=False,
    )


def _score_artifacts(
    pipeline: Any,
    features: pd.DataFrame,
    target: pd.Series,
    model_name: str,
    imputation_strategy: str,
    calibrated: bool,
    calibration_method: str | None,
) -> TrainingArtifacts:
    probabilities = pipeline.predict_proba(features)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)
    return TrainingArtifacts(
        pipeline=pipeline,
        auc=roc_auc_score(target, probabilities),
        accuracy=accuracy_score(target, predictions),
        precision=precision_score(target, predictions, zero_division=0),
        recall=recall_score(target, predictions, zero_division=0),
        f1=f1_score(target, predictions, zero_division=0),
        pr_auc=average_precision_score(target, probabilities),
        model_name=model_name,
        imputation_strategy=imputation_strategy,
        calibrated=calibrated,
        calibration_method=calibration_method,
    )


def train_propensity_model(
    frame: pd.DataFrame,
    model_name: str | None = None,
    imputation_strategy: str | None = None,
) -> TrainingArtifacts:
    selected_model = model_name or PRODUCTION_CONFIG["model_name"]
    selected_imputer = imputation_strategy or PRODUCTION_CONFIG["imputation_strategy"]
    features = frame[FEATURE_COLUMNS]
    target = frame["responded"]
    calibration_folds = choose_calibration_folds(target)
    pipeline = build_serving_estimator(selected_model, selected_imputer, calibration_folds)
    pipeline.fit(features, target)
    return _score_artifacts(
        pipeline,
        features,
        target,
        selected_model,
        selected_imputer,
        calibrated=calibration_folds is not None,
        calibration_method="sigmoid" if calibration_folds is not None else None,
    )
