from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import UTC, datetime
import json
from pathlib import Path

import pandas as pd

from .data import build_training_frame, eligibility_for_customer, generate_customers
from .domain import OFFERS, OfferRule
from .model import FEATURE_COLUMNS, TrainingArtifacts, train_propensity_model


@dataclass
class RecommendationResult:
    customer_id: int
    offer_code: str
    offer_label: str
    score: float
    drivers: list[str]


@dataclass
class FeedbackEvent:
    customer_id: int
    offer_code: str
    responded: int
    channel: str
    recorded_at: str


class RecommendationService:
    def __init__(
        self,
        customers: pd.DataFrame,
        model_artifacts: TrainingArtifacts,
        training_frame: pd.DataFrame,
        feedback_path: Path | None = None,
    ):
        self.customers = customers
        self.model_artifacts = model_artifacts
        self.training_frame = training_frame.copy()
        self.offer_names = {offer.code: offer.label for offer in OFFERS}
        self.feedback_path = feedback_path
        self.feedback_events: list[FeedbackEvent] = []

    @classmethod
    def bootstrap(
        cls,
        customer_count: int = 250,
        seed: int = 7,
        feedback_path: Path | None = None,
    ) -> "RecommendationService":
        customers = generate_customers(count=customer_count, seed=seed)
        training_frame = build_training_frame(customers, seed=seed + 1)
        artifacts = train_propensity_model(training_frame)
        return cls(
            customers=customers,
            model_artifacts=artifacts,
            training_frame=training_frame,
            feedback_path=feedback_path,
        )

    def recommend(self, customer_id: int, top_n: int = 3) -> list[RecommendationResult]:
        customer_row = self.customers.loc[self.customers["customer_id"] == customer_id]
        if customer_row.empty:
            raise KeyError(f"Customer {customer_id} was not found")

        customer = customer_row.iloc[0]
        eligible_codes = eligibility_for_customer(customer)
        if not eligible_codes:
            return []

        candidates = []
        for offer_code in eligible_codes:
            candidate = customer.to_dict()
            candidate["offer_code"] = offer_code
            candidates.append(candidate)

        candidate_frame = pd.DataFrame(candidates)[FEATURE_COLUMNS]
        probabilities = self.model_artifacts.pipeline.predict_proba(candidate_frame)[:, 1]

        ranked = sorted(
            (
                RecommendationResult(
                    customer_id=customer_id,
                    offer_code=offer_code,
                    offer_label=self.offer_names[offer_code],
                    score=float(score),
                    drivers=self._build_drivers(customer, offer_code),
                )
                for offer_code, score in zip(eligible_codes, probabilities)
            ),
            key=lambda item: item.score,
            reverse=True,
        )
        return ranked[:top_n]

    def record_feedback(
        self,
        customer_id: int,
        offer_code: str,
        responded: bool,
        channel: str = "api",
    ) -> FeedbackEvent:
        customer = self._get_customer(customer_id)
        if offer_code not in self.offer_names:
            raise KeyError(f"Offer {offer_code} was not found")
        if offer_code not in eligibility_for_customer(customer):
            raise ValueError(f"Offer {offer_code} is not eligible for customer {customer_id}")

        event = FeedbackEvent(
            customer_id=customer_id,
            offer_code=offer_code,
            responded=int(responded),
            channel=channel,
            recorded_at=datetime.now(UTC).replace(microsecond=0).isoformat(),
        )
        self.feedback_events.append(event)
        self._persist_feedback_event(event)
        return event

    def retrain_from_feedback(self) -> dict[str, int | str]:
        if not self.feedback_events:
            return {"status": "skipped", "feedback_events": 0}

        feedback_rows = [self._feedback_row(event) for event in self.feedback_events]
        feedback_frame = pd.DataFrame(feedback_rows)
        retraining_frame = pd.concat([self.training_frame, feedback_frame], ignore_index=True)
        self.model_artifacts = train_propensity_model(retraining_frame)
        self.training_frame = retraining_frame
        return {"status": "retrained", "feedback_events": len(feedback_frame)}

    def _get_customer(self, customer_id: int) -> pd.Series:
        customer_row = self.customers.loc[self.customers["customer_id"] == customer_id]
        if customer_row.empty:
            raise KeyError(f"Customer {customer_id} was not found")
        return customer_row.iloc[0]

    def _persist_feedback_event(self, event: FeedbackEvent) -> None:
        if self.feedback_path is None:
            return

        self.feedback_path.parent.mkdir(parents=True, exist_ok=True)
        with self.feedback_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(asdict(event)) + "\n")

    def _feedback_row(self, event: FeedbackEvent) -> dict:
        customer = self._get_customer(event.customer_id).to_dict()
        customer["offer_code"] = event.offer_code
        customer["eligible"] = 1
        customer["responded"] = event.responded
        customer["observation_date"] = event.recorded_at[:10]
        return customer

    def _build_drivers(self, customer: pd.Series, offer_code: str) -> list[str]:
        offer = next(offer for offer in OFFERS if offer.code == offer_code)
        drivers = self._offer_drivers(customer, offer)
        drivers.append(self._engagement_driver(customer))
        drivers.append(self._renewal_driver(customer))
        unique_drivers = list(dict.fromkeys(driver for driver in drivers if driver))
        return unique_drivers[:3]

    def _offer_drivers(self, customer: pd.Series, offer: OfferRule) -> list[str]:
        offer_code = offer.code
        if offer_code == "motor_addon":
            return [
                "Existing motor policy makes the add-on immediately attachable.",
                "Older insured vehicle increases interest in extra protection.",
                "Premium level supports add-on affordability.",
            ] if customer["has_motor_policy"] else []
        if offer_code == "family_health":
            return [
                "Dependents indicate family coverage needs.",
                "No current health policy leaves a protection gap.",
                "Agent-assisted journeys tend to convert well for family health decisions.",
            ]
        if offer_code == "home_insurance":
            return [
                "Homeownership makes property protection relevant.",
                "Regional profile aligns with home coverage targeting.",
            ]
        if offer_code == "travel_insurance":
            return [
                "Frequent travel pattern matches trip protection needs.",
                "Recent web activity suggests active trip planning or research.",
            ]
        if offer_code == "critical_illness":
            return [
                "Existing life coverage indicates protection-oriented behavior.",
                "Customer age falls in the target range for critical illness education.",
                "Tier profile supports rider upsell potential.",
            ]
        if offer_code == "wellness_app":
            return [
                "Existing health policy creates a natural wellness activation path.",
                "Digital engagement level suggests room for app-led servicing.",
                "Mobile or web preference supports activation through digital channels.",
            ]
        if offer_code == "renewal_outreach":
            return [
                "Upcoming renewal date needs retention outreach.",
                "Recent servicing or payment friction raises retention risk.",
                "Proactive contact can reduce lapse probability.",
            ]
        return []

    def _engagement_driver(self, customer: pd.Series) -> str:
        if float(customer["digital_engagement"]) >= 0.65:
            return "Strong digital engagement raises response likelihood through direct channels."
        return "Moderate digital engagement still supports a service-led or assisted outreach path."

    def _renewal_driver(self, customer: pd.Series) -> str:
        if int(customer["days_to_renewal"]) <= 45:
            return "Near-term renewal timing increases action urgency."
        return "Current policy timing leaves room for proactive cross-sell messaging."
