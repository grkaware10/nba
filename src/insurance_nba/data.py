from __future__ import annotations

import math
import random
from pathlib import Path
from typing import Iterable

import pandas as pd

from .domain import OFFERS


def _sigmoid(value: float) -> float:
    return 1.0 / (1.0 + math.exp(-value))


def generate_customers(count: int = 250, seed: int = 7) -> pd.DataFrame:
    rng = random.Random(seed)
    anchor_date = pd.Timestamp("2026-09-01")
    rows = []
    for customer_id in range(1, count + 1):
        age = rng.randint(21, 68)
        tenure_years = rng.randint(0, 20)
        observation_date = anchor_date - pd.to_timedelta(rng.randint(0, 540), unit="D")
        monthly_income = round(rng.lognormvariate(10.45, 0.42), 2)
        annual_premium = round(max(12000, monthly_income * rng.uniform(0.08, 0.24)), 2)
        claims_last_24m = rng.randint(0, 4)
        claim_amount_24m = round(
            0.0 if claims_last_24m == 0 else rng.lognormvariate(9.1, 0.75) * claims_last_24m,
            2,
        )
        digital_engagement = round(rng.uniform(0.05, 0.95), 2)
        has_motor_policy = rng.random() < 0.55
        has_health_policy = rng.random() < 0.45
        has_life_policy = rng.random() < 0.35
        homeowner = rng.random() < 0.42
        frequent_traveler = rng.random() < 0.3
        dependents = rng.randint(0, 4)
        days_to_renewal = rng.randint(1, 365)
        last_campaign_response = rng.random() < 0.25
        web_sessions_30d = rng.randint(0, 32)
        call_center_contacts_90d = rng.randint(0, 8)
        payment_delay_days = rng.randint(0, 25)
        policy_count = rng.randint(1, 5)
        vehicle_age = rng.randint(0, 15) if has_motor_policy else 0
        credit_score = rng.randint(580, 840)
        region = rng.choice(["north", "south", "east", "west"])
        payment_method = rng.choice(["auto_debit", "credit_card", "upi", "net_banking"])
        channel_preference = rng.choice(["agent", "mobile", "web", "branch"])
        policy_tier = rng.choice(["silver", "gold", "platinum"])
        rows.append(
            {
                "customer_id": customer_id,
                "observation_date": observation_date.date().isoformat(),
                "age": age,
                "tenure_years": tenure_years,
                "annual_premium": annual_premium,
                "monthly_income": monthly_income,
                "claims_last_24m": claims_last_24m,
                "claim_amount_24m": claim_amount_24m,
                "digital_engagement": digital_engagement,
                "has_motor_policy": int(has_motor_policy),
                "has_health_policy": int(has_health_policy),
                "has_life_policy": int(has_life_policy),
                "homeowner": int(homeowner),
                "frequent_traveler": int(frequent_traveler),
                "dependents": dependents,
                "days_to_renewal": days_to_renewal,
                "last_campaign_response": int(last_campaign_response),
                "web_sessions_30d": web_sessions_30d,
                "call_center_contacts_90d": call_center_contacts_90d,
                "payment_delay_days": payment_delay_days,
                "policy_count": policy_count,
                "vehicle_age": vehicle_age,
                "credit_score": credit_score,
                "region": region,
                "payment_method": payment_method,
                "channel_preference": channel_preference,
                "policy_tier": policy_tier,
            }
        )
    customers = pd.DataFrame(rows)
    return inject_data_quality_issues(customers, seed=seed + 101)


def offer_codes() -> Iterable[str]:
    return [offer.code for offer in OFFERS]


def build_training_frame(customers: pd.DataFrame, seed: int = 11) -> pd.DataFrame:
    rng = random.Random(seed)
    rows = []
    for customer in customers.to_dict(orient="records"):
        for offer_code in offer_codes():
            eligible = int(_is_eligible(customer, offer_code))
            if not eligible:
                continue

            raw_score = (
                0.8 * customer["last_campaign_response"]
                + 1.2 * customer["digital_engagement"]
                - 0.18 * customer["claims_last_24m"]
                + 0.15 * customer["tenure_years"]
                - 0.025 * customer["payment_delay_days"]
                + 0.000015 * customer["monthly_income"]
                + 0.000004 * customer["annual_premium"]
                + _offer_bias(customer, offer_code)
            )
            probability = _sigmoid(raw_score - 1.0)
            responded = int(rng.random() < probability)

            row = dict(customer)
            row.update(
                {
                    "offer_code": offer_code,
                    "eligible": eligible,
                    "responded": responded,
                }
            )
            rows.append(row)
    return pd.DataFrame(rows)


def _offer_bias(customer: dict, offer_code: str) -> float:
    if offer_code == "motor_addon":
        return (
            0.9 * customer["has_motor_policy"]
            + 0.15 * customer["digital_engagement"]
            + 0.1 * (customer["vehicle_age"] > 6)
        )
    if offer_code == "family_health":
        return (
            0.5 * (customer["dependents"] > 0)
            + 0.35 * (1 - customer["has_health_policy"])
            + 0.2 * (customer["channel_preference"] == "agent")
        )
    if offer_code == "home_insurance":
        return 0.9 * customer["homeowner"] + 0.15 * (customer["region"] in {"north", "west"})
    if offer_code == "travel_insurance":
        return 1.1 * customer["frequent_traveler"] + 0.05 * customer["web_sessions_30d"]
    if offer_code == "critical_illness":
        return (
            0.7 * customer["has_life_policy"]
            + 0.2 * (customer["age"] > 35)
            + 0.15 * (customer["policy_tier"] == "gold")
        )
    if offer_code == "wellness_app":
        return (
            0.6 * customer["has_health_policy"]
            + 0.5 * (customer["digital_engagement"] < 0.6)
            + 0.2 * (customer["channel_preference"] == "mobile")
        )
    if offer_code == "renewal_outreach":
        return (
            0.8 * (customer["days_to_renewal"] < 45)
            + 0.2 * (customer["payment_delay_days"] > 10)
            + 0.1 * (customer["call_center_contacts_90d"] > 3)
        )
    return 0.0


def _is_eligible(customer: dict, offer_code: str) -> bool:
    if offer_code == "motor_addon":
        return bool(customer["has_motor_policy"])
    if offer_code == "family_health":
        return customer["dependents"] > 0 and not customer["has_health_policy"]
    if offer_code == "home_insurance":
        return bool(customer["homeowner"])
    if offer_code == "travel_insurance":
        return bool(customer["frequent_traveler"])
    if offer_code == "critical_illness":
        return bool(customer["has_life_policy"]) and 18 <= customer["age"] <= 60
    if offer_code == "wellness_app":
        return bool(customer["has_health_policy"])
    if offer_code == "renewal_outreach":
        return customer["days_to_renewal"] <= 60
    return False


def eligibility_for_customer(customer: pd.Series) -> list[str]:
    customer_dict = customer.to_dict()
    return [offer for offer in offer_codes() if _is_eligible(customer_dict, offer)]


def inject_data_quality_issues(customers: pd.DataFrame, seed: int = 108) -> pd.DataFrame:
    rng = random.Random(seed)
    frame = customers.copy()
    for column in ["annual_premium", "monthly_income", "claim_amount_24m", "web_sessions_30d"]:
        frame[column] = frame[column].astype(float)

    missing_plan = {
        "annual_premium": 0.06,
        "monthly_income": 0.08,
        "digital_engagement": 0.05,
        "vehicle_age": 0.07,
        "payment_method": 0.04,
        "channel_preference": 0.04,
    }
    for column, ratio in missing_plan.items():
        indices = frame.sample(frac=ratio, random_state=rng.randint(1, 1_000_000)).index
        frame.loc[indices, column] = pd.NA

    for column, multiplier_range in {
        "annual_premium": (3.5, 6.5),
        "monthly_income": (2.8, 4.8),
        "claim_amount_24m": (4.0, 8.0),
        "web_sessions_30d": (2.5, 4.0),
    }.items():
        indices = frame.sample(frac=0.015, random_state=rng.randint(1, 1_000_000)).index
        for index in indices:
            frame.at[index, column] = round(float(frame.at[index, column]) * rng.uniform(*multiplier_range), 2)

    return frame


def create_project_dataset(customer_count: int = 1200, seed: int = 17) -> pd.DataFrame:
    customers = generate_customers(count=customer_count, seed=seed)
    return build_training_frame(customers, seed=seed + 1)


def write_project_dataset(output_path: str | Path, customer_count: int = 1200, seed: int = 17) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dataset = create_project_dataset(customer_count=customer_count, seed=seed)
    dataset.to_csv(path, index=False)
    return path
