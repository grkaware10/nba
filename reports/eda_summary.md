# Insurance NBA EDA Summary

## Dataset Overview
- Rows: 51
- Columns: 30
- Response rate: 68.63%
- Observation window: 2025-04-05 to 2026-08-30

## Missingness
- monthly_income: 7.84%
- annual_premium: 5.88%
- vehicle_age: 5.88%
- digital_engagement: 3.92%
- channel_preference: 1.96%
- payment_method: 1.96%
- age: 0.00%
- claims_last_24m: 0.00%
- claim_amount_24m: 0.00%
- has_motor_policy: 0.00%

## Highest Absolute Skewness
- annual_premium: 4.469
- last_campaign_response: 1.286
- homeowner: 1.159
- claim_amount_24m: 1.055
- responded: -0.827
- has_life_policy: -0.634
- web_sessions_30d: 0.616
- frequent_traveler: 0.543
- has_health_policy: 0.543
- vehicle_age: 0.470

## Offer Response Rate
- travel_insurance: 83.33%
- critical_illness: 80.00%
- wellness_app: 71.43%
- renewal_outreach: 66.67%
- family_health: 60.00%
- home_insurance: 60.00%
- motor_addon: 60.00%

## Treatment Decisions
- Missing values are intentionally present to benchmark mean, median, mode, and KNN imputers.
- Heavy-tailed numeric features are capped with IQR bounds before modeling to reduce extreme leverage without dropping rows.
- Residual skewness is handled with Yeo-Johnson because it supports zero and near-zero values, unlike Box-Cox.
- Feature engineering adds claim severity, premium-income ratio, service engagement ratio, policy density, and renewal urgency.
- Evaluation uses a time-aware holdout ordered by observation_date so later customer-offer events simulate production scoring.
- Propensity outputs are calibrated with Platt scaling via sigmoid calibration before ranking recommendations.