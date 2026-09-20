# Data Dictionary

This dataset is synthetic and structured at the customer-plus-offer-plus-observation-date grain.

| Column | Type | Domain | Missing % | Description |
| --- | --- | --- | ---: | --- |
| customer_id | int64 | identity | 0.00 | Synthetic customer identifier. |
| observation_date | object | governance | 0.00 | Observation date for the customer-offer training instance used in time-aware validation. |
| age | int64 | profile | 0.00 | Customer age in years. |
| tenure_years | int64 | profile | 0.00 | Years since the customer joined the insurer. |
| annual_premium | float64 | portfolio | 5.88 | Annualized premium paid across active policies. |
| monthly_income | float64 | affordability | 7.84 | Estimated monthly income for affordability features. |
| claims_last_24m | int64 | claims | 0.00 | Claim count in the last 24 months. |
| claim_amount_24m | float64 | claims | 0.00 | Total claim amount in the last 24 months. |
| digital_engagement | float64 | engagement | 3.92 | Synthetic engagement score between 0 and 1. |
| has_motor_policy | int64 | portfolio | 0.00 | Flag indicating an active motor policy. |
| has_health_policy | int64 | portfolio | 0.00 | Flag indicating an active health policy. |
| has_life_policy | int64 | portfolio | 0.00 | Flag indicating an active life policy. |
| homeowner | int64 | profile | 0.00 | Flag indicating whether the customer owns a home. |
| frequent_traveler | int64 | lifestyle | 0.00 | Flag indicating frequent travel behavior. |
| dependents | int64 | profile | 0.00 | Number of declared dependents. |
| days_to_renewal | int64 | retention | 0.00 | Days until the nearest renewal event. |
| last_campaign_response | int64 | marketing | 0.00 | Flag indicating response to the prior campaign. |
| web_sessions_30d | float64 | engagement | 0.00 | Web sessions observed in the last 30 days. |
| call_center_contacts_90d | int64 | servicing | 0.00 | Call center contacts observed in the last 90 days. |
| payment_delay_days | int64 | billing | 0.00 | Days of payment delay across recent billing cycles. |
| policy_count | int64 | portfolio | 0.00 | Count of active policies held by the customer. |
| vehicle_age | float64 | portfolio | 5.88 | Insured vehicle age in years when a motor policy exists. |
| credit_score | int64 | risk | 0.00 | Synthetic credit score proxy. |
| region | object | profile | 0.00 | Customer region segment. |
| payment_method | object | billing | 1.96 | Preferred payment method. |
| channel_preference | object | engagement | 1.96 | Preferred interaction channel. |
| policy_tier | object | portfolio | 0.00 | Portfolio tier label. |
| offer_code | object | offer | 0.00 | Candidate offer or action code evaluated for the customer. |
| eligible | int64 | governance | 0.00 | Eligibility flag after business-rule filtering. |
| responded | int64 | target | 0.00 | Target flag showing whether the customer accepted or responded. |

## Notes
- Missing values are injected intentionally to benchmark imputation strategies.
- `eligible` is rule-derived and only eligible customer-offer rows are retained in the training frame.
- `responded` is a synthetic propensity outcome and should be replaced with observed production outcomes when available.