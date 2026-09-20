# Monitoring And Governance

## Deployment Guardrails
- Apply eligibility rules before scoring responses are shown to agent, web, app, or servicing channels.
- Keep logistic regression as a governance challenger against the selected production model.
- Review any schema change, rule change, or material score shift before promoting a retrained model.

## Monitoring KPIs
- Current benchmark leader: random_forest with ROC-AUC 0.9667 and PR-AUC 0.9818.
- Baseline synthetic response rate: 68.63%.
- Monitor Precision@K, Recall@K, ROC-AUC, PR-AUC, recommendation coverage, acceptance rate, renewal uplift, and digital engagement rate.
- Track calibration drift with reliability checks on the latest observation window before threshold changes.

## Data Quality Checks
- monthly_income: 7.84% missing in the prototype dataset.
- annual_premium: 5.88% missing in the prototype dataset.
- vehicle_age: 5.88% missing in the prototype dataset.
- digital_engagement: 3.92% missing in the prototype dataset.
- channel_preference: 1.96% missing in the prototype dataset.
- Alert on spikes in missing payment method, channel, premium, income, or vehicle-age fields because they affect ranking stability.

## Governance Controls
- Preserve audit logs for recommendation requests, top-N outputs, explanations, and feedback events.
- Restrict retraining inputs to consented and policy-compliant customer outcomes.
- Validate that underwriting or compliance exclusions remain enforced ahead of recommendation delivery.
- Review cold-start fallback logic and coverage concentration by offer to avoid over-serving popular products.

## Retraining Policy
- Retrain when enough new feedback accumulates or when monitored KPI degradation breaches thresholds.
- Use the newest observation window as the holdout slice to preserve time order during validation.
- Re-run calibration checks after every retraining cycle before exposing new scores to channels.