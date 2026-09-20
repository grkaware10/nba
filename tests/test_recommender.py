import json

from insurance_nba.recommender import RecommendationService


def test_recommendations_are_sorted_descending() -> None:
    service = RecommendationService.bootstrap(customer_count=80, seed=12)
    results = service.recommend(customer_id=1, top_n=3)
    assert len(results) <= 3
    assert results == sorted(results, key=lambda item: item.score, reverse=True)
    assert all(result.drivers for result in results)


def test_unknown_customer_raises_key_error() -> None:
    service = RecommendationService.bootstrap(customer_count=20, seed=4)
    try:
        service.recommend(customer_id=9999)
    except KeyError:
        pass
    else:
        raise AssertionError("Expected a KeyError for an unknown customer")


def test_recommendations_are_eligible_for_customer() -> None:
    service = RecommendationService.bootstrap(customer_count=120, seed=9)
    results = service.recommend(customer_id=10, top_n=5)

    for result in results:
        if result.offer_code == "motor_addon":
            customer = service.customers.loc[service.customers["customer_id"] == result.customer_id].iloc[0]
            assert customer["has_motor_policy"] == 1


def test_model_metrics_include_selected_model() -> None:
    service = RecommendationService.bootstrap(customer_count=100, seed=5)
    assert service.model_artifacts.model_name == "random_forest"
    assert service.model_artifacts.imputation_strategy == "mode"
    assert service.model_artifacts.calibrated is True
    assert service.model_artifacts.calibration_method == "sigmoid"
    assert 0.5 <= service.model_artifacts.auc <= 1.0


def test_feedback_is_persisted_and_retrain_runs(tmp_path) -> None:
    feedback_path = tmp_path / "feedback_events.jsonl"
    service = RecommendationService.bootstrap(customer_count=90, seed=14, feedback_path=feedback_path)
    recommendation = service.recommend(customer_id=1, top_n=1)[0]

    event = service.record_feedback(
        customer_id=recommendation.customer_id,
        offer_code=recommendation.offer_code,
        responded=True,
        channel="agent_portal",
    )
    outcome = service.retrain_from_feedback()

    persisted = [json.loads(line) for line in feedback_path.read_text(encoding="utf-8").splitlines()]
    assert event.offer_code == recommendation.offer_code
    assert outcome == {"status": "retrained", "feedback_events": 1}
    assert persisted[0]["channel"] == "agent_portal"
    assert service.training_frame.iloc[-1]["offer_code"] == recommendation.offer_code