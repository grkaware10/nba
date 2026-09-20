from __future__ import annotations

from functools import lru_cache

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from .recommender import FeedbackEvent, RecommendationService


class RecommendationResponse(BaseModel):
    customer_id: int
    offer_code: str
    offer_label: str
    score: float
    drivers: list[str]


class FeedbackRequest(BaseModel):
    offer_code: str
    responded: bool
    channel: str = "api"


class FeedbackResponse(BaseModel):
    customer_id: int
    offer_code: str
    responded: int
    channel: str
    recorded_at: str


class RetrainResponse(BaseModel):
    status: str
    feedback_events: int


@lru_cache
def get_service() -> RecommendationService:
    return RecommendationService.bootstrap(feedback_path=None)


app = FastAPI(title="Insurance Next Best Action API", version="0.1.0")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/metrics/model")
def model_metrics() -> dict[str, float | str]:
    service = get_service()
    return {
        "model_name": service.model_artifacts.model_name,
        "imputation_strategy": service.model_artifacts.imputation_strategy,
        "calibrated": str(service.model_artifacts.calibrated).lower(),
        "calibration_method": service.model_artifacts.calibration_method or "none",
        "roc_auc": round(service.model_artifacts.auc, 4),
        "accuracy": round(service.model_artifacts.accuracy, 4),
        "pr_auc": round(service.model_artifacts.pr_auc, 4),
    }


@app.get("/recommendations/{customer_id}", response_model=list[RecommendationResponse])
def recommendations(customer_id: int, top_n: int = 3) -> list[RecommendationResponse]:
    service = get_service()
    try:
        results = service.recommend(customer_id=customer_id, top_n=top_n)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return [RecommendationResponse(**result.__dict__) for result in results]


@app.post("/feedback/{customer_id}", response_model=FeedbackResponse)
def capture_feedback(customer_id: int, request: FeedbackRequest) -> FeedbackResponse:
    service = get_service()
    try:
        event = service.record_feedback(
            customer_id=customer_id,
            offer_code=request.offer_code,
            responded=request.responded,
            channel=request.channel,
        )
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    return FeedbackResponse(**event.__dict__)


@app.post("/retrain", response_model=RetrainResponse)
def retrain_model() -> RetrainResponse:
    service = get_service()
    outcome = service.retrain_from_feedback()
    return RetrainResponse(**outcome)
