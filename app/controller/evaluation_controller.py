# app/controller/evaluation_controller.py
from model.evaluation import EvaluationRequest
from model.response import SuccessResponse
from service.evaluation_service import EvaluationService
from fastapi import APIRouter, HTTPException, status
from typing import List

router = APIRouter(
    prefix="/evaluation",
    tags=["Evaluation"],
)

@router.post("/", response_model=SuccessResponse)
def get_evaluation(request: EvaluationRequest):
    service = EvaluationService()

    """ evaluation_results = service.get_evaluation()
    print(evaluation_results) """

    evaluation_results_german = [
        {
            "Sensitive Column": "Gender",
            "Model Accuracy": 0.91,
            "Statistical Parity Difference": 0.4,
            "Equal Opportunity Difference": 0.35,
            "Average Odds Difference": 0.32,
            "Disparate Impact": 0.65,
            "Theil Index": -0.15,
        },
        {
            "Sensitive Column": "Age",
            "Model Accuracy": 0.91,
            "Statistical Parity Difference": 0.2,
            "Equal Opportunity Difference": 0.18,
            "Average Odds Difference": 0.1,
            "Disparate Impact": 0.8,
            "Theil Index": -0.12,
        },
    ]

    evaluation_results_adult = [
        {
            "Sensitive Column": "Gender",
            "Model Accuracy": 0.91,
            "Statistical Parity Difference": 0.4,
            "Equal Opportunity Difference": 0.35,
            "Average Odds Difference": 0.32,
            "Disparate Impact": 0.65,
            "Theil Index": -0.15,
        },
        {
            "Sensitive Column": "Age",
            "Model Accuracy": 0.91,
            "Statistical Parity Difference": 0.2,
            "Equal Opportunity Difference": 0.18,
            "Average Odds Difference": 0.1,
            "Disparate Impact": 0.8,
            "Theil Index": -0.12,
        },
        {
            "Sensitive Column": "Race",
            "Model Accuracy": 0.91,
            "Statistical Parity Difference": 0.2,
            "Equal Opportunity Difference": 0.18,
            "Average Odds Difference": 0.1,
            "Disparate Impact": 0.8,
            "Theil Index": -0.12,
        },
    ]

    if request.dataset_name.value == "german": 
        SuccessResponse(data=evaluation_results_german)
    elif request.dataset_name.value == "adult":
        return SuccessResponse(data=evaluation_results_adult)
    
    return SuccessResponse(data=evaluation_results_german)

    

    