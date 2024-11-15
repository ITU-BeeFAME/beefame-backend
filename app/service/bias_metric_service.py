# app/service/item_service.py
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from sqlalchemy.orm import Session
from typing import List, Optional


class BiasMetricService:
    def __init__(self):
        self.bias_metrics = [
            BiasMetric(
                protectedAttribute="Sex",
                privilegedGroup="Male",
                unprivilegedGroup="Female",
                accuracyRatio=75,
                description="With default thresholds, bias against unprivileged group detected in 0 out of 5 metrics",
                metrics=["Statistical Parity Difference", "Equal Opportunity Difference", "Average Odds Difference", "Disparate Impact", "Theil Index"],
            ), 
            BiasMetric(
                protectedAttribute="Age",
                privilegedGroup="Old",
                unprivilegedGroup="Young",
                accuracyRatio=75,
                description="With default thresholds, bias against unprivileged group detected in 4 out of 5 metrics",
                metrics=["Statistical Parity Difference", "Equal Opportunity Difference", "Average Odds Difference", "Disparate Impact", "Theil Index"],
            ),  
        ]

    def get_bias_metrics(self, dataset: List[BiasMetricRequest]) -> List[BiasMetric]:
        # TODO : business logic based on dataset
        return self.bias_metrics

