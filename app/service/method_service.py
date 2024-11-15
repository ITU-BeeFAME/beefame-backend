# app/service/item_service.py
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from app.model.method import MethodInfo
from sqlalchemy.orm import Session
from typing import List, Optional


class MethodService:
    def __init__(self):
        self.methods = [
        MethodInfo(
            name="Data Repaierer",
            type="Preprocessing",
            url="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/data_repairer.py",
            description="Transforms the data distribution so that a given feature distribution is marginally independent of the sensitive attribute, s."
        ),
        MethodInfo(
            name="Prevalence Sampling",
            type="Preprocessing",
            url="https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/prevalence_sample.py",
            description="Predict whether income exceeds $50K/yr based on census data. Also known as Adult dataset."
        ),
        MethodInfo(
            name="Relabeller",
            type="Preprocessing",
            url="https://github.com/cosmicBboy/themis-ml/blob/master/themis_ml/preprocessing/relabelling.py",
            description="Relabels target variables using a function that can compute a decision boundary in input data space using heuristic."
        )
    ]

    def get_methods(self) -> List[MethodInfo]:
        return self.methods

