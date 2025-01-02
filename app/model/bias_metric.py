# app/model/classifier.py

from pydantic import BaseModel, HttpUrl
from typing import List

class BiasMetric(BaseModel):
    id: str
    protectedAttribute: str
    privilegedGroup: str
    unprivilegedGroup: str
    accuracyRatio: float
    description: str
    metrics: List[str]

class BiasMetricRequest(BaseModel):
    dataset: List[str]