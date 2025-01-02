# app/model/dataset.py

from pydantic import BaseModel, HttpUrl
from typing import List

class DatasetInfo(BaseModel):
    id: str
    name: str
    url: HttpUrl
    instances: int
    description: str
    sensitive_features: dict

class DatasetSelectionRequest(BaseModel):
    names: List[str]

class DatasetAnalysis(BaseModel):
    sensitive_column : str
    model_accuracy : float
    statistical_parity_difference : float
    equal_opportunity_difference : float
    average_odds_difference : float
    disparate_impact : float
    theil_index : float