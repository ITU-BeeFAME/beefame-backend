# app/model/evaluation.py
from pydantic import BaseModel
from typing import Dict, Any

from model.classifier import ClassifierName
from model.dataset import DatasetName
from model.method import MethodName

class ClassificationReport(BaseModel):
    precision: Dict[str, float]
    recall: Dict[str, float]
    f1_score: Dict[str, float]
    support: Dict[str, float]

class EvaluationResult(BaseModel):
    name: str
    accuracy: float
    balanced_accuracy: float
    auc_roc: float
    classification_report: ClassificationReport
    disparate_impact: float
    statistical_parity_difference: float
    equal_opportunity_difference: float
    average_odds_difference: float
    theil_index: float

class EvaluationRequest(BaseModel):
    dataset_name: DatasetName
    classifier_name: ClassifierName
    method_name: MethodName