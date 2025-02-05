# app/service/analyse_service.py
from typing import List
from model.classifier import ClassifierName
from model.dataset import DatasetName
from service.utils.dataset_utils import initial_dataset_analysis
class AnalysisService:
    def __init__(self):
        pass

    def analyse(self, dataset_names: List[DatasetName], classifier_names: List[ClassifierName]) -> str:
        return initial_dataset_analysis(dataset_names, classifier_names)
    
    