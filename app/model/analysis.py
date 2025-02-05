from typing import List
from pydantic import BaseModel

from model.classifier import ClassifierName
from model.dataset import DatasetName

class AnalyseRequest(BaseModel):
    dataset_names: List[DatasetName]
    classifier_names: List[ClassifierName]
