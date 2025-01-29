from pydantic import BaseModel

from model.classifier import ClassifierName
from model.dataset import DatasetName

class AnalyseRequest(BaseModel):
    dataset_name: DatasetName
    classifier_name: ClassifierName
