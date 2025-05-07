from typing import List, Dict, Any
from pydantic import BaseModel
from model.dataset import DatasetName

class ClassifierRequest(BaseModel):
    name: str
    params: Dict[str, Any] = {}

class AnalyseRequest(BaseModel):
    dataset_names: List[DatasetName]
    classifiers: List[ClassifierRequest]
