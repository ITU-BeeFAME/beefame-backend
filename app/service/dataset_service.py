# app/service/dataset_service.py
from app.model.dataset import DatasetAnalysis, DatasetInfo
from app.service.utils.dataset_utils import initial_dataset_analysis
from typing import List

class DatasetService:
    def __init__(self):
        self.datasets = [
        DatasetInfo(
            id = 1,
            name="Statlog (German Credit Data)",
            url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            instances=1000,
            description="This dataset classifies people described by a set of attributes as good or bad credit risks. Comes in two formats (one all numeric). Also comes with a cost matrix.",
            sensitive_features = {"Gender": {"privilaged" : "Male", "Unprivilaged" : "Female"},
                                  "Age": {"privilaged" : "Old", "Unprivilaged" : "Young"}}
        ),
        DatasetInfo(
            id = 2,
            name="Census Income",
            url="https://archive.ics.uci.edu/dataset/20/census+income",
            instances=48842,
            description="Predict whether income exceeds $50K/yr based on census data. Also known as Adult dataset.",
            sensitive_features = {"Gender": {"privilaged" : "Male", "Unprivilaged" : "Female"},
                                  "Race": {"privilaged" : "White", "Unprivilaged" : "Non-white"},
                                  "Age": {"privilaged" : "Old", "Unprivilaged" : "Young"}}
        )
    ]

    def get_datasets(self) -> List[DatasetInfo]:
        return self.datasets
    
    def get_initial_dataset_analysis(self, dataset_id) -> List[DatasetAnalysis]:
        return initial_dataset_analysis(dataset_id)

