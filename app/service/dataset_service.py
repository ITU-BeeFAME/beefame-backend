# app/service/item_service.py
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from app.model.dataset import DatasetInfo
from sqlalchemy.orm import Session
from typing import List, Optional


class DatasetService:
    def __init__(self):
        self.datasets = [
        DatasetInfo(
            name="Statlog (German Credit Data)",
            url="https://archive.ics.uci.edu/dataset/144/statlog+german+credit+data",
            instances=1000,
            description="This dataset classifies people described by a set of attributes as good or bad credit risks. Comes in two formats (one all numeric). Also comes with a cost matrix."
        ),
        DatasetInfo(
            name="Census Income",
            url="https://archive.ics.uci.edu/dataset/20/census+income",
            instances=48842,
            description="Predict whether income exceeds $50K/yr based on census data. Also known as Adult dataset."
        )
    ]

    def get_datasets(self) -> List[DatasetInfo]:
        return self.datasets

