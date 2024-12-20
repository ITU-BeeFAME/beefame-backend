# app/service/item_service.py
from app.db.firebaseConfig import FirebaseConfig
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from sqlalchemy.orm import Session
from typing import List, Optional

class BiasMetricService:
    def __init__(self):
        firebase_config = FirebaseConfig()
        self.db = firebase_config.get_db()

    def fetch_all_bias_metrics(self, dataset: List[BiasMetricRequest]) -> List[BiasMetric]:
        bias_metrics_ref = self.db.collection('bias_metrics')
        docs = bias_metrics_ref.stream()

        bias_metrics = []
        for doc in docs:
            data = doc.to_dict()
            
            bias_metric = BiasMetric(
                protectedAttribute=data.get('protectedAttribute'),
                privilegedGroup=data.get('privilegedGroup'),
                unprivilegedGroup=data.get('unprivilegedGroup'),
                accuracyRatio=data.get('accuracyRatio'),
                description=data.get('description'),
                metrics=data.get('metrics', [])
            )
            bias_metrics.append(bias_metric)
        
        self.bias_metrics = bias_metrics
        return bias_metrics

    def add_bias_metric(self, protectedAttribute: str, privilegedGroup: str, unprivilegedGroup: str,
                        accuracyRatio: float, description: str, metrics: List[str]) -> BiasMetric:
        bias_metrics_ref = self.db.collection('bias_metrics')
        bias_metrics_ref.add({
            'protectedAttribute': protectedAttribute,
            'privilegedGroup': privilegedGroup,
            'unprivilegedGroup': unprivilegedGroup,
            'accuracyRatio': accuracyRatio,
            'description': description,
            'metrics': metrics
        })

        new_bias_metric = BiasMetric(
            protectedAttribute=protectedAttribute,
            privilegedGroup=privilegedGroup,
            unprivilegedGroup=unprivilegedGroup,
            accuracyRatio=accuracyRatio,
            description=description,
            metrics=metrics
        )
        
        return new_bias_metric
