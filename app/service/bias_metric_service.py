# app/service/item_service.py
from db.firebaseConfig import FirebaseConfig
from model.bias_metric import BiasMetric, BiasMetricRequest
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
                id = data.get('id'),
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

    def fetch_bias_metrics(self, dataset: List[BiasMetricRequest]) -> List[BiasMetric]:
        bias_metrics_ref = self.db.collection('bias_metrics')
        bias_metrics = []
        
        for dataset_id in dataset:
            doc_ref = bias_metrics_ref.document(dataset_id)
            doc = doc_ref.get()
            
            if doc.exists:
                data = doc.to_dict()
                bias_metric = BiasMetric(
                    id=doc.id,
                    protectedAttribute=data.get('protectedAttribute'),
                    privilegedGroup=data.get('privilegedGroup'),
                    unprivilegedGroup=data.get('unprivilegedGroup'),
                    accuracyRatio=data.get('accuracyRatio'),
                    description=data.get('description'),
                    metrics=data.get('metrics', [])
                )
                bias_metrics.append(bias_metric)
            else:
                print(f"Bias Metric with dataset ID {dataset_id} not found.")

        return bias_metrics
        
    def add_bias_metric(self, protectedAttribute: str, privilegedGroup: str, unprivilegedGroup: str,
                        accuracyRatio: float, description: str, metrics: List[str]) -> BiasMetric:
        bias_metrics_ref = self.db.collection('bias_metrics')

        result = bias_metrics_ref.add({
            'protectedAttribute': protectedAttribute,
            'privilegedGroup': privilegedGroup,
            'unprivilegedGroup': unprivilegedGroup,
            'accuracyRatio': accuracyRatio,
            'description': description,
            'metrics': metrics
        })

        doc_ref = result[1]
        bias_metric_id = doc_ref.id

        doc_ref.update({
            'id': bias_metric_id
        })

        new_bias_metric = BiasMetric(
            id=bias_metric_id,
            protectedAttribute=protectedAttribute,
            privilegedGroup=privilegedGroup,
            unprivilegedGroup=unprivilegedGroup,
            accuracyRatio=accuracyRatio,
            description=description,
            metrics=metrics
        )
        
        return new_bias_metric
