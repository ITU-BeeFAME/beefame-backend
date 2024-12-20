# app/service/dataset_service.py
from app.db.firebaseConfig import FirebaseConfig
from app.model.dataset import DatasetAnalysis, DatasetInfo
from app.service.utils.dataset_utils import initial_dataset_analysis
from typing import List

class DatasetService:
    def __init__(self):
        firebase_config = FirebaseConfig()
        self.db = firebase_config.get_db()

    def fetch_all_datasets(self) -> List[DatasetInfo]:
        dataset_info_ref = self.db.collection('dataset_info')
        docs = dataset_info_ref.stream()

        dataset_info = []
        for doc in docs:
            data = doc.to_dict()
            
            dataset = DatasetInfo(
                id=data.get('id'),
                name=data.get('name'),
                url=data.get('url'),
                instances=data.get('instances'),
                description=data.get('description'),
                sensitive_features=data.get('sensitive_features', [])
            )
            dataset_info.append(dataset)
        
        self.datasets = dataset_info
        return self.datasets
    
    def get_initial_dataset_analysis(self, dataset_id) -> List[DatasetAnalysis]:
        return initial_dataset_analysis(dataset_id)

    def add_dataset(self, name: str, url: str, instances: int, description: str, sensitive_features: dict) -> DatasetInfo:
        dataset_info_ref = self.db.collection('dataset_info')

        # may be better if we exclude id inside individual dataset document since auto generated id's are created by firestore
        dataset_info_ref.add({
            'name': name,
            'url': url,
            'instances': instances,
            'description': description,
            'sensitive_features': sensitive_features
        })

        
        new_dataset = DatasetInfo(
            name=name,
            url=url,
            instances=instances,
            description=description,
            sensitive_features=sensitive_features
        )

        return new_dataset
