# app/service/dataset_service.py
from db.firebaseConfig import FirebaseConfig
from model.dataset import DatasetAnalysis, DatasetInfo, SensitiveFeatures
from service.utils.dataset_utils import initial_dataset_analysis
from typing import List

class DatasetService:
    def __init__(self):
        firebase_config = FirebaseConfig()
        self.db = firebase_config.get_db()

    def fetch_all_datasets(self) -> List[DatasetInfo]:
        datasets_ref = self.db.collection('datasets')
        docs = datasets_ref.stream()
        datasets = []
        for doc in docs:
            data = doc.to_dict()
            sensitive_features_data = data.get('sensitive_features', [])
            sensitive_features = [
                SensitiveFeatures(**feature) for feature in sensitive_features_data
            ]

            # DatasetInfo nesnesini oluştur
            dataset = DatasetInfo(
                id=doc.id,
                name=data.get('name', ""),
                slug=data.get('slug', ""),
                url=data.get('url', ""),
                instances=data.get('instances', 0),
                description=data.get('description', "No description provided."),
                sensitive_features=sensitive_features
            )
            datasets.append(dataset)
        
        
        return datasets
    
    def get_initial_dataset_analysis(self, dataset_id) -> List[DatasetAnalysis]:
        return initial_dataset_analysis(dataset_id)

    def add_dataset(self, name: str, url: str, instances: int, description: str, sensitive_features: dict) -> DatasetInfo:
        dataset_info_ref = self.db.collection('dataset_info')

        result = dataset_info_ref.add({
            'name': name,
            'url': url,
            'instances': instances,
            'description': description,
            'sensitive_features': sensitive_features
        })

        doc_ref = result[1]
        dataset_id = doc_ref.id
        
        doc_ref.update({
            'id': dataset_id
        })
        
        new_dataset = DatasetInfo(
            id=dataset_id,
            name=name,
            url=url,
            instances=instances,
            description=description,
            sensitive_features=sensitive_features
        )

        return new_dataset
