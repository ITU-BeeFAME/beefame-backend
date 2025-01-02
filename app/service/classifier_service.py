# app/service/item_service.py
from app.db.firebaseConfig import FirebaseConfig
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from app.model.classifier import ClassifierInfo
from sqlalchemy.orm import Session
from typing import List, Optional


class ClassifierService:
    def __init__(self):
        firebase_config = FirebaseConfig()
        self.db = firebase_config.get_db()
    
    def fetch_all_classifiers(self) -> List[ClassifierInfo]:
        classifiers_ref = self.db.collection('classifiers')
        docs = classifiers_ref.stream()

        classifiers = []
        for doc in docs:
            data = doc.to_dict()
            
            classifier = ClassifierInfo(
                id=data.get('id'),
                name=data.get('name'),
                url=data.get('url'),
            )
            classifiers.append(classifier)
        
        self.classifiers = classifiers
        return self.classifiers

    def add_classifier(self, name: str, url: str) -> ClassifierInfo:
        classifiers_ref = self.db.collection('classifiers')
        classifiers_ref.add({
            'name': name,
            'url': url
        })

        new_classifier = ClassifierInfo(
            name=name,
            url=url,
        )
        return new_classifier