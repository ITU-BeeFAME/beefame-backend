# app/service/item_service.py
from db.firebaseConfig import FirebaseConfig
from model.bias_metric import BiasMetric, BiasMetricRequest
from model.classifier import ClassifierInfo
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
        result = classifiers_ref.add({
            'name': name,
            'url': url
        })

        doc_ref = result[1]
        classifier_id = doc_ref.id

        doc_ref.update({
            'id': classifier_id
        })

        new_classifier = ClassifierInfo(
            id=classifier_id,
            name=name,
            url=url,
        )

        return new_classifier