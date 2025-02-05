# app/service/item_service.py
from repository.classifier_repository import ClassifierRepository
from model.classifier import ClassifierInfo
from typing import List

class ClassifierService:
    def __init__(self):
        pass
    
    def get_classifiers(self) -> List[ClassifierInfo]:
        return ClassifierRepository.get_classifiers()
    
    
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