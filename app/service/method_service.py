# app/service/item_service.py
from app.db.firebaseConfig import FirebaseConfig
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from app.model.method import MethodInfo
from sqlalchemy.orm import Session
from typing import List, Optional


class MethodService:
    def __init__(self):
        firebase_config = FirebaseConfig()
        self.db = firebase_config.get_db()

    def fetch_all_methods(self) -> List[MethodInfo]:
        methods_ref = self.db.collection('methods')
        docs = methods_ref.stream()

        methods = []
        for doc in docs:
            data = doc.to_dict()
            
            method = MethodInfo(
                name=data.get('name'),
                url=data.get('url'),
            )
            methods.append(method)
        
        self.methods = methods
        return self.methods
    
    def add_method(self, name: str, url: str) -> MethodInfo:
        methods_ref = self.db.collection('methods')
        methods_ref.add({
            'name': name,
            'url': url
        })

        new_method = MethodInfo(
            name=name,
            url=url,
        )
        
        return new_method