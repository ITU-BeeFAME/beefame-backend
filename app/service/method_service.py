# app/service/item_service.py
from app.db.firebaseConfig import FirebaseConfig
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from app.model.method import MethodInfo
from sqlalchemy.orm import Session
from typing import List, Optional
from pydantic import BaseModel, HttpUrl

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
                id=data.get('id'),
                name=data.get('name'),
                url=data.get('url'),
                description=data.get('description'),
                type=data.get('type')
            )
            methods.append(method)
        
        self.methods = methods
        return self.methods
    
    def add_method(self, name: str, url: HttpUrl, description: str, type: str) -> MethodInfo:
        methods_ref = self.db.collection('methods')
        methods_ref.add({
            'name': name,
            'url': url,
            'description': description,
            'type': type
        })

        new_method = MethodInfo(
            name= name,
            url= url,
            description= description,
            type= type
        )
        
        return new_method