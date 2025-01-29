# app/service/analyse_service.py
from model.classifier import ClassifierName
from db.firebaseConfig import FirebaseConfig
from model.dataset import DatasetName
from service.utils.dataset_utils import initial_dataset_analysis
class AnalysisService:
    def __init__(self):
        firebase_config = FirebaseConfig()
        self.db = firebase_config.get_db()

    def analyse(self, dataset_name: DatasetName, classifier_name: ClassifierName) -> str:
        return initial_dataset_analysis(dataset_name.value, classifier_name.value)
    
    