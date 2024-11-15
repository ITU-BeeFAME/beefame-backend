# app/service/item_service.py
from app.model.bias_metric import BiasMetric, BiasMetricRequest
from app.model.classifier import ClassifierInfo
from sqlalchemy.orm import Session
from typing import List, Optional


class ClassifierService:
    def __init__(self):
        self.classifiers = [
        ClassifierInfo(
            name="Support Vector Classification (SVC)",
            url="https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html"
        ),
        ClassifierInfo(
            name="Random Forest Classifier",
            url="https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html"
        ),
        ClassifierInfo(
            name="Logistic Regression",
            url="https://scikit-learn.org/1.5/modules/generated/sklearn.linear_model.LogisticRegression.html"
        ),
        ClassifierInfo(
            name="XGBClassifier",
            url="https://xgboost.readthedocs.io/en/stable/get_started.html"
        )
    ]

    def get_classifiers(self) -> List[ClassifierInfo]:
        return self.classifiers

