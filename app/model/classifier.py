# app/model/classifier.py

from pydantic import BaseModel, HttpUrl
from enum import Enum


class ClassifierInfo(BaseModel):
    id: str
    name: str
    url: HttpUrl


class ClassifierName(Enum):
    XGB = "XGBClassifier"
    SVC = "Support Vector Classification (SVC)"
    RFC = "Random Forest Classifier"
    LR = "Logistic Regression"