# app/controller/item_controller.py
from model.response import SuccessResponse
from service.classifier_service import ClassifierService
from fastapi import APIRouter, HTTPException, status
from typing import List
from model.classifier import ClassifierInfo

router = APIRouter(
    prefix="/classifiers",
    tags=["Classifiers"],
)

@router.get("/all", response_model=SuccessResponse)
def get_classifiers():
    service = ClassifierService()
    classifiers = service.fetch_all_classifiers()

    # example usage of adding a classifier to db
    # classifier_added = service.add_classifier("Support Vector Classification (SVC)", "https://scikit-learn.org/dev/modules/generated/sklearn.svm.SVC.html")
    
    return SuccessResponse(data=classifiers)