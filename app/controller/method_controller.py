# app/controller/item_controller.py
from model.response import SuccessResponse
from service.method_service import MethodService
from fastapi import APIRouter, HTTPException, status
from typing import List
from model.method import MethodInfo

router = APIRouter(
    prefix="/methods",
    tags=["Methods"],
)

@router.get("/", response_model=SuccessResponse)
def get_methods():
    service = MethodService()
    methods = service.fetch_all_methods()

    # example usage of adding a method to db
    # method_added = service.add_method("Prevalence Sampling", "https://github.com/dssg/aequitas/blob/master/src/aequitas/flow/methods/preprocessing/prevalence_sample.py", "Predict whether income exceeds $50K/yr based on census data. Also known as Adult dataset.", "Preprocessing")

    return SuccessResponse(data=methods)