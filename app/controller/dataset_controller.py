# app/controller/item_controller.py
from app.model.response import SuccessResponse
from app.service.dataset_service import DatasetService
from fastapi import APIRouter, HTTPException, status
from typing import List
from app.model.dataset import DatasetInfo, DatasetSelectionRequest

router = APIRouter(
    prefix="/datasets",
    tags=["Datasets"],
)

@router.get("/", response_model=SuccessResponse)
def get_datasets():
    service = DatasetService()
    datasets = service.fetch_all_datasets()

    # example usage of adding a dataset to db
    # dataset_added = service.add_dataset("Statlog (German Credit Data)", "https://archive.ics.uci.edu/dataset/20/census+income", 1000, "description", {"Age": {"Unpreviliged": "Young", "Previliged": "Old"}})

    return SuccessResponse(data=datasets)

@router.get("/{dataset_id}", response_model=SuccessResponse)
def analyse_dataset(dataset_id: int):
    service = DatasetService()
    dataset_analysis = service.get_initial_dataset_analysis(dataset_id)

    return SuccessResponse(data=dataset_analysis)
