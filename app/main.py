# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.controller import dataset_controller, method_controller, classifier_controller, bias_metrics_controller, evaluation_controller

app = FastAPI(
    title="BeeFair REST API",
    description="BeeFair Fair Testing Tool REST API",
    version="1.0.0",
)

# CORS related configuration 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"], # Frontend url:port
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

app.include_router(dataset_controller.router)
app.include_router(method_controller.router)
app.include_router(classifier_controller.router)
app.include_router(bias_metrics_controller.router)
app.include_router(evaluation_controller.router)
