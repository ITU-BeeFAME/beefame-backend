# app/main.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from controller import analysis_router, dataset_router, method_router, classifier_router, evaluation_router

app = FastAPI(
    title="BeeFair REST API",
    description="BeeFair Fair Testing Tool REST API",
    version="1.0.1",
)

# CORS related configuration 
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Frontend URL:port
    allow_credentials=True,
    allow_methods=["*"],  
    allow_headers=["*"],
)

# Include routers
app.include_router(analysis_router)
app.include_router(dataset_router)
app.include_router(method_router)
app.include_router(classifier_router)
app.include_router(evaluation_router)
