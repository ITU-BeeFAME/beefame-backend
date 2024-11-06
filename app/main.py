# app/main.py
from fastapi import FastAPI
from app.schema.item_model import Item
from app.schema.item_model import Base  # Import Base from schema
from app.database import engine
from app.controller import item_controller

# Veritabanı tablolarını oluştur
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="BeeFair REST API",
    description="BeeFair Fair Testing Tool REST API",
    version="1.0.0",
)

# Router'ları ekle
app.include_router(item_controller.router)
