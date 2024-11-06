# app/service/item_service.py
from sqlalchemy.orm import Session
from typing import List, Optional
from app.repository.item_repository import ItemRepository
from app.model.item import ItemCreate, ItemUpdate
from app.schema.item_model import Item

class ItemService:
    def __init__(self, db: Session):
        self.repository = ItemRepository(db)

    def get_item(self, item_id: int) -> Optional[Item]:
        return self.repository.get_item(item_id)

    def get_items(self, skip: int = 0, limit: int = 10) -> List[Item]:
        return self.repository.get_items(skip, limit)

    def create_item(self, item: ItemCreate) -> Item:
        return self.repository.create_item(item)

    def update_item(self, item_id: int, item: ItemUpdate) -> Optional[Item]:
        return self.repository.update_item(item_id, item)

    def delete_item(self, item_id: int) -> bool:
        return self.repository.delete_item(item_id)
