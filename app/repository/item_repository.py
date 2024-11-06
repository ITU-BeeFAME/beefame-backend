# app/repository/item_repository.py
from sqlalchemy.orm import Session
from typing import List, Optional
from app.schema.item_model import Item
from app.model.item import ItemCreate, ItemUpdate

class ItemRepository:
    def __init__(self, db: Session):
        self.db = db

    def get_item(self, item_id: int) -> Optional[Item]:
        return self.db.query(Item).filter(Item.id == item_id).first()

    def get_items(self, skip: int = 0, limit: int = 10) -> List[Item]:
        return self.db.query(Item).offset(skip).limit(limit).all()

    def create_item(self, item: ItemCreate) -> Item:
        db_item = Item(**item.dict())
        self.db.add(db_item)
        self.db.commit()
        self.db.refresh(db_item)
        return db_item

    def update_item(self, item_id: int, item: ItemUpdate) -> Optional[Item]:
        db_item = self.get_item(item_id)
        if db_item:
            for key, value in item.dict(exclude_unset=True).items():
                setattr(db_item, key, value)
            self.db.commit()
            self.db.refresh(db_item)
        return db_item

    def delete_item(self, item_id: int) -> bool:
        db_item = self.get_item(item_id)
        if db_item:
            self.db.delete(db_item)
            self.db.commit()
            return True
        return False
