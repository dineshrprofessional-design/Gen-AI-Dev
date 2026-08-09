from fastapi import APIRouter, HTTPException, status

from app.models import Item, ItemCreate, ItemUpdate

router = APIRouter(prefix="/items", tags=["items"])

# In-memory store; swap for a real database when you need persistence.
_items: dict[int, Item] = {}
_next_id = 1


@router.get("", response_model=list[Item])
def list_items(skip: int = 0, limit: int = 50) -> list[Item]:
    return list(_items.values())[skip : skip + limit]


@router.post("", response_model=Item, status_code=status.HTTP_201_CREATED)
def create_item(payload: ItemCreate) -> Item:
    global _next_id
    item = Item(id=_next_id, **payload.model_dump())
    _items[item.id] = item
    _next_id += 1
    return item


@router.get("/{item_id}", response_model=Item)
def get_item(item_id: int) -> Item:
    item = _items.get(item_id)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")
    return item


@router.patch("/{item_id}", response_model=Item)
def update_item(item_id: int, payload: ItemUpdate) -> Item:
    item = _items.get(item_id)
    if item is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")
    updated = item.model_copy(update=payload.model_dump(exclude_unset=True))
    _items[item_id] = updated
    return updated


@router.delete("/{item_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_item(item_id: int) -> None:
    if _items.pop(item_id, None) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Item not found")
