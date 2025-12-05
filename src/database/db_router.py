from fastapi import APIRouter

from src.database.db_main import engine
from src.database import models

router = APIRouter(prefix="/database", tags=["Database"])


@router.post("/init")
async def create_tables():
    async with engine.begin() as conn:
        await conn.run_sync(models.Base.metadata.create_all)
    return {"OK": True}
