from typing import Annotated

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from src.authorization.security import security
from src.database.db_main import get_session
from src.database.repositories import ModelsRepository

router = APIRouter(prefix="/analysis", tags=["Analysis"])

SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_models_repo(session: AsyncSession = Depends(get_session)) -> ModelsRepository:
    return ModelsRepository(session)


async def get_current_user_id(token=Depends(security.access_token_required)):
    return int(token.sub)


@router.get("/models/names")
async def get_model_names(
    user_id: int = Depends(get_current_user_id),
    repo: ModelsRepository = Depends(get_models_repo),
):
    rows = await repo.get_by_user(user_id)
    names = [row.name for row in rows]
    return names
