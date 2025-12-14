from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.authorization.security import security
from src.database.db_main import get_session
from src.database.repositories import ModelsRepository

router = APIRouter(prefix="/analysis")

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
    return [{"id": row.id, "name": row.name} for row in rows]


@router.get("/models/{model_id}/analysis")
async def get_model_analysis(
    model_id: int,
    user_id: int = Depends(get_current_user_id),
    repo: ModelsRepository = Depends(get_models_repo),
):
    model = await repo.get_by_id(model_id)
    if not model or model.user_id != user_id:
        raise HTTPException(status_code=404, detail="Model not found")
    return model.report if model.report is not None else {"message": "no analysis yet"}


@router.get("/models/{model_id}/url")
async def get_model_url(
    model_id: int,
    user_id: int = Depends(get_current_user_id),
    repo: ModelsRepository = Depends(get_models_repo),
):
    model = await repo.get_by_id(model_id)
    if not model or model.user_id != user_id:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"name": model.name, "url": f"/models/{user_id}/{model.stored_name}"}
