from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from typing import Annotated

from src.database.db_main import get_session
from src.database import models
from src.database import schemas

router = APIRouter(
    prefix="/database"
)

SessionDep = Annotated[AsyncSession, Depends(get_session)]


@router.post("/auth")
async def add_auth(data: schemas.AuthAddSchema, session: SessionDep):
    print("Password to hash:", data.password, type(data.password))
    new_auth = models.AuthModel(
        login=data.login,
        password=data.password,
    )
    session.add(new_auth)
    await session.commit()
    return {"OK": True}


@router.get("/auth")
async def get_auth(auth):
    ...
