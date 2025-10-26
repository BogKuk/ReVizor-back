from typing import Annotated

from fastapi import APIRouter, HTTPException, Depends
from authx import AuthX, AuthXConfig
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.authorization.auth_utils import verify_password
from src.config import settings
from src.database.db_main import get_session
from src.database import schemas, models

auth_config = AuthXConfig()
auth_config.JWT_SECRET_KEY = settings.JWT_SECRET_KEY
auth_config.JWT_ACCESS_COOKIE_NAME = "user_access_token"
auth_config.JWT_TOKEN_LOCATION = ["cookies"]

security = AuthX(config=auth_config)


SessionDep = Annotated[AsyncSession, Depends(get_session)]


class UserLoginSchema(BaseModel):
    username: str
    password: str


router = APIRouter(
    prefix="/authorization"
)


@router.post("/login")
async def login(data: schemas.AuthAddSchema, session: AsyncSession = Depends(get_session)):
    user = await session.execute(
        select(models.AuthModel).where(models.AuthModel.login == data.login)
    )
    user = user.scalar_one_or_none()

    if not user or not verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Incorrect login or password")

    token = security.create_access_token(uid=str(user.id))
    return {"access_token": token}
