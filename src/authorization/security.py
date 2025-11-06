from datetime import timedelta

from passlib.context import CryptContext
from authx import AuthX, AuthXConfig
from sqlalchemy.orm import Mapped
from fastapi import Request
from fastapi.responses import JSONResponse

from src.config import settings

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

auth_config = AuthXConfig()
auth_config.JWT_SECRET_KEY = settings.JWT_SECRET_KEY
auth_config.JWT_ACCESS_COOKIE_NAME = "user_access_token"
auth_config.JWT_REFRESH_COOKIE_NAME = "refresh_token"
auth_config.JWT_TOKEN_LOCATION = ["cookies"]
auth_config.JWT_ACCESS_TOKEN_EXPIRES = timedelta(minutes=15)
auth_config.JWT_REFRESH_TOKEN_EXPIRES = timedelta(days=7)

security = AuthX(config=auth_config)


async def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


async def get_tokens(uid: Mapped[int]):
    access_token = security.create_access_token(uid=str(uid))
    refresh_token = security.create_refresh_token(uid=str(uid))

    response = JSONResponse(content={"access_token": access_token})

    security.set_refresh_cookies(refresh_token, response)

    return response


async def refresh_token(request: Request):
    refresh_payload = await security.refresh_token_required(request)
    access_token = security.create_access_token(refresh_payload.sub)

    return access_token
