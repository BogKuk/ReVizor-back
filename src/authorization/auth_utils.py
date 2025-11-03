from passlib.context import CryptContext
from authx import AuthX, AuthXConfig
from sqlalchemy.orm import Mapped

from src.config import settings

pwd_context = CryptContext(schemes=["argon2"], deprecated="auto")

auth_config = AuthXConfig()
auth_config.JWT_SECRET_KEY = settings.JWT_SECRET_KEY
auth_config.JWT_ACCESS_COOKIE_NAME = "user_access_token"
auth_config.JWT_TOKEN_LOCATION = ["cookies"]

security = AuthX(config=auth_config)


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_token(uid: Mapped[int]):
    return security.create_access_token(uid=str(uid))
