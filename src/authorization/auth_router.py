from fastapi import APIRouter, HTTPException
from authx import AuthX, AuthXConfig
from pydantic import BaseModel
from src.config import settings

auth_config = AuthXConfig()
auth_config.JWT_SECRET_KEY = settings.JWT_SECRET_KEY
auth_config.JWT_ACCESS_COOKIE_NAME = "user_access_token"
auth_config.JWT_TOKEN_LOCATION = ["cookies"]

security = AuthX(config=auth_config)


class UserLoginSchema(BaseModel):
    username: str
    password: str


router = APIRouter(
    prefix="/authorization"
)


@router.post("/login")
async def login(creds: UserLoginSchema):
    if creds.username == "test" and creds.password == "test":
        token = security.create_access_token(uid="1")
        return {"access_token": token}
    raise HTTPException(status_code=401, detail="Incorrect username or password")
