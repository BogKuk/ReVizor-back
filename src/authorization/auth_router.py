from typing import Annotated

from fastapi import APIRouter, HTTPException, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from src.authorization.security import verify_password, get_tokens, refresh_token, get_password_hash
from src.database import schemas
from src.database.db_main import get_session
from src.database.repositories import AuthRepository

SessionDep = Annotated[AsyncSession, Depends(get_session)]

router = APIRouter(
    prefix="/authorization"
)


async def get_auth_repo(session: AsyncSession = Depends(get_session)) -> AuthRepository:
    return AuthRepository(session)


@router.post("/login")
async def login(data: schemas.AuthAddSchema, repo: AuthRepository = Depends(get_auth_repo)):
    user = await repo.get_by_login(data.login)

    if not user or not await verify_password(data.password, user.password):
        raise HTTPException(status_code=401, detail="Incorrect login or password")

    response = await get_tokens(user.id)
    return response


@router.post("/refresh")
async def refresh(request: Request):
    try:
        response = await refresh_token(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=401, detail=str(e))


@router.post("/register")
async def register(data: schemas.AuthAddSchema, repo: AuthRepository = Depends(get_auth_repo)):
    existing = await repo.get_by_login(data.login)

    if existing:
        raise HTTPException(status_code=409, detail="User already exists")

    password_hash = await get_password_hash(data.password)
    user = await repo.create_user(login=data.login, password_hash=password_hash)

    response = await get_tokens(user.id)
    return response
