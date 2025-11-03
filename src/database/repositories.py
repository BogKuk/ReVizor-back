from typing import Generic, TypeVar, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import AuthModel

T = TypeVar("T")

class RepositoryError(Exception):
    pass

class BaseRepository(Generic[T]):
    async def get_by_id(self, id: int) -> Optional[T]:
        raise NotImplementedError

    async def add(self, obj: T) -> T:
        raise NotImplementedError

class AuthRepository(BaseRepository[AuthModel]):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def get_by_login(self, login: str) -> Optional[AuthModel]:
        result = await self.session.execute(select(AuthModel).where(AuthModel.login == login))
        return result.scalar_one_or_none()

    async def _add(self, auth: AuthModel) -> AuthModel:
        self.session.add(auth)
        await self.session.flush()
        return auth

    async def create_user(self, login: str, password_hash: str) -> AuthModel:
        new_user = AuthModel(login=login, password=password_hash)
        await self._add(new_user)
        await self.session.commit()
        return new_user
