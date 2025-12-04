from typing import Generic, TypeVar, Optional
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.database.models import AuthModel, ModelsModel

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


class ModelsRepository(BaseRepository[ModelsModel]):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def add(self, obj: ModelsModel) -> ModelsModel:
        self.session.add(obj)
        await self.session.flush()
        return obj

    async def get_by_id(self, id: int) -> Optional[ModelsModel]:
        result = await self.session.execute(select(ModelsModel).where(ModelsModel.id == id))
        return result.scalar_one_or_none()

    async def get_by_user(self, user_id: int):
        result = await self.session.execute(select(ModelsModel.id, ModelsModel.name).where(ModelsModel.user_id == user_id))
        return result.all()

    async def create_model(self, user_id: int, name: str, stored_name: str, report: dict | None = None) -> ModelsModel:
        model = ModelsModel(user_id=user_id, name=name, stored_name=stored_name, report=report)
        await self.add(model)
        await self.session.commit()
        return model

    async def update_report(self, model_id: int, report: dict) -> Optional[ModelsModel]:
        model = await self.get_by_id(model_id)
        if not model:
            return None
        model.report = report
        await self.session.commit()
        return model
