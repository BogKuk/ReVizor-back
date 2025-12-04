from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy import JSON, ForeignKey
from src.database.db_main import Base


class AuthModel(Base):
    __tablename__ = "auth_database"

    id: Mapped[int] = mapped_column(primary_key=True)
    login: Mapped[str]
    password: Mapped[str]


class ModelsModel(Base):
    __tablename__ = "models_database"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("auth_database.id"))
    name: Mapped[str]
    stored_name: Mapped[str]
    report: Mapped[dict | None] = mapped_column(JSON, nullable=True)
