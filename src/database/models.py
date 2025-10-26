from sqlalchemy.orm import Mapped, mapped_column
from src.database.db_main import Base


class AuthModel(Base):
    __tablename__ = "auth_database"

    id: Mapped[int] = mapped_column(primary_key=True)
    login: Mapped[str]
    password: Mapped[str]
