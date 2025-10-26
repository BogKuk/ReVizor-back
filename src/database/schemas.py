from pydantic import BaseModel


class AuthAddSchema(BaseModel):
    login: str
    password: str


class AuthSchema(AuthAddSchema):
    id: int
