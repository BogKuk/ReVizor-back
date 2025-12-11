import os
import uuid

from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.authorization.security import security
from src.database.db_main import get_session
from src.database.repositories import ModelsRepository

router = APIRouter(prefix="/upload", tags=["Upload"])

ALLOWED_EXT = (".obj", ".fbx", ".glb", ".gltf")
UPLOAD_ROOT = "models"


async def get_models_repo(session: AsyncSession = Depends(get_session)) -> ModelsRepository:
    return ModelsRepository(session)


async def get_current_user_id(token=Depends(security.access_token_required)):
    return int(token.sub)


@router.post("/")
async def upload_model(
    file: UploadFile = File(...),
    user_id: int = Depends(get_current_user_id),
    repo: ModelsRepository = Depends(get_models_repo),
):
    original_name = file.filename.lower()
    if not original_name.endswith(ALLOWED_EXT):
        raise HTTPException(status_code=400, detail="Invalid file format. Allowed: .obj, .fbx, .glb, .gltf")

    extension = os.path.splitext(original_name)[1]
    stored_name = f"{uuid.uuid4().hex}{extension}"

    user_dir = os.path.join(UPLOAD_ROOT, str(user_id))
    os.makedirs(user_dir, exist_ok=True)

    save_path = os.path.join(user_dir, stored_name)

    try:
        file_bytes = await file.read()
        with open(save_path, "wb") as f:
            f.write(file_bytes)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save file: {str(e)}")

    new_model = await repo.create_model(
        user_id=user_id,
        name=file.filename,
        stored_name=stored_name,
    )

    return {
        "name": new_model.name,
        "url": f"/models/{user_id}/{stored_name}",
    }
