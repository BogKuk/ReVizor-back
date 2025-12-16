from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.authorization.security import security
from src.database.db_main import get_session
from src.database.repositories import ModelsRepository
from src.analysis.schemas import AnalyzeParams
from src.analysis.thresholds import get_thresholds
from src.analysis.mesh_utils import get_model_path, load_mesh, compute_metrics
from src.analysis.thresholds import THRESHOLDS

router = APIRouter(prefix="/analysis")

SessionDep = Annotated[AsyncSession, Depends(get_session)]


async def get_models_repo(session: AsyncSession = Depends(get_session)) -> ModelsRepository:
    return ModelsRepository(session)


async def get_current_user_id(token=Depends(security.access_token_required)):
    return int(token.sub)


@router.get("/models/names")
async def get_model_names(
    user_id: int = Depends(get_current_user_id),
    repo: ModelsRepository = Depends(get_models_repo),
):
    rows = await repo.get_by_user(user_id)
    return [{"id": row.id, "name": row.name} for row in rows]


@router.get("/models/{model_id}/analysis")
async def get_model_analysis(
    model_id: int,
    user_id: int = Depends(get_current_user_id),
    repo: ModelsRepository = Depends(get_models_repo),
):
    model = await repo.get_by_id(model_id)
    if not model or model.user_id != user_id:
        raise HTTPException(status_code=404, detail="Model not found")
    return model.report if model.report is not None else {"message": "no analysis yet"}


@router.get("/models/{model_id}/url")
async def get_model_url(
    model_id: int,
    user_id: int = Depends(get_current_user_id),
    repo: ModelsRepository = Depends(get_models_repo),
):
    model = await repo.get_by_id(model_id)
    if not model or model.user_id != user_id:
        raise HTTPException(status_code=404, detail="Model not found")
    return {"name": model.name, "url": f"/models/{user_id}/{model.stored_name}"}

@router.get("/options")
async def get_analysis_options(
    user_id: int = Depends(get_current_user_id),
):
    game_types = list(THRESHOLDS.keys())
    usage_map = {gt: list(THRESHOLDS[gt].keys()) for gt in game_types}
    return {"game_types": game_types, "usage_areas_by_game_type": usage_map}


@router.post("/models/{model_id}/analyze")
async def analyze_model(
    model_id: int,
    params: AnalyzeParams,
    user_id: int = Depends(get_current_user_id),
    repo: ModelsRepository = Depends(get_models_repo),
):
    model = await repo.get_by_id(model_id)
    if not model or model.user_id != user_id:
        raise HTTPException(status_code=404, detail="Model not found")
    path = get_model_path(user_id, model.stored_name)
    try:
        mesh = load_mesh(path)
    except Exception:
        raise HTTPException(status_code=400, detail="Failed to load model")
    faces, density = compute_metrics(mesh)
    try:
        max_faces, max_density = get_thresholds(params.game_type, params.usage_area)
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Invalid parameter: {str(e)}")
    payload = {
        "params": params.model_dump(),
        "metrics": {"faces": faces, "density": density},
        "limits": {"max_faces": max_faces, "max_density": max_density},
        "result": {"faces_ok": faces <= max_faces, "density_ok": density <= max_density},
    }
    await repo.update_report(model_id, payload)
    return payload
