from typing import Annotated
import os

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from src.authorization.security import security
from src.database.db_main import get_session
from src.database.repositories import ModelsRepository
from src.analysis.schemas import AnalyzeParams
from src.analysis.thresholds import get_thresholds
from src.analysis.mesh_utils import (
    get_model_path,
    get_model_variant_path,
    load_mesh,
    compute_metrics,
    recolor_mesh_red,
    fix_and_color_inverted_polygons,
    color_by_face_density,
    save_mesh,
    save_uv_svg_from_path,
    compute_uv_overlap_from_path,
    compute_uv_distortion_from_path,
    compute_texel_density_from_path,
)
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

    user_dir = os.path.join("models", str(user_id))
    new_path = os.path.join(user_dir, model.stored_name, model.stored_name)
    old_path = os.path.join(user_dir, model.stored_name)

    if os.path.exists(new_path):
        url = f"/models/{user_id}/{model.stored_name}/{model.stored_name}"
    elif os.path.exists(old_path):
        url = f"/models/{user_id}/{model.stored_name}"
    else:
        url = f"/models/{user_id}/{model.stored_name}/{model.stored_name}"

    res = {
        "name": model.name,
        "url": url,
    }

    if model.report:
        if "recolored_model_url" in model.report:
            res["recolored_url"] = model.report["recolored_model_url"]
        if "density_model_url" in model.report:
            res["density_url"] = model.report["density_model_url"]
        if "uv_image_url" in model.report:
            res["uv_url"] = model.report["uv_image_url"]
        if "uv_overlap_url" in model.report:
            res["uv_overlap_url"] = model.report["uv_overlap_url"]
        if "uv_distortion_url" in model.report:
            res["uv_distortion_url"] = model.report["uv_distortion_url"]
        if "uv_texel_density_url" in model.report:
            res["uv_texel_density_url"] = model.report["uv_texel_density_url"]
        
    return res

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

    recolored_name = "recolored.glb"
    recolored_path = get_model_variant_path(user_id, model.stored_name, recolored_name)
    try:
        recolored_mesh = fix_and_color_inverted_polygons(mesh)
        save_mesh(recolored_mesh, recolored_path)
    except Exception as e:
        print(f"Error creating recolored mesh: {e}")
        recolored_name = None
    density_name = "density.glb"
    density_path = get_model_variant_path(user_id, model.stored_name, density_name)
    try:
        density_mesh = color_by_face_density(mesh)
        save_mesh(density_mesh, density_path)
    except Exception as e:
        print(f"Error creating density mesh: {e}")
        density_name = None

    uv_name = "uv_layout.svg"
    uv_path = get_model_variant_path(user_id, model.stored_name, uv_name)
    
    # Генерируем все варианты UV разверток
    uv_present = save_uv_svg_from_path(path, uv_path, mode="original")
    
    uv_overlap_name = "uv_overlap.svg"
    uv_overlap_path = get_model_variant_path(user_id, model.stored_name, uv_overlap_name)
    
    uv_distortion_name = "uv_distortion.svg"
    uv_distortion_path = get_model_variant_path(user_id, model.stored_name, uv_distortion_name)
    
    uv_texel_name = "uv_texel_density.svg"
    uv_texel_path = get_model_variant_path(user_id, model.stored_name, uv_texel_name)
    
    uv_overlap = 0.0
    uv_distortion = 0.0
    texel_density = 0.0
    texel_uniformity = 0.0
    
    if uv_present:
        # Генерируем дополнительные визуализации
        save_uv_svg_from_path(path, uv_overlap_path, mode="overlap")
        save_uv_svg_from_path(path, uv_distortion_path, mode="distortion")
        save_uv_svg_from_path(path, uv_texel_path, mode="texel_density")
        
        # Считаем метрики
        uv_overlap = compute_uv_overlap_from_path(path)
        uv_distortion = compute_uv_distortion_from_path(path)
        texel_res = compute_texel_density_from_path(path)
        texel_density = texel_res["avg_density"]
        texel_uniformity = texel_res["uniformity"]

    payload = {
        "params": params.model_dump(),
        "metrics": {
            "faces": faces, 
            "density": density, 
            "uv_overlap": uv_overlap,
            "uv_distortion": uv_distortion,
            "texel_density": texel_density,
            "texel_uniformity": texel_uniformity
        },
        "limits": {
            "max_faces": max_faces, 
            "max_density": max_density, 
            "max_uv_overlap": 0.0,
            "max_uv_distortion": 0.5,
            "max_texel_uniformity": 0.2
        },
        "result": {
            "faces_ok": faces <= max_faces, 
            "density_ok": density <= max_density,
            "uv_overlap_ok": uv_overlap <= 0.001,
            "uv_distortion_ok": uv_distortion <= 0.5,
            "texel_uniformity_ok": texel_uniformity <= 0.2
        },
        "uv_present": uv_present,
    }

    if recolored_name:
        import time
        payload["recolored_model_url"] = f"/models/{user_id}/{model.stored_name}/{recolored_name}?t={int(time.time())}"
    if density_name:
        import time
        payload["density_model_url"] = f"/models/{user_id}/{model.stored_name}/{density_name}?t={int(time.time())}"
    
    # Добавляем все URL для UV
    if uv_present:
        import time
        t = int(time.time())
        payload["uv_image_url"] = f"/models/{user_id}/{model.stored_name}/{uv_name}?t={t}"
        payload["uv_overlap_url"] = f"/models/{user_id}/{model.stored_name}/{uv_overlap_name}?t={t}"
        payload["uv_distortion_url"] = f"/models/{user_id}/{model.stored_name}/{uv_distortion_name}?t={t}"
        payload["uv_texel_density_url"] = f"/models/{user_id}/{model.stored_name}/{uv_texel_name}?t={t}"

    await repo.update_report(model_id, payload)
    return payload
