import os
from typing import Tuple
import numpy as np
import trimesh

UPLOAD_ROOT = "models"

def get_model_path(user_id: int, stored_name: str) -> str:
    return os.path.join(UPLOAD_ROOT, str(user_id), stored_name)

def load_mesh(path: str) -> trimesh.Trimesh:
    obj = trimesh.load(path, force="scene", skip_materials=True)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if hasattr(obj, "geometry"):
        geoms = list(obj.geometry.values())
        if not geoms:
            raise ValueError("empty_scene")
        return trimesh.util.concatenate(geoms)
    meshes = getattr(obj, "dump", None)
    if callable(meshes):
        parts = obj.dump()
        if isinstance(parts, list) and parts:
            return trimesh.util.concatenate([p for p in parts if isinstance(p, trimesh.Trimesh)])
    raise ValueError("unsupported_format")

def compute_metrics(mesh: trimesh.Trimesh) -> Tuple[int, float]:
    faces = int(mesh.faces.shape[0])
    area = float(mesh.area)
    density = float(np.inf) if area <= 1e-12 else faces / area
    return faces, density
