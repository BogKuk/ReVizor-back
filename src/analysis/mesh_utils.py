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

def recolor_mesh_red(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.visual.face_colors = [255, 0, 0, 255]
    return mesh

def fix_and_color_inverted_polygons(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    working_mesh = mesh.copy()

    working_mesh.visual = trimesh.visual.ColorVisuals(mesh=working_mesh)

    working_mesh.merge_vertices()

    old_face_data = {}
    for i, face in enumerate(working_mesh.faces):
        key = tuple(sorted(face))
        old_face_data[key] = working_mesh.face_normals[i].copy()

    trimesh.repair.fix_winding(working_mesh)
    trimesh.repair.fix_normals(working_mesh)
    
    new_faces = working_mesh.faces
    new_face_normals = working_mesh.face_normals

    face_colors = np.full((len(new_faces), 4), [220, 220, 220, 255], dtype=np.uint8)

    for i, face in enumerate(new_faces):
        key = tuple(sorted(face))
        if key in old_face_data:
            old_normal = old_face_data[key]
            new_normal = new_face_normals[i]

            dot = np.dot(old_normal, new_normal)

            if dot < -0.5:
                face_colors[i] = [255, 0, 0, 255]

    working_mesh.visual.face_colors = face_colors
    
    return working_mesh

def save_mesh(mesh: trimesh.Trimesh, path: str):
    mesh.export(path)
