import os
from typing import Tuple
import numpy as np
import trimesh

UPLOAD_ROOT = "models"

def get_model_dir(user_id: int, stored_name: str) -> str:
    return os.path.join(UPLOAD_ROOT, str(user_id), stored_name)

def get_model_path(user_id: int, stored_name: str) -> str:
    base_dir = get_model_dir(user_id, stored_name)
    candidate = os.path.join(base_dir, stored_name)
    if os.path.exists(candidate):
        return candidate
    return os.path.join(UPLOAD_ROOT, str(user_id), stored_name)

def get_model_variant_path(user_id: int, stored_name: str, variant_name: str) -> str:
    base_dir = get_model_dir(user_id, stored_name)
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, variant_name)

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

    v = working_mesh.vertices
    f = working_mesh.faces
    new_face_normals = working_mesh.face_normals

    red_mask = np.zeros(len(f), dtype=bool)
    for i, face in enumerate(f):
        key = tuple(sorted(face))
        if key in old_face_data:
            old_normal = old_face_data[key]
            new_normal = new_face_normals[i]
            dot = np.dot(old_normal, new_normal)
            if dot < -0.5:
                red_mask[i] = True

    new_vertices = v[f].reshape(-1, 3)
    new_faces = np.arange(len(f) * 3, dtype=np.int64).reshape(-1, 3)
    base_color = np.array([220, 220, 220, 255], dtype=np.uint8)
    red_color = np.array([255, 0, 0, 255], dtype=np.uint8)
    face_colors = np.where(red_mask[:, None], red_color, base_color)
    vertex_colors = np.repeat(face_colors, 3, axis=0)

    result = trimesh.Trimesh(
        vertices=new_vertices,
        faces=new_faces,
        vertex_colors=vertex_colors,
        process=False
    )
    return result

def save_mesh(mesh: trimesh.Trimesh, path: str):
    mesh.export(path)

def color_by_face_density(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    m = mesh.copy()
    areas = getattr(m, "area_faces", None)
    if areas is None:
        tri = m.triangles
        areas = trimesh.triangles.area(tri)
    x = np.log(areas + 1e-12)
    med = float(np.median(x))
    mad = float(np.median(np.abs(x - med))) + 1e-6
    d = (x - med) / (2.0 * mad)
    d = np.clip(d, -1.0, 1.0)
    base = np.array([180, 180, 180, 255], dtype=np.uint8)
    red = np.array([255, 0, 0, 255], dtype=np.uint8)
    blue = np.array([0, 64, 255, 255], dtype=np.uint8)
    face_colors = np.zeros((len(d), 4), dtype=np.uint8)
    for i, val in enumerate(d):
        if val < 0:
            t = -val
            face_colors[i] = (base * (1 - t) + red * t).astype(np.uint8)
        elif val > 0:
            t = val
            face_colors[i] = (base * (1 - t) + blue * t).astype(np.uint8)
        else:
            face_colors[i] = base
    v = m.vertices
    f = m.faces
    new_vertices = v[f].reshape(-1, 3)
    new_faces = np.arange(len(f) * 3, dtype=np.int64).reshape(-1, 3)
    vertex_colors = np.repeat(face_colors, 3, axis=0)
    res = trimesh.Trimesh(vertices=new_vertices, faces=new_faces, vertex_colors=vertex_colors, process=False)
    return res
