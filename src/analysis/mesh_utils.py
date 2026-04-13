import os
from typing import Tuple, List, Dict
import numpy as np
import trimesh
from shapely.geometry import Polygon
from shapely.ops import unary_union

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
    obj = trimesh.load(path, force="scene", skip_materials=False)
    if isinstance(obj, trimesh.Trimesh):
        return obj
    if hasattr(obj, "geometry"):
        geoms = list(obj.geometry.values())
        if not geoms:
            raise ValueError("empty_scene")
        if len(geoms) == 1 and isinstance(geoms[0], trimesh.Trimesh):
            return geoms[0]
        return trimesh.util.concatenate([g for g in geoms if isinstance(g, trimesh.Trimesh)])
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

def _extract_uv(mesh: trimesh.Trimesh) -> np.ndarray | None:
    vis = getattr(mesh, "visual", None)
    if vis is None:
        return None

    uv = getattr(vis, "uv", None)

    if uv is None:
        attrs = getattr(vis, "vertex_attributes", None)
        if isinstance(attrs, dict):
            # Список возможных имен для UV координат в разных форматах
            candidates = ["uv", "uv_0", "uv0", "texcoord", "texcoord_0", "texcoord0", "texture"]
            for cand in candidates:
                if cand in attrs:
                    uv = attrs[cand]
                    break

    if uv is None:
        if hasattr(mesh, 'vertex_attributes'):
             attrs = mesh.vertex_attributes
             candidates = ["uv", "uv_0", "uv0", "texcoord", "texcoord_0", "texcoord0"]
             for cand in candidates:
                 if cand in attrs:
                     uv = attrs[cand]
                     break

    if uv is None:
        return None
        
    arr = np.asarray(uv, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2:
        return None
    return arr[:, :2]

def has_uv(mesh: trimesh.Trimesh) -> bool:
    uv = _extract_uv(mesh)
    return uv is not None and uv.shape[0] >= 3

def generate_uv_svg(mesh: trimesh.Trimesh, size: int = 1024, stroke: int = 1, face_colors: np.ndarray = None) -> str:
    uv = _extract_uv(mesh)
    if uv is None:
        raise ValueError("no_uv")
    f = mesh.faces
    n_verts = mesh.vertices.shape[0]
    n_faces = f.shape[0]
    n_corners = n_faces * 3
    n_uv = uv.shape[0]
    if n_uv == n_verts:
        uvf = uv[f]
    elif n_uv == n_corners:
        uvf = uv.reshape((-1, 3, 2))
    else:
        raise ValueError("uv_mismatch")
    u = uvf[..., 0]
    v = uvf[..., 1]
    finite = np.isfinite(u) & np.isfinite(v)
    mask = finite.all(axis=1)
    uvf = uvf[mask]
    
    # Также фильтруем цвета, если они есть
    current_face_colors = face_colors[mask] if face_colors is not None else None
    
    if uvf.shape[0] == 0:
        raise ValueError("no_uv")
    mins = uvf.min(axis=(0, 1))
    maxs = uvf.max(axis=(0, 1))
    span = maxs - mins
    span[span == 0] = 1.0
    norm = (uvf - mins) / span
    w = float(size)
    h = float(size)
    pts = norm.copy()
    pts[..., 0] = pts[..., 0] * (w - 2.0) + 1.0
    pts[..., 1] = (1.0 - pts[..., 1]) * (h - 2.0) + 1.0
    header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
    bg = f'<rect x="0" y="0" width="{size}" height="{size}" fill="#111"/>'
    lines = []
    
    for i, tri in enumerate(pts):
        x1, y1 = tri[0]
        x2, y2 = tri[1]
        x3, y3 = tri[2]
        
        fill = "none"
        if current_face_colors is not None:
            c = current_face_colors[i]
            # Предполагаем, что c - это [R, G, B] или [R, G, B, A]
            fill = f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"
            if len(c) > 3:
                fill = f"rgba({int(c[0])},{int(c[1])},{int(c[2])},{c[3]/255.0:.2f})"
        
        path = f'<path d="M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f} L {x3:.2f} {y3:.2f} Z" stroke="#33d17a" stroke-width="{stroke}" fill="{fill}"/>'
        lines.append(path)
    grid = []
    for i in range(11):
        t = i / 10.0
        gx = t * (w - 2.0) + 1.0
        gy = t * (h - 2.0) + 1.0
        grid.append(f'<line x1="{gx:.2f}" y1="1" x2="{gx:.2f}" y2="{h-1:.2f}" stroke="#333" stroke-width="1"/>')
        grid.append(f'<line x1="1" y1="{gy:.2f}" x2="{w-1:.2f}" y2="{gy:.2f}" stroke="#333" stroke-width="1"/>')
    content = header + bg + "".join(grid) + "".join(lines) + "</svg>"
    return content

def save_uv_svg(mesh: trimesh.Trimesh, path: str, size: int = 1024, stroke: int = 1):
    svg = generate_uv_svg(mesh, size=size, stroke=stroke)
    with open(path, "w", encoding="utf-8") as f:
        f.write(svg)

def get_uv_overlap_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Возвращает цвета граней для визуализации перекрытий."""
    uv = _extract_uv(mesh)
    if uv is None:
        return None
    
    faces = mesh.faces
    if uv.shape[0] == mesh.vertices.shape[0]:
        uv_faces = uv[faces]
    else:
        uv_faces = uv.reshape((-1, 3, 2))
        if uv_faces.shape[0] != faces.shape[0]:
            return None

    polygons = []
    for f_uv in uv_faces:
        try:
            poly = Polygon(f_uv)
            if not poly.is_valid:
                poly = poly.buffer(0)
            polygons.append(poly)
        except:
            polygons.append(Polygon())

    n = len(polygons)
    # По умолчанию серый
    colors = np.full((n, 4), [100, 100, 100, 100], dtype=np.uint8)
    
    # Очень простой поиск пересечений: для каждой грани проверяем пересечение с объединением остальных
    # Но это медленно O(N^2). Для визуализации лучше использовать дерево индексов.
    from shapely.strtree import STRtree
    tree = STRtree(polygons)
    
    for i, poly in enumerate(polygons):
        if poly.is_empty or poly.area < 1e-12:
            continue
            
        # Находим всех кандидатов
        indices = tree.query(poly)
        for idx in indices:
            if idx == i:
                continue
            
            other = polygons[idx]
            if other.is_empty:
                continue
                
            # Если есть значимое пересечение
            if poly.intersects(other):
                inter = poly.intersection(other)
                if inter.area > 1e-12:
                    colors[i] = [255, 0, 0, 200] # Красный для перекрытий
                    break
                    
    return colors

def get_uv_distortion_colors(mesh: trimesh.Trimesh) -> np.ndarray:
    """Возвращает цвета граней для визуализации искажений."""
    uv = _extract_uv(mesh)
    if uv is None:
        return None
        
    faces = mesh.faces
    area_3d = mesh.area_faces
    
    if uv.shape[0] == mesh.vertices.shape[0]:
        uv_faces = uv[faces]
    else:
        uv_faces = uv.reshape((-1, 3, 2))
        
    u = uv_faces[:, :, 0]
    v = uv_faces[:, :, 1]
    area_uv = 0.5 * np.abs(u[:,0]*(v[:,1]-v[:,2]) + u[:,1]*(v[:,2]-v[:,0]) + u[:,2]*(v[:,0]-v[:,1]))
    
    valid = area_3d > 1e-12
    ratios = np.ones(len(faces))
    
    sum_3d = np.sum(area_3d[valid])
    sum_uv = np.sum(area_uv[valid])
    
    if sum_uv > 1e-12 and sum_3d > 1e-12:
        ratios[valid] = (area_uv[valid] / sum_uv) / (area_3d[valid] / sum_3d)
    
    # Логарифмическое искажение
    dist = np.abs(np.log(np.clip(ratios, 0.1, 10.0)))
    dist_norm = np.clip(dist / np.log(2.0), 0, 1) # 0 - нет искажения, 1 - 2x искажение и выше
    
    colors = np.zeros((len(faces), 4), dtype=np.uint8)
    base = np.array([180, 180, 180, 150], dtype=np.uint8)
    red = np.array([255, 0, 0, 200], dtype=np.uint8)
    
    for i in range(len(faces)):
        t = dist_norm[i]
        colors[i] = (base * (1 - t) + red * t).astype(np.uint8)
        
    return colors

def get_uv_texel_density_colors(mesh: trimesh.Trimesh, resolution: int = 1024) -> np.ndarray:
    """Возвращает цвета граней для визуализации плотности текселей."""
    uv = _extract_uv(mesh)
    if uv is None:
        return None
        
    faces = mesh.faces
    area_3d = mesh.area_faces
    
    if uv.shape[0] == mesh.vertices.shape[0]:
        uv_faces = uv[faces]
    else:
        uv_faces = uv.reshape((-1, 3, 2))
        
    u = uv_faces[:, :, 0]
    v = uv_faces[:, :, 1]
    area_uv = 0.5 * np.abs(u[:,0]*(v[:,1]-v[:,2]) + u[:,1]*(v[:,2]-v[:,0]) + u[:,2]*(v[:,0]-v[:,1]))
    
    total_pixels = resolution * resolution
    densities_sq = np.zeros(len(faces))
    valid = area_3d > 1e-12
    densities_sq[valid] = (total_pixels * area_uv[valid]) / area_3d[valid]
    densities = np.sqrt(np.clip(densities_sq, 0, None))
    
    if len(densities[valid]) == 0:
        return np.full((len(faces), 4), [180, 180, 180, 150], dtype=np.uint8)
        
    avg = np.mean(densities[valid])
    if avg < 1e-6:
        return np.full((len(faces), 4), [180, 180, 180, 150], dtype=np.uint8)
        
    # Отклонение от среднего
    diff = (densities - avg) / (avg + 1e-6)
    diff = np.clip(diff, -1.0, 1.0)
    
    colors = np.zeros((len(faces), 4), dtype=np.uint8)
    base = np.array([180, 180, 180, 150], dtype=np.uint8)
    red = np.array([255, 0, 0, 200], dtype=np.uint8)
    blue = np.array([0, 64, 255, 200], dtype=np.uint8)
    
    for i in range(len(faces)):
        val = diff[i]
        if val < 0:
            t = -val
            colors[i] = (base * (1 - t) + red * t).astype(np.uint8)
        elif val > 0:
            t = val
            colors[i] = (base * (1 - t) + blue * t).astype(np.uint8)
        else:
            colors[i] = base
            
    return colors

def generate_uv_svg_from_path(path: str, size: int = 1024, stroke: int = 1, mode: str = "original") -> str:
    obj = trimesh.load(path, force="scene", skip_materials=False)
    tris = []
    colors_list = []
    
    def process_geom(g):
        uv = _extract_uv(g)
        if uv is None:
            return None, None
        
        f = g.faces
        n_verts = g.vertices.shape[0]
        n_faces = f.shape[0]
        n_corners = n_faces * 3
        n_uv = uv.shape[0]
        
        if n_uv == n_verts:
            uvf = uv[f]
        elif n_uv == n_corners:
            uvf = uv.reshape((-1, 3, 2))
        else:
            return None, None
            
        colors = None
        if mode == "overlap":
            colors = get_uv_overlap_colors(g)
        elif mode == "distortion":
            colors = get_uv_distortion_colors(g)
        elif mode == "texel_density":
            colors = get_uv_texel_density_colors(g)
            
        return uvf, colors

    if isinstance(obj, trimesh.Trimesh):
        uvf, colors = process_geom(obj)
        if uvf is not None:
            tris.append(uvf)
            if colors is not None:
                colors_list.append(colors)
    elif hasattr(obj, "geometry"):
        for g in obj.geometry.values():
            if not isinstance(g, trimesh.Trimesh):
                continue
            uvf, colors = process_geom(g)
            if uvf is not None:
                tris.append(uvf)
                if colors is not None:
                    colors_list.append(colors)
    else:
        raise ValueError("unsupported_format")
        
    if not tris:
        raise ValueError("no_uv")
        
    all_tris = np.concatenate(tris, axis=0)
    all_colors = np.concatenate(colors_list, axis=0) if colors_list else None
    
    # Фильтрация по маске конечных координат (как в оригинале)
    u = all_tris[..., 0]
    v = all_tris[..., 1]
    finite = np.isfinite(u) & np.isfinite(v)
    mask = finite.all(axis=1)
    
    filtered_tris = all_tris[mask]
    filtered_colors = all_colors[mask] if all_colors is not None else None
    
    if filtered_tris.shape[0] == 0:
        raise ValueError("no_uv")
        
    # Нормализация и создание SVG (вызываем базовую функцию, но передаем подготовленные данные)
    # Но generate_uv_svg ожидает меш. Чтобы не создавать фейковый меш, я просто скопирую логику отрисовки.
    
    mins = filtered_tris.min(axis=(0, 1))
    maxs = filtered_tris.max(axis=(0, 1))
    span = maxs - mins
    span[span == 0] = 1.0
    norm = (filtered_tris - mins) / span
    
    w = float(size)
    h = float(size)
    pts = norm.copy()
    pts[..., 0] = pts[..., 0] * (w - 2.0) + 1.0
    pts[..., 1] = (1.0 - pts[..., 1]) * (h - 2.0) + 1.0
    
    header = f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
    bg = f'<rect x="0" y="0" width="{size}" height="{size}" fill="#111"/>'
    lines = []
    
    for i, tri in enumerate(pts):
        x1, y1 = tri[0]
        x2, y2 = tri[1]
        x3, y3 = tri[2]
        
        fill = "none"
        if filtered_colors is not None:
            c = filtered_colors[i]
            fill = f"rgb({int(c[0])},{int(c[1])},{int(c[2])})"
            if len(c) > 3:
                fill = f"rgba({int(c[0])},{int(c[1])},{int(c[2])},{c[3]/255.0:.2f})"
        
        path = f'<path d="M {x1:.2f} {y1:.2f} L {x2:.2f} {y2:.2f} L {x3:.2f} {y3:.2f} Z" stroke="#33d17a" stroke-width="{stroke}" fill="{fill}"/>'
        lines.append(path)
        
    grid = []
    for i in range(11):
        t = i / 10.0
        gx = t * (w - 2.0) + 1.0
        gy = t * (h - 2.0) + 1.0
        grid.append(f'<line x1="{gx:.2f}" y1="1" x2="{gx:.2f}" y2="{h-1:.2f}" stroke="#333" stroke-width="1"/>')
        grid.append(f'<line x1="1" y1="{gy:.2f}" x2="{w-1:.2f}" y2="{gy:.2f}" stroke="#333" stroke-width="1"/>')
        
    return header + bg + "".join(grid) + "".join(lines) + "</svg>"

def save_uv_svg_from_path(path: str, out_path: str, size: int = 1024, stroke: int = 1, mode: str = "original") -> bool:
    try:
        svg = generate_uv_svg_from_path(path, size=size, stroke=stroke, mode=mode)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(svg)
        return True
    except Exception as e:
        print(f"UV SVG Error (mode {mode}): {e}")
        return False

def compute_uv_overlap(mesh: trimesh.Trimesh) -> float:
    uv = _extract_uv(mesh)
    if uv is None:
        return 0.0
    
    faces = mesh.faces
    if uv.shape[0] == mesh.vertices.shape[0]:
        uv_faces = uv[faces]
    else:
        uv_faces = uv.reshape((-1, 3, 2))
        if uv_faces.shape[0] != faces.shape[0]:
            return 0.0

    polygons = []
    for f_uv in uv_faces:
        try:
            poly = Polygon(f_uv)
            if poly.is_valid and poly.area > 1e-9:
                polygons.append(poly)
        except:
            continue

    if not polygons:
        return 0.0
    
    try:
        # Суммарная площадь всех полигонов (с учетом наложений)
        total_area = sum(p.area for p in polygons)
        if total_area < 1e-12:
            return 0.0

        # Площадь объединения (без наложений)
        union_poly = unary_union(polygons)
        union_area = union_poly.area

        overlap_ratio = max(0.0, (total_area - union_area) / total_area)
        return float(overlap_ratio)
    except Exception as e:
        print(f"Overlap calculation error: {e}")
        return 0.0

def load_mesh_raw(path: str):
    return trimesh.load(path, force="scene", skip_materials=False)

def compute_uv_overlap_from_path(path: str) -> float:
    try:
        obj = load_mesh_raw(path)
        if isinstance(obj, trimesh.Scene):
            max_overlap = 0.0
            for g in obj.geometry.values():
                if isinstance(g, trimesh.Trimesh):
                    overlap = compute_uv_overlap(g)
                    max_overlap = max(max_overlap, overlap)
            return max_overlap
        else:
            return compute_uv_overlap(obj)
    except Exception as e:
        print(f"Overlap error: {e}")
        return 0.0

def compute_uv_distortion(mesh: trimesh.Trimesh) -> float:
    uv = _extract_uv(mesh)
    if uv is None:
        return 0.0

    faces = mesh.faces
    vertices = mesh.vertices
    
    if uv.shape[0] == vertices.shape[0]:
        uv_faces = uv[faces]
    else:
        uv_faces = uv.reshape((-1, 3, 2))
        if uv_faces.shape[0] != faces.shape[0]:
            return 0.0

    area_3d = mesh.area_faces

    u = uv_faces[:, :, 0]
    v = uv_faces[:, :, 1]
    area_uv = 0.5 * np.abs(
        u[:, 0] * (v[:, 1] - v[:, 2]) + 
        u[:, 1] * (v[:, 2] - v[:, 0]) + 
        u[:, 2] * (v[:, 0] - v[:, 1])
    )

    valid_mask = area_3d > 1e-12
    if not np.any(valid_mask):
        return 0.0
    
    area_3d = area_3d[valid_mask]
    area_uv = area_uv[valid_mask]

    sum_3d = np.sum(area_3d)
    sum_uv = np.sum(area_uv)
    
    if sum_uv < 1e-12:
        return 1.0

    rel_3d = area_3d / sum_3d
    rel_uv = area_uv / sum_uv

    ratios = rel_uv / rel_3d

    distortions = np.abs(np.log(np.clip(ratios, 1e-5, 1e5)))

    mean_distortion = np.average(distortions, weights=area_3d)
    
    return float(mean_distortion)

def compute_uv_distortion_from_path(path: str) -> float:
    try:
        obj = load_mesh_raw(path)
        if isinstance(obj, trimesh.Scene):
            max_dist = 0.0
            for g in obj.geometry.values():
                if isinstance(g, trimesh.Trimesh):
                    dist = compute_uv_distortion(g)
                    max_dist = max(max_dist, dist)
            return max_dist
        else:
            return compute_uv_distortion(obj)
    except Exception as e:
        print(f"Distortion error: {e}")
        return 0.0

def compute_texel_density(mesh: trimesh.Trimesh, resolution: int = 1024) -> Dict[str, float]:
    uv = _extract_uv(mesh)
    if uv is None:
        return {"avg_density": 0.0, "uniformity": 0.0}

    faces = mesh.faces
    vertices = mesh.vertices
    
    if uv.shape[0] == vertices.shape[0]:
        uv_faces = uv[faces]
    else:
        uv_faces = uv.reshape((-1, 3, 2))
        if uv_faces.shape[0] != faces.shape[0]:
            return {"avg_density": 0.0, "uniformity": 0.0}

    area_3d = mesh.area_faces

    u = uv_faces[:, :, 0]
    v = uv_faces[:, :, 1]
    area_uv = 0.5 * np.abs(
        u[:, 0] * (v[:, 1] - v[:, 2]) + 
        u[:, 1] * (v[:, 2] - v[:, 0]) + 
        u[:, 2] * (v[:, 0] - v[:, 1])
    )

    valid_mask = area_3d > 1e-12
    if not np.any(valid_mask):
        return {"avg_density": 0.0, "uniformity": 0.0}
    
    area_3d = area_3d[valid_mask]
    area_uv = area_uv[valid_mask]

    total_pixels = resolution * resolution
    densities_sq = (total_pixels * area_uv) / area_3d

    densities = np.sqrt(np.clip(densities_sq, 0, None))
    
    avg_density = np.average(densities, weights=area_3d)
    std_density = np.sqrt(np.average((densities - avg_density)**2, weights=area_3d))

    uniformity = std_density / avg_density if avg_density > 1e-6 else 0.0
    
    return {
        "avg_density": float(avg_density),
        "uniformity": float(uniformity)
    }

def compute_texel_density_from_path(path: str, resolution: int = 1024) -> Dict[str, float]:
    try:
        obj = load_mesh_raw(path)
        if isinstance(obj, trimesh.Scene):
            all_densities = []
            all_areas = []
            all_uniformities = []
            for g in obj.geometry.values():
                if isinstance(g, trimesh.Trimesh):
                    res = compute_texel_density(g, resolution)
                    area = g.area
                    if area > 1e-12:
                        all_densities.append(res["avg_density"])
                        all_areas.append(area)
                        all_uniformities.append(res["uniformity"])
            
            if not all_areas:
                return {"avg_density": 0.0, "uniformity": 0.0}
                
            avg = np.average(all_densities, weights=all_areas)
            unif = np.average(all_uniformities, weights=all_areas)
            return {"avg_density": float(avg), "uniformity": float(unif)}
        else:
            return compute_texel_density(obj, resolution)
    except Exception as e:
        print(f"Texel Density error: {e}")
        return {"avg_density": 0.0, "uniformity": 0.0}
