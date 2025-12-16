from typing import Dict, Tuple

Thresholds = Dict[str, Dict[str, Dict[str, float]]]

THRESHOLDS: Thresholds = {
    "low-poly": {
        "background": {"max_faces": 150, "max_density": 250.0},
        "prop": {"max_faces": 400, "max_density": 450.0},
        "hero": {"max_faces": 750, "max_density": 650.0},
    },
    "indie": {
        "background": {"max_faces": 300, "max_density": 500.0},
        "prop": {"max_faces": 800, "max_density": 900.0},
        "hero": {"max_faces": 1500, "max_density": 1300.0},
    },
    "aa": {
        "background": {"max_faces": 600, "max_density": 1000.0},
        "prop": {"max_faces": 1600, "max_density": 1800.0},
        "hero": {"max_faces": 3000, "max_density": 2600.0},
    },
    "aaa": {
        "background": {"max_faces": 1200, "max_density": 2000.0},
        "prop": {"max_faces": 3200, "max_density": 3600.0},
        "hero": {"max_faces": 6000, "max_density": 5200.0},
    },
    "cinematic": {
        "background": {"max_faces": 2400, "max_density": 4000.0},
        "prop": {"max_faces": 6400, "max_density": 7200.0},
        "hero": {"max_faces": 12000, "max_density": 10400.0},
    },
}

def get_thresholds(game_type: str, usage_area: str) -> Tuple[int, float]:
    gt = THRESHOLDS.get(game_type)
    if not gt:
        raise KeyError("game_type")
    ua = gt.get(usage_area)
    if not ua:
        raise KeyError("usage_area")
    return int(ua["max_faces"]), float(ua["max_density"])
