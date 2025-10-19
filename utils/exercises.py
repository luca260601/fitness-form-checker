import os, glob, yaml
from typing import Any, Dict, List, Optional
from .parsing import slugify

def config_dir(base_dir: str) -> str:
    return os.path.join(base_dir, "data", "exercises")

def load_all_configs(base_dir: str) -> List[Dict[str, Any]]:
    cfgs = []
    for p in glob.glob(os.path.join(config_dir(base_dir), "*.yaml")):
        with open(p, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            cfg["__path__"] = p
            cfgs.append(cfg)
    return cfgs

def find_config(base_dir: str, exercise_name: str) -> Optional[Dict[str, Any]]:
    ex = (exercise_name or "").strip().lower()
    for cfg in load_all_configs(base_dir):
        names = [str(cfg.get("name","")).lower(), *(a.lower() for a in cfg.get("aliases", []))]
        if ex in names: return cfg
    yml = os.path.join(config_dir(base_dir), f"{slugify(exercise_name)}.yaml")
    if os.path.exists(yml):
        with open(yml, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
            cfg["name"] = cfg.get("name") or exercise_name
            cfg["__path__"] = yml
            return cfg
    return None

def ensure_stub_config(base_dir: str, exercise_name: str) -> str:
    os.makedirs(config_dir(base_dir), exist_ok=True)
    slug = slugify(exercise_name)
    path = os.path.join(config_dir(base_dir), f"{slug}.yaml")
    if os.path.exists(path): return path
    # simple Heuristik
    ex = exercise_name.lower()
    if any(k in ex for k in ["curl", "bizeps", "biceps"]):
        cfg = {
            "name": exercise_name, "aliases": [], "targets": ["elbow"],
            "segments": [["shoulder","elbow"],["elbow","wrist"]],
            "angles": [{"id":"elbow","label":"Ellbogen","points":["shoulder","elbow","wrist"]}],
            "overlays": {"arrows_at": ["elbow"]},
            "moments": {"elbow": "per_arm_N * 0.03 * sin_deg(180 - angles.elbow)"}
        }
    else:
        cfg = {
            "name": exercise_name, "aliases": [], "targets": ["knee","hip"],
            "segments": [["shoulder","hip"],["hip","knee"],["knee","ankle"]],
            "angles": [
                {"id":"knee","label":"Knie","points":["hip","knee","ankle"]},
                {"id":"hip","label":"Hüfte","points":["shoulder","hip","knee"]}
            ],
            "overlays": {"arrows_at": ["knee","hip"]},
            "moments": {
                "knee": "total_N * 0.05 * sin_deg(180 - angles.knee)",
                "hip":  "total_N * 0.06 * sin_deg(180 - angles.hip)"
            }
        }
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path
