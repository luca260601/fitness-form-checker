import os, json, datetime
from typing import Dict, Any

def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)

def make_session_dir(base: str, name: str, exercise: str) -> str:
    ts = datetime.datetime.now().strftime("%d%m%Y_%H%M")
    path = os.path.join(base, "sessions", f"{name}_{exercise}_{ts}")
    ensure_dir(path)
    return path

def save_text(path: str, filename: str, content: str) -> str:
    ensure_dir(path)
    fpath = os.path.join(path, filename)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(content)
    return fpath

def save_json(path: str, filename: str, data: Dict[str, Any]) -> str:
    ensure_dir(path)
    fpath = os.path.join(path, filename)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return fpath
