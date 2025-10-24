# utils/bootstrap_fs.py
import os, yaml
import sys
import subprocess
from pathlib import Path

def ensure_picture_tree(base_dir: str, exercise_name: str, cfg_path: str) -> dict:
    """
    Legt die Bilderstruktur an und erzeugt labels.json + spec.yaml falls fehlend.
    Liest KEINE cfg, wenn sie noch nicht existiert.
    """
    from pathlib import Path
    import yaml, re

    def _slug(s: str) -> str:
        s = s.strip()
        s = re.sub(r"[\\/:*?\"<>|]+", "_", s)  # dateisystem-sicher
        return s

    base = Path(base_dir)
    pics_root = base / "pictures"
    ex_dir = pics_root / _slug(exercise_name)
    good_dir = ex_dir / "good"
    bad_dir  = ex_dir / "bad"
    ex_dir.mkdir(parents=True, exist_ok=True)
    good_dir.mkdir(exist_ok=True)
    bad_dir.mkdir(exist_ok=True)

    # labels.json
    labels_path = ex_dir / "labels.json"
    if not labels_path.exists():
        labels_path.write_text('{"images":{}}', encoding="utf-8")

    # spec.yaml: falls vorhanden, so lassen; sonst leer anlegen.
    spec_path = ex_dir / "spec.yaml"
    if not spec_path.exists():
        # Versuche ggf. Winkel aus bereits vorhandener cfg zu übernehmen – aber nur wenn sie existiert.
        angles = []
        cfg_p = Path(cfg_path)
        if cfg_p.exists():
            try:
                cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) or {}
                if isinstance(cfg.get("angles"), list):
                    for a in cfg["angles"]:
                        if isinstance(a, dict) and a.get("id") and a.get("points"):
                            angles.append({"id": a["id"], "points": a["points"]})
            except Exception:
                angles = []
        with spec_path.open("w", encoding="utf-8") as f:
            yaml.safe_dump({"angles": angles}, f, sort_keys=False, allow_unicode=True)

    # refdb_path: nicht erzwingen – wenn die cfg noch nicht existiert, dann leer zurückgeben
    refdb_path = ""
    cfg_p = Path(cfg_path)
    if cfg_p.exists():
        try:
            cfg = yaml.safe_load(cfg_p.read_text(encoding="utf-8")) or {}
            refdb_path = ((cfg.get("meta") or {}).get("refdb_path") or "")
        except Exception:
            refdb_path = ""

    return {
        "pictures_dir": str(ex_dir),
        "good_dir": str(good_dir),
        "bad_dir":  str(bad_dir),
        "labels_json": str(labels_path),
        "spec_yaml": str(spec_path),
        "refdb_path": refdb_path,
    }


def init_refdb_if_missing(base_dir: str, refdb_location: str | None = None) -> str:
    base = Path(base_dir)
    refdb = base / "ref.db"

    # Kandidaten in der Reihenfolge der Präferenz
    candidates = []
    if refdb_location:
        candidates.append(Path(refdb_location))
    candidates += [
        base / "refdb.py",
        base / "database" / "refdb.py",
        base / "tools" / "refdb.py",
    ]

    refdb_py = next((p for p in candidates if p.exists()), None)

    # DB anlegen, falls fehlt und Script vorhanden
    if not refdb.exists() and refdb_py:
        subprocess.run([sys.executable, str(refdb_py), "init", "--db", str(refdb)], check=False)

    return str(refdb)
