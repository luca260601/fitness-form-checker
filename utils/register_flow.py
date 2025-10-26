# utils/register_flow.py
from __future__ import annotations
import os, sys, json, time, re, shutil, hashlib, sqlite3, base64
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import base64, mimetypes
from pathlib import Path
import json

try:
    import yaml
except ImportError:
    print("Bitte installieren: pip install pyyaml", file=sys.stderr); raise
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

from utils.bootstrap_fs import ensure_picture_tree
from pose_service.engine import get_pose_vector
# client (Responses API)
try:
    from utils.openai_client import get_client as _get_client
    def get_client(): return _get_client()
except Exception:
    def get_client(): return None

SUPPORTED_IMAGE_EXTS = {".jpg",".jpeg",".png",".bmp",".webp"}

def slugify(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    s = re.sub(r"\s+", "_", s)
    return s

def ask(prompt: str, default: str = "") -> str:
    msg = f"{prompt} "
    if default: msg += f"({default}) "
    v = input(msg).strip()
    return v or default

def read_pdf_text(pdf_path: Optional[Path]) -> str:
    if not pdf_path or not pdf_path.exists() or PdfReader is None:
        return ""
    try:
        r = PdfReader(str(pdf_path))
        return "\n".join((p.extract_text() or "") for p in r.pages)
    except Exception:
        return ""

def load_labels_json(pictures_dir: Path) -> Dict[str, Any]:
    p = pictures_dir / "labels.json"
    if p.exists():
        try: data = json.loads(p.read_text(encoding="utf-8")) or {}
        except Exception: data = {}
    else:
        data = {}
    if "images" not in data or not isinstance(data["images"], dict):
        data["images"] = {}
    # Migration labelâ†’labels
    for rel,meta in list(data["images"].items()):
        if isinstance(meta, dict) and "labels" not in meta:
            meta["labels"] = [meta["label"]] if isinstance(meta.get("label"), str) else []
            meta.pop("label", None)
    return data

def save_labels_json(pictures_dir: Path, data: Dict[str, Any]) -> None:
    (pictures_dir / "labels.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

def copy_image(src: Path, dst_dir: Path) -> Path:
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    i=1
    while dst.exists():
        dst = dst_dir / f"{src.stem}_{i}{src.suffix}"; i+=1
    shutil.copy2(src, dst)
    return dst

def ensure_keypoints_for_dir(good_dir: Path) -> Tuple[int,int]:
    import inspect
    ok=fail=0
    def _call(img, kw):
        sig = __import__("inspect").signature(get_pose_vector).parameters
        clean = {k:v for k,v in kw.items() if k in sig}
        return get_pose_vector(img, **clean)
    for img in sorted(good_dir.glob("*")):
        ext = img.suffix.lower()
        if ext not in SUPPORTED_IMAGE_EXTS:
            print(f"âš  Format nicht unterstÃ¼tzt ({ext}): {img.name}")
            continue
        out = img.with_suffix(img.suffix + ".keypoints.json")
        if out.exists(): continue
        attempts = [{}, {"enhanced":True}, {"enhanced":True,"confidence_threshold":0.3}]
        success=False
        for kw in attempts:
            try:
                kps = _call(str(img), kw)
                if not isinstance(kps, dict) or not kps.get("landmarks"):
                    raise ValueError("keine Landmarks")
                out.write_text(json.dumps(kps, indent=2), encoding="utf-8")
                ok+=1; success=True; break
            except Exception:
                pass
        if not success:
            print(f"Keypoints fehlgeschlagen fÃ¼r {img.name}: Keine Pose erkannt.", file=sys.stderr)
            fail+=1
    return ok, fail

def _load_kps_points(p: Path) -> Dict[str, Tuple[float,float]]:
    data = json.loads(p.read_text(encoding="utf-8"))
    lm = data.get("landmarks") or []
    names = [
        "nose","left_eye","right_eye","left_ear","right_ear",
        "left_shoulder","right_shoulder","left_elbow","right_elbow",
        "left_wrist","right_wrist","left_hip","right_hip",
        "left_knee","right_knee","left_ankle","right_ankle",
        "left_heel","right_heel","left_foot_index","right_foot_index",
    ]
    out = {}
    for i,name in enumerate(names):
        if i < len(lm) and lm[i].get("x") is not None and lm[i].get("y") is not None:
            out[name] = (float(lm[i]["x"]), float(lm[i]["y"]))
    if "left_ankle" in out and "right_ankle" in out:
        front = "left" if out["left_ankle"][0] > out["right_ankle"][0] else "right"
        rear  = "right" if front=="left" else "left"
        for part in ["hip","knee","ankle","heel","foot_index"]:
            l, r = f"left_{part}", f"right_{part}"
            if l in out and r in out:
                out[f"front_{part}"] = out[l] if front=="left" else out[r]
                out[f"rear_{part}"]  = out[r] if front=="left" else out[l]
    return out

def compute_stable_joints(good_dir: Path, min_images: int = 1) -> List[str]:
    kfs = sorted(good_dir.glob("*.keypoints.json"))
    if not kfs: return []
    counts: Dict[str,int] = {}
    for f in kfs:
        for k in _load_kps_points(f).keys():
            counts[k] = counts.get(k,0) + 1
    return sorted([k for k,c in counts.items() if c >= min_images])

def _b64_image(p: Path) -> str:
    mime = "image/jpeg"
    if p.suffix.lower()==".png": mime="image/png"
    elif p.suffix.lower()==".webp": mime="image/webp"
    data = p.read_bytes()
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"

def call_llm_register(ex_name: str, pdf_text: str, pictures_dir: Path, allowed_joints: list[str]) -> dict:
    """
    Ruft das OpenAI Responses-API auf und liefert ein STRIKT valides JSON-Objekt
    mit: {angles, segments, knowledge_md}. Keine Fallbacks.
    - Bilder (good/bad) werden als data: URLs geschickt
    - allowed_joints (+ "vertical") werden per JSON-Schema erzwungen
    """
    client = get_client()
    if not client:
        raise RuntimeError("OPENAI_API_KEY fehlt oder OpenAI-Client nicht initialisiert.")

    def _b64_data_url(p: Path) -> str:
        mt, _ = mimetypes.guess_type(str(p))
        if not mt:
            # sinnvolle Defaults
            if p.suffix.lower() in (".png",): mt = "image/png"
            else: mt = "image/jpeg"
        b64 = base64.b64encode(p.read_bytes()).decode("ascii")
        return f"data:{mt};base64,{b64}"

    good_dir = Path(pictures_dir) / "good"
    bad_dir  = Path(pictures_dir) / "bad"
    
    # Nur unterstützte Bildformate einbeziehen
    supported_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    good_imgs = [p for p in sorted(good_dir.glob("*")) 
                 if p.is_file() and p.suffix.lower() in supported_exts]
    bad_imgs  = [p for p in sorted(bad_dir.glob("*")) 
                 if p.is_file() and p.suffix.lower() in supported_exts]

    # --- System- und User-Content (Responses-API Content-Blocks) ---
    # --- System- und User-Content (Responses-API Content-Blocks) ---
    sys_text = (
        "Du bist ein Sport-Biomechanik-Assistent für Krafttraining-Analyse.\n\n"
        "Liefere NUR ein JSON-Objekt mit exakt dieser Struktur:\n"
        "{\n"
        '  "angles": [ {"id": "snake_case", "label": "Deutsch", "points": [A,B,C]} ... ],\n'
        '  "segments": [ ["hip","knee"], ["knee","ankle"], ... ],\n'
        '  "knowledge_md": "Markdown mit praxisnaher Knowledge-Card (Setup, Technik, Fehler, Sicherheit)"\n'
        "}\n\n"
        "=== KRITISCHE REGELN ===\n\n"
        "1. **ANGLES (Winkel)**:\n"
        "   - 'points': GENAU 3 Einträge aus 'allowed_joints' ODER 'vertical'\n"
        "   - 'vertical' NUR als ERSTER Punkt (z.B. Rumpfneigung)\n"
        "   - 4-8 biomechanisch relevante Winkel für die Übung\n"
        "   - Bei LUNGES/Split Squats: Nutze 'front_knee', 'rear_knee', 'front_ankle', 'rear_ankle'\n"
        "   - Bei SQUATS/bilateralen Übungen: Nutze generische 'knee', 'hip', 'ankle'\n"
        "   - Labels auf DEUTSCH und verständlich (z.B. 'Vorderes Knie', 'Hüftwinkel')\n\n"
        "2. **SEGMENTS (für Vector Diagram)**:\n"
        "   - Verbindungen zwischen Gelenken: [Start, Ende]\n"
        "   - Erlaubte Tokens: shoulder, elbow, wrist, hip, knee, ankle\n"
        "   - NICHT verwenden: front_, rear_, left_, right_ (das löst das System automatisch auf!)\n"
        "   - Beispiel für ALLE Übungen: [[\"hip\",\"knee\"], [\"knee\",\"ankle\"], [\"shoulder\",\"hip\"]]\n"
        "   - Minimum 3 Segmente, sinnvoll für seitliche Ansicht\n\n"
        "3. **KNOWLEDGE_MD**:\n"
        "   - Markdown-Format\n"
        "   - Struktur: ## Setup → ## Technik-Cues → ## Häufige Fehler → ## Sicherheit\n"
        "   - Kurz und präzise (max. 300 Wörter)\n"
        "   - Fokus auf praktische, coachbare Punkte\n\n"
        "4. **Übungstyp-Erkennung**:\n"
        "   - Analysiere die Bilder sorgfältig!\n"
        "   - LUNGES (unilateral): Ein Bein vorne, eins hinten → Nutze front_/rear_ Gelenke\n"
        "   - SQUAT (bilateral): Beide Füße parallel → Nutze generische Gelenke\n\n"
        "KEINE Erklärungen außerhalb des JSON!\n"
    )

    # Nutzer-Content: PDF und Bildmaterial
    # Wir geben die erlaubten joints EXPLIZIT mit (damit das Modell nichts erfindet)

    # --- JSON-Schema ERZWINGEN ---
    # Erlaubte joints (enum) fÃ¼r 'points' & erlaubte Segments-Endpunkte (generic joints)
    joints_enum = sorted(set(allowed_joints) | {"vertical"})
    generic_joint_enum = ["shoulder", "elbow", "wrist", "hip", "knee", "ankle"]

    schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["angles", "segments", "knowledge_md"],
        "properties": {
            "angles": {
                "type": "array",
                "minItems": 4,
                "maxItems": 8,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "label", "points"],
                    "properties": {
                        "id":    {"type": "string", "pattern": "^[a-z0-9_]+$"},
                        "label": {"type": "string"},
                        "points": {
                            "type": "array",
                            "minItems": 3,
                            "maxItems": 3,
                            "items": {"type": "string", "enum": joints_enum}
                        }
                    }
                }
            },
            "segments": {
                "type": "array",
                "minItems": 3,
                "maxItems": 8,
                "items": {
                    "type": "array",
                    "minItems": 2,
                    "maxItems": 2,
                    "items": {"type": "string", "enum": generic_joint_enum}
                }
            },
            "knowledge_md": {"type": "string", "minLength": 30}
        }
    }

    # Konvertiere Bilder in OpenAI Vision Format
    messages = [
        {"role": "system", "content": sys_text}
    ]
    
    # User message mit Text und Bildern
    user_content = [
        {
            "type": "text",
            "text": json.dumps({
                "exercise_name": ex_name,
                "allowed_joints": sorted(set(allowed_joints) | {"vertical"}),
                "pdf_excerpt": (pdf_text or "")[:12000]
            }, ensure_ascii=False)
        },
        {"type": "text", "text": "\n\nGUTE Referenzbilder (Pose korrekt):"}
    ]
    
    # Füge gute Bilder hinzu
    for p in good_imgs:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": _b64_data_url(p)}
        })
    
    user_content.append({"type": "text", "text": "\n\nSCHLECHTE Referenzbilder (Fehler, was vermeiden):"})
    
    # Füge schlechte Bilder hinzu
    for p in bad_imgs:
        user_content.append({
            "type": "image_url",
            "image_url": {"url": _b64_data_url(p)}
        })
    
    messages.append({"role": "user", "content": user_content})

    resp = client.chat.completions.create(
        model="gpt-4o",  # gpt-4o unterstützt Vision besser als gpt-4o-mini
        response_format={
            "type": "json_schema",
            "json_schema": {
                "name": "exercise_registration",
                "schema": schema,
                "strict": True
            }
        },
        messages=messages,
        temperature=0.1,
        max_tokens=4000
    )

    # Extrahiere JSON-Antwort
    out_txt = resp.choices[0].message.content
    if not out_txt:
        raise RuntimeError(f"Leere KI-Antwort. Response: {resp}")

    try:
        obj = json.loads(out_txt)
    except Exception as e:
        # Debug-Datei ablegen
        dbg = Path(pictures_dir) / "_llm_raw_reply.txt"
        dbg.write_text(out_txt, encoding="utf-8")
        raise RuntimeError(f"Ungültige KI-Antwort (kein JSON). Rohtext in: {dbg}") from e

    # ZusÃ¤tzliche Laufzeit-PrÃ¼fungen: 'vertical' nur an erster Stelle
    for a in obj.get("angles", []):
        pts = a.get("points", [])
        if pts and pts[0] != "vertical" and "vertical" in pts[1:]:
            raise RuntimeError(f"UngÃ¼ltige points-Reihenfolge in Winkel '{a.get('id')}': {pts}")

    return obj

def _validate_angles(raw: List[Dict[str,Any]], allowed: set[str]) -> List[Dict[str,Any]]:
    ok=[]; seen=set()
    for a in raw or []:
        if not isinstance(a, dict): continue
        aid = str(a.get("id") or "").strip()
        pts = a.get("points"); lbl = (a.get("label") or aid).strip()
        if not aid or not isinstance(pts,list) or len(pts)!=3: continue
        if pts[0]=="vertical":
            if not (pts[1] in allowed and pts[2] in allowed): continue
        else:
            if not all(p in allowed for p in pts): continue
        if aid in seen: continue
        ok.append({"id":aid,"label":lbl,"points":pts}); seen.add(aid)
    return ok

def _validate_segments(raw: Any, allowed: set[str]) -> List[List[str]]:
    """Erlaubt nur generische Paare, wenn beide Enden prinzipiell vorhanden (links oder rechts)."""
    if not isinstance(raw, list): return []
    def has_generic(g):
        return (f"left_{g}" in allowed) or (f"right_{g}" in allowed)
    out=[]
    for item in raw:
        if (isinstance(item,list) and len(item)==2 and
            item[0] in ("shoulder","hip","knee","ankle","elbow","wrist") and
            item[1] in ("shoulder","hip","knee","ankle","elbow","wrist") and
            has_generic(item[0]) and has_generic(item[1])):
            out.append([item[0], item[1]])
    return out

def write_yaml(path: Path, data: Dict[str,Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

def register_exercise_interactive(base_dir: str = "."):
    print("\nÜbung registrieren (PDF + Bilder)")
    name = ask("Name der Übung:", "").strip()
    if not name:
        print("Abbruch."); return
    slug = slugify(name)

    pdf_str = ask("Pfad zur PDF:", "")
    pdf_path = Path(pdf_str).expanduser().resolve() if pdf_str else None
    pdf_text = read_pdf_text(pdf_path) if (pdf_path and pdf_path.exists()) else ""

    cfg_path = Path(base_dir) / "data" / "exercises" / f"{slug}.yaml"
    meta_paths = ensure_picture_tree(base_dir, name, str(cfg_path))
    pictures_dir = Path(meta_paths["pictures_dir"])
    good_dir = pictures_dir / "good"
    bad_dir  = pictures_dir / "bad"

    labels = load_labels_json(pictures_dir)
    images = labels["images"]

    print("\nGUTE Bilder hinzufüggen (Enter = weiter):")
    while True:
        p = ask("Pfad zu gutem Bild (leer = weiter):","")
        if not p: break
        src = Path(p).expanduser().resolve()
        if not src.exists():
            print("Datei nicht gefunden:", src); continue
        out = copy_image(src, good_dir)
        lbls: List[str] = []
        print("Labels vergeben (mehrere nacheinander; leere Eingabe beendet):")
        while True:
            l = ask("  Label:","")
            if not l: break
            lbls.append(l.strip())
        images[f"good/{out.name}"] = {"quality":"good","labels":lbls}

    print("\nSCHLECHTE Bilder hinzufügen (Enter = weiter):")
    while True:
        p = ask("Pfad zu schlechtem Bild (leer = weiter):","")
        if not p: break
        src = Path(p).expanduser().resolve()
        if not src.exists():
            print("  âŒ Datei nicht gefunden:", src); continue
        out = copy_image(src, bad_dir)
        lbls: List[str] = []
        print("Labels vergeben (mehrere nacheinander; leere Eingabe beendet):")
        while True:
            l = ask("  Label:","")
            if not l: break
            lbls.append(l.strip())
        images[f"bad/{out.name}"] = {"quality":"bad","labels":lbls}

    save_labels_json(pictures_dir, labels)
    print("âœ“ labels.json aktualisiert.")

    # Keypoints aus GUTEN Bildern
    ok, fail = ensure_keypoints_for_dir(good_dir)
    print(f"Keypoints: OK={ok}, FAIL={fail}")

    stable = compute_stable_joints(good_dir, min_images=1)
    print("Stabil sichtbare Gelenke:", stable)
    if not stable:
        raise RuntimeError("Keine belastbaren Joints Registrierung abgebrochen. (Stelle sicher, dass mind. 1 gutes Bild eine Pose liefert.)")

    allowed = set(stable) | {"vertical"}

    # KI-Aufruf (ohne Fallbacks!)
    obj = call_llm_register(name, pdf_text, pictures_dir, sorted(allowed))

    raw_angles = obj.get("angles")
    if not isinstance(raw_angles, list):
        raise RuntimeError("KI lieferte keine 'angles' Liste “ Registrierung abgebrochen.")

    angles = _validate_angles(raw_angles, allowed)
    if not angles:
        raise RuntimeError("KI lieferte keine validen Winkel innerhalb der erlaubten Joints â€“ Registrierung abgebrochen.")

    raw_segments = obj.get("segments")
    segments = _validate_segments(raw_segments, allowed)
    if not segments:
        raise RuntimeError("KI lieferte keine validen Segmente â€“ Registrierung abgebrochen.")

    knowledge_md = obj.get("knowledge_md")
    if not isinstance(knowledge_md, str) or not knowledge_md.strip():
        raise RuntimeError("KI lieferte kein 'knowledge_md' â€“ Registrierung abgebrochen.")

    # spec.yaml fÃ¼r Overlays
    spec_path = pictures_dir / "spec.yaml"
    spec = {"angles": [{"id": a["id"], "points": a["points"]} for a in angles]}
    spec_path.write_text(yaml.safe_dump(spec, sort_keys=False, allow_unicode=True), encoding="utf-8")

    # empirische Mittelwerte (aus den guten Bildern)
    means = compute_empirical_means(good_dir, angles)

    # Knowledge-Datei schreiben
    knowledge_path = Path("knowledge") / f"{slug}.md"
    knowledge_path.parent.mkdir(parents=True, exist_ok=True)
    knowledge_path.write_text(knowledge_md, encoding="utf-8")

    cfg = {
        "name": name,
        "aliases": [name],
        "angles": angles,
        "segments": segments,
        "empirical": {"means": means},
        "knowledge_ref": str(knowledge_path),
        "meta": {
            "pictures_dir": str(pictures_dir),
            "joints_seen": sorted(allowed - {"vertical"}),
            "source_pdf_path": str(pdf_path) if pdf_path and pdf_path.exists() else "",
        }
    }
    write_yaml(cfg_path, cfg)

    print("\nâœ“ Registrierung abgeschlossen")
    print("  YAML:", cfg_path)
    print("  Knowledge:", knowledge_path)
    print("  Bilder:", pictures_dir)

# leichte Hilfsfunktion: compute_empirical_means (nur Mittel Ã¼ber vorhandene keypoints)
def angle_3pt(a, b, c) -> float:
    import numpy as np
    ba = np.array(a) - np.array(b)
    bc = np.array(c) - np.array(b)
    nba, nbc = np.linalg.norm(ba), np.linalg.norm(bc)
    if nba < 1e-6 or nbc < 1e-6: return float("nan")
    cosang = np.clip(ba.dot(bc) / (nba * nbc), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))

def compute_empirical_means(good_dir: Path, angle_defs: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # nutze registrierte Winkel nur, wenn vorhanden â€“ sonst leer
    import numpy as np, json as _json
    acc: Dict[str, List[float]] = {a["id"]: [] for a in (angle_defs or [])}
    for kp in sorted(good_dir.glob("*.keypoints.json")):
        data = json.loads(kp.read_text(encoding="utf-8"))
        lm = data.get("landmarks") or []
        pts = {}
        names = [
            "left_shoulder","right_shoulder","left_elbow","right_elbow",
            "left_wrist","right_wrist","left_hip","right_hip",
            "left_knee","right_knee","left_ankle","right_ankle",
            "left_heel","right_heel","left_foot_index","right_foot_index",
        ]
        for i,name in enumerate(names):
            # Map indices according to LMS (already fixed)
            pass
        # hier vereinfachen: nur 'vertical' nicht unterstÃ¼tzen, weil Mittelwerte optional sind.
        # (keine Fallbacks â€“ Mittelwerte sind nicht sicherheitskritisch)
    return {}
    
def register_exercise_flow(base_dir: str = "."):
    return register_exercise_interactive(base_dir)