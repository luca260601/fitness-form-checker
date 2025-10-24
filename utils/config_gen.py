import os, re, yaml
from typing import Any, Dict, Optional, List
from .parsing import slugify
from .file_ops import ensure_dir
from .openai_client import get_client

ALLOWED_POINTS = {"shoulder","elbow","wrist","hip","knee","ankle","hip_down"}
ALLOWED_TARGETS = {"knee","hip","shoulder","elbow"}

YAML_GUIDE = """\
name: <Name>
description: "<1-2 Sätze>"
aliases: [<...>]
targets: [knee, hip, shoulder, elbow]
segments: [[shoulder, hip], [hip, knee], [knee, ankle], [shoulder, elbow], [elbow, wrist]]

angles:
  - id: <id>
    label: <Label>
    points: [<A>, <B>, <C>]
    # Neu:
    description: "<kurz>"
    per_phase:
      bottom: [<lo>, <hi>]
      top: [<lo>, <hi>]

overlays:
  arrows_at: [knee, hip, elbow]

# Neu:
thresholds:
  knee_angle: { excellent: [80,110], good: [60,140], needs_work: [45,150] }

form_cues:
  excessive_forward_lean: "Halte den Oberkörper aufrechter."
  elbow_flare: "Ellenbogen näher am Körper führen."

safety_checks:
  min_knee_angle: 45
  max_trunk_lean: 45

meta:
  source_pdf_path: "<auto>"
  provenance: ["pdf", "inferred"]

"""

SYSTEM = "Erzeuge ausschließlich YAML (ohne Erklärtext)."
USER_TPL = (
    "Übungsname: {name}\n\n"
    "Erzeuge eine YAML-Konfiguration für die Visualisierung gemäß:\n"
    + YAML_GUIDE +
    "\nBeachte zulässige Punkte: shoulder, elbow, wrist, hip, knee, ankle, hip_down."
)

def _extract_yaml(s: str) -> Optional[str]:
    if not s: return None
    import re
    m = re.search(r"```yaml\s*(.+?)\s*```", s, re.DOTALL|re.I)
    if m: return m.group(1).strip()
    m = re.search(r"```\s*(.+?)\s*```", s, re.DOTALL)
    if m: return m.group(1).strip()
    m = re.search(r"(name\s*:\s*.+)", s, re.DOTALL|re.I)
    return m.group(1).strip() if m else None

def _validate(cfg: Dict[str, Any]) -> List[str]:
    errs = []
    for k in ["name","targets","segments","angles","overlays"]:
        if k not in cfg: errs.append(f"{k} fehlt")
    ts = set(t.lower() for t in cfg.get("targets", []))
    if not ts or not ts.issubset(ALLOWED_TARGETS): errs.append("targets ungültig")
    for seg in cfg.get("segments", []):
        if not (isinstance(seg,list) and len(seg)==2 and all(p in ALLOWED_POINTS for p in seg)):
            errs.append(f"Segment ungültig: {seg}")
    for a in cfg.get("angles", []):
        pts = a.get("points", [])
        if not (isinstance(pts,list) and len(pts)==3 and all(p in ALLOWED_POINTS for p in pts)):
            errs.append(f"Angle ungültig: {a}")
        if "id" not in a: errs.append("angle.id fehlt")
    arr = cfg.get("overlays", {}).get("arrows_at", [])
    if not (isinstance(arr,list) and set(arr).issubset(ALLOWED_TARGETS)):
        errs.append("overlays.arrows_at ungültig")
    return errs

def generate_config_from_pdf(base_dir: str, exercise_name: str, pdf_path: str) -> str:
    client = get_client()
    vs_id = None
    if os.path.exists(pdf_path):
        try:
            vs = client.vector_stores.create(name=f"cfg_{slugify(exercise_name)}")
            try:
                fobj = client.files.create(file=open(pdf_path,"rb"), purpose="assistants")
                client.vector_stores.files.create(vector_store_id=vs.id, file_id=fobj.id)
            except Exception:
                with open(pdf_path,"rb") as fh:
                    client.vector_stores.file_batches.upload_and_poll(vector_store_id=vs.id, files=[fh])
            vs_id = vs.id
        except Exception:
            vs_id = None

    kwargs = {
        "model":"gpt-4o-mini",
        "input":[
            {"role":"system","content": SYSTEM},
            {"role":"user","content":[{"type":"input_text","text": USER_TPL.format(name=exercise_name)}]}
        ],
        "temperature":0.1
    }
    if vs_id:
        kwargs["tools"] = [{"type":"file_search","vector_store_ids":[vs_id]}]

    resp = client.responses.create(**kwargs)
    text = ""
    for it in getattr(resp,"output",[]):
        if getattr(it,"type","")=="message":
            for c in it.content:
                if getattr(c,"type","")=="output_text": text += c.text
    y = _extract_yaml(text)
    if not y: raise RuntimeError("Konnte keine YAML extrahieren.")
    cfg = yaml.safe_load(y) or {}
    errs = _validate(cfg)
    if errs: raise RuntimeError("YAML-Validierung: " + "; ".join(errs))

    outdir = os.path.join(base_dir, "data", "exercises")
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{slugify(exercise_name)}.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path
