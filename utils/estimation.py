# utils/estimation.py
import os, json, re
from typing import Dict, Any, Optional
from .openai_client import get_client

# ---------- kleine Helfer ----------

def _extract_json_loose(s: str) -> Optional[dict]:
    """zieht JSON auch dann raus, wenn GPT Fences/Prosa drum herum hat"""
    if not s:
        return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r"```\s*(\{.*?\})\s*```", s, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    start = s.find("{")
    while start != -1:
        lvl = 0
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{": lvl += 1
            elif ch == "}":
                lvl -= 1
                if lvl == 0:
                    try: return json.loads(s[start:i+1])
                    except: break
        start = s.find("{", start+1)
    try:
        return json.loads(s.strip())
    except:
        return None

def _read_pdf_locally(pdf_path: str) -> str:
    """liest eingebetteten Text aus PDF (kein OCR)"""
    try:
        from PyPDF2 import PdfReader  # lazy import
    except Exception:
        return ""
    try:
        reader = PdfReader(pdf_path)
        chunks = []
        for i, page in enumerate(reader.pages):
            try:
                t = page.extract_text() or ""
            except Exception:
                t = ""
            if t.strip():
                chunks.append(f"[Seite {i+1}]\n{t.strip()}")
        return "\n\n".join(chunks)
    except Exception:
        return ""

def _ocr_pdf(pdf_path: str, lang: str = "deu+eng") -> str:
    """OCR-Fallback; gibt leeren String zurück, wenn pdf2image/tesseract fehlen"""
    try:
        from pdf2image import convert_from_path  # lazy import
        import pytesseract
    except Exception:
        return ""
    try:
        pages = convert_from_path(pdf_path, dpi=200)
        texts = []
        for idx, img in enumerate(pages):
            if img.mode != "L":
                img = img.convert("L")
            txt = pytesseract.image_to_string(img, lang=lang)
            if txt.strip():
                texts.append(f"[Seite {idx+1} OCR]\n{txt.strip()}")
        return "\n\n".join(texts)
    except Exception:
        return ""

# ---------- öffentliche Funktionen ----------

def estimate_angles_from_text(cfg: Dict[str, Any], description: str) -> Dict[str, float]:
    """
    Schätzt Winkel (0..180°) für die in cfg['angles'] definierten IDs aus einer kurzen Beschreibung.
    Ergebnis-Keys: <id>_deg  (+ "__side__")
    """
    client = get_client()
    wanted = [a["id"] for a in (cfg.get("angles") or []) if "id" in a]
    schema_hint = {aid: "Gradzahl (0-180)" for aid in wanted}

    sys = ("Du schätzt aus einer Übungs-/Posenbeschreibung die ungefähren Gelenkwinkel in Grad. "
           "Antworte NUR mit einem JSON-Objekt ohne Erklärtext, "
           f"mit den Keys {wanted}. Werte 0..180.")
    user = f"Beschreibung:\n{description}\n\nGib die Winkel als JSON (nur Zahlen):\n{json.dumps(schema_hint, ensure_ascii=False)}"

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role":"system","content": sys},
            {"role":"user","content":[{"type":"input_text","text": user}]}
        ],
        temperature=0.2
    )
    raw = ""
    for it in getattr(resp, "output", []):
        if getattr(it, "type", "") == "message":
            for c in it.content:
                if getattr(c, "type", "") == "output_text":
                    raw += c.text

    data = _extract_json_loose(raw) or {}
    out: Dict[str, float] = {}
    for aid in wanted:
        try:
            v = float(data.get(aid, 0))
            v = max(0.0, min(180.0, v))
        except Exception:
            v = 0.0
        out[f"{aid}_deg"] = float(round(v, 1))
    out["__side__"] = "LEFT"
    return out

def estimate_angles_from_pdf(cfg: Dict[str, Any], pdf_path: str) -> Dict[str, float]:
    """
    Liest PDF-Text (OCR-Fallback) und schätzt daraus die typischen Winkel (0..180°).
    Ergebnis-Keys: <id>_deg  (+ "__side__")
    """
    if not os.path.isfile(pdf_path):
        raise FileNotFoundError(pdf_path)

    text = _read_pdf_locally(pdf_path)
    if len(text.strip()) < 300:
        text = _ocr_pdf(pdf_path)

    client = get_client()
    wanted = [a["id"] for a in (cfg.get("angles") or []) if "id" in a]
    schema_hint = {aid: "Gradzahl (0-180)" for aid in wanted}

    sys = ("Du liest Trainingsunterlagen (PDF-Text) und schätzt die typischen Gelenkwinkel "
           "für die beschriebene Standard-Ausführung. Antworte NUR mit JSON, Keys = "
           f"{wanted}, Werte 0..180 (Grad). Keine Erklärungen.")
    user = (
        "=== BEGINN PDF TEXT ===\n" + text + "\n=== ENDE PDF TEXT ===\n\n"
        "Gib die Winkel als JSON (nur Zahlen):\n" + json.dumps(schema_hint, ensure_ascii=False)
    )

    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role":"system","content": sys},
            {"role":"user","content":[{"type":"input_text","text": user}]}
        ],
        temperature=0.2
    )
    raw = ""
    for it in getattr(resp, "output", []):
        if getattr(it, "type", "") == "message":
            for c in it.content:
                if getattr(c, "type", "") == "output_text":
                    raw += c.text

    data = _extract_json_loose(raw) or {}
    out: Dict[str, float] = {}
    for aid in wanted:
        try:
            v = float(data.get(aid, 0))
            v = max(0.0, min(180.0, v))
        except Exception:
            v = 0.0
        out[f"{aid}_deg"] = float(round(v, 1))
    out["__side__"] = "LEFT"
    return out
