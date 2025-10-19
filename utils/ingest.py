import os, json, re
from typing import Dict, Any, Optional

from .parsing import slugify
from .file_ops import ensure_dir, save_text
from .openai_client import get_client

SCHEMA_JSON = {
    "type": "object",
    "properties": {
        "setup": {"type": "array", "items": {"type": "string"}},
        "execution": {"type": "array", "items": {"type": "string"}},
        "cues": {
            "type": "array",
            "items": {"type": "object", "properties": {
                "cue": {"type":"string"},
                "explanation": {"type":"string"}
            }, "required": ["cue"], "additionalProperties": False}
        },
        "common_errors": {"type": "array", "items": {"type": "string"}},
        "safety": {"type": "array", "items": {"type": "string"}},
        "references": {"type": "array", "items": {"type": "string"}}
    },
    "required": ["setup","execution","cues","common_errors","safety"],
    "additionalProperties": False
}

INSTRUCTIONS = (
    "Extrahiere einen Leitfaden mit Setup, Ausführung, Cues (mit Erklärung), "
    "Häufige Fehler, Sicherheit, optional Quellen. Antworte NUR mit JSON:\n\n"
    + json.dumps(SCHEMA_JSON, ensure_ascii=False, indent=2)
    + "\n\nBeginne mit '{' und ende mit '}'."
)

def _read_pdf_locally(pdf_path: str) -> str:
    from PyPDF2 import PdfReader
    reader = PdfReader(pdf_path)
    chunks = []
    for i, page in enumerate(reader.pages):
        try: text = page.extract_text() or ""
        except Exception: text = ""
        if text.strip(): chunks.append(f"[Seite {i+1}]\n{text.strip()}")
    return "\n\n".join(chunks)

def _ocr_pdf(pdf_path: str, lang: str = "deu+eng") -> str:
    from pdf2image import convert_from_path
    import pytesseract
    pages = convert_from_path(pdf_path, dpi=200)
    texts = []
    for idx, img in enumerate(pages):
        if img.mode != "L": img = img.convert("L")
        txt = pytesseract.image_to_string(img, lang=lang)
        if txt.strip(): texts.append(f"[Seite {idx+1} OCR]\n{txt.strip()}")
    return "\n\n".join(texts)

def _extract_json_loose(s: str) -> Optional[Dict[str, Any]]:
    if not s: return None
    m = re.search(r"```json\s*(\{.*?\})\s*```", s, re.DOTALL); 
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    m = re.search(r"```\s*(\{.*?\})\s*```", s, re.DOTALL); 
    if m:
        try: return json.loads(m.group(1))
        except Exception: pass
    start = s.find("{")
    while start != -1:
        lvl = 0
        for i, ch in enumerate(s[start:], start=start):
            if ch == "{": lvl += 1
            elif ch == "}":
                lvl -= 1
                if lvl == 0:
                    try: return json.loads(s[start:i+1])
                    except Exception: break
        start = s.find("{", start+1)
    cleaned = s.strip().strip("` \n\r\t")
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)
    try: return json.loads(cleaned)
    except Exception: return None

def _json_to_markdown(exercise: str, data: Dict[str, Any]) -> str:
    lines = [f"# {exercise} – Kurzleitfaden\n"]
    if sec := data.get("setup"):         lines += ["## Setup", *[f"- {x}" for x in sec]]
    if sec := data.get("execution"):     lines += ["\n## Ausführung", *[f"- {x}" for x in sec]]
    if sec := data.get("cues"):
        lines.append("\n## Cues")
        for x in sec:
            cue = (x.get("cue") or "").strip()
            exp = (x.get("explanation") or "").strip()
            lines.append(f"- **{cue}**: {exp}" if exp else f"- **{cue}**")
    if sec := data.get("common_errors"): lines += ["\n## Häufige Fehler", *[f"- {x}" for x in sec]]
    if sec := data.get("safety"):        lines += ["\n## Sicherheit", *[f"- {x}" for x in sec]]
    if sec := data.get("references"):    lines += ["\n## Quellen", *[f"- {x}" for x in sec]]
    return "\n".join(lines).strip() + "\n"

def extract_knowledge_from_pdf(base_dir: str, exercise_name: str, pdf_path: str) -> str:
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)
    client = get_client()
    slug = slugify(exercise_name)
    kdir = os.path.join(base_dir, "knowledge", "exercises")
    ensure_dir(kdir)

    system_msg = "Gib ausschließlich JSON im vorgegebenen Schema zurück."
    user_msg = f"Übung: {exercise_name}\n\n{INSTRUCTIONS}"

    # 1) zuerst File Search versuchen (kompatibel, ohne response_format)
    data = None
    try:
        vs = client.vector_stores.create(name=f"ex_{slug}")
        try:
            fobj = client.files.create(file=open(pdf_path, "rb"), purpose="assistants")
            client.vector_stores.files.create(vector_store_id=vs.id, file_id=fobj.id)
        except Exception:
            with open(pdf_path, "rb") as fh:
                client.vector_stores.file_batches.upload_and_poll(vector_store_id=vs.id, files=[fh])

        resp = client.responses.create(
            model="gpt-4o-mini",
            tools=[{"type":"file_search","vector_store_ids":[vs.id]}],
            input=[
                {"role":"system","content": system_msg},
                {"role":"user","content":[{"type":"input_text","text": user_msg}]}
            ],
            temperature=0.0
        )
        raw = ""
        for it in getattr(resp, "output", []):
            if getattr(it, "type", "") == "message":
                for c in it.content:
                    if c.type == "output_text": raw += c.text
        data = _extract_json_loose(raw)
    except Exception:
        data = None

    # 2) Fallback: lokal lesen / OCR → Prompt
    if data is None:
        text = _read_pdf_locally(pdf_path)
        if len(text.strip()) < 300:
            text = _ocr_pdf(pdf_path)
        resp = client.responses.create(
            model="gpt-4o-mini",
            input=[
                {"role":"system","content": system_msg},
                {"role":"user","content":[
                    {"type":"input_text","text": INSTRUCTIONS},
                    {"type":"input_text","text": "=== BEGINN PDF TEXT ===\n"+text+"\n=== ENDE PDF TEXT ==="}
                ]}
            ],
            temperature=0.0
        )
        raw = ""
        for it in getattr(resp, "output", []):
            if getattr(it, "type", "") == "message":
                for c in it.content:
                    if c.type == "output_text": raw += c.text
        data = _extract_json_loose(raw)

    if not isinstance(data, dict):
        raise RuntimeError("Konnte kein gültiges JSON aus dem PDF erzeugen.")

    md = _json_to_markdown(exercise_name, data)
    path = os.path.join(kdir, f"{slug}.md")
    save_text(kdir, f"{slug}.md", md)
    return path
