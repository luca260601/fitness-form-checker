import re, unicodedata

def parse_kg(s: str) -> float:
    s0 = (s or "").strip().lower()
    if s0 in {"", "0", "ohne", "leer", "none", "keine"}:
        return 0.0
    if any(k in s0 for k in ["stange", "bar", "langhantel"]):
        m = re.search(r"(\d+[.,]?\d*)", s0)
        if m:
            return float(m.group(1).replace(",", "."))
        return 20.0
    s0 = s0.replace("kg", "").replace(",", ".")
    return float(s0)

def slugify(text: str) -> str:
    t = unicodedata.normalize("NFKD", text)
    t = t.encode("ascii", "ignore").decode("ascii")
    t = re.sub(r"[^a-zA-Z0-9]+", "-", t).strip("-").lower()
    return t or "x"
