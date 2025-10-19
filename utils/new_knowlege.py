import os
from .parsing import slugify

DEFAULT_PROMPT = """Du bist ein präziser, freundlicher Krafttrainings-Coach.
Gib konkretes, priorisiertes Technik-Feedback in kurzen Stichpunkten.
Sage, was gut ist, was zu verbessern ist, und nenne 2–4 konkrete Cues.
Arbeite mit den gegebenen Winkeln/Momenten und Kurznotizen (Wissen).
Gib am Ende eine vorsichtige Belastungseinordnung ("nur grobe Schätzung").
Sprache: Deutsch. Vermeide Fachchinesisch.
"""

def read_knowledge(exercise: str | None, base_dir: str) -> str:
    kb = []
    common = os.path.join(base_dir, "knowledge", "_common.md")
    if os.path.exists(common):
        with open(common, "r", encoding="utf-8") as f:
            kb.append(f"# _common.md\n{f.read()}")
    if exercise:
        slug = slugify(exercise)
        p = os.path.join(base_dir, "knowledge", "exercises", f"{slug}.md")
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                kb.append(f"# {slug}.md\n{f.read()}")
    return "\n\n".join(kb)

def read_system_prompt(base_dir: str) -> str:
    p = os.path.join(base_dir, "instructions", "system_prompt.txt")
    if not os.path.exists(p):
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f: f.write(DEFAULT_PROMPT)
        return DEFAULT_PROMPT
    with open(p, "r", encoding="utf-8") as f:
        txt = f.read().strip()
    return txt or DEFAULT_PROMPT
