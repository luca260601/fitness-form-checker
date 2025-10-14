import argparse, requests
from pathlib import Path

API_URL = "http://127.0.0.1:8000/analyze"

def main():
    p = argparse.ArgumentParser(description="Multi-Exercise CLI (Squat MVP)")
    p.add_argument("--video", required=True)
    p.add_argument("--exercise", default="squat")
    p.add_argument("--fps", type=int, default=8)
    args = p.parse_args()

    vp = Path(args.video)
    if not vp.exists():
        raise SystemExit(f"Datei nicht gefunden: {vp}")

    files = {"file": (vp.name, open(vp, "rb"), "video/mp4")}
    data = {"exercise_hint": args.exercise, "fps": str(args.fps)}

    r = requests.post(API_URL, files=files, data=data, timeout=60)
    r.raise_for_status()
    res = r.json()
    

    print("\n--- Ergebnis ---")
    print("exercise:", res.get("exercise"))
    print("metrics:", res.get("metrics"))
    print("issues :", res.get("issues"))
    if res.get("notes"): print("notes  :", res["notes"])
    print("\nEmpfohlenes Feedback:")
    SEV_ORDER = {"high": 0, "warn": 1, "info": 2}
    issues_sorted = sorted(res.get("issues", []), key=lambda i: SEV_ORDER.get(i.get("severity","info"), 9))

    cues = []
    codes = {i["code"] for i in issues_sorted}

    if "knee_over_toe_high" in codes:
        cues.append("Knie wandern deutlich vor die Zehen → verlagere das Gewicht mehr auf die Fersen, starte die Bewegung mit der Hüfte nach hinten. Tempo kontrollieren.")

    if "low_depth" in codes:
        cues.append("Tiefe evtl. nicht unter parallel → atme ein & brace, Po aktiv nach hinten/unten führen. Box-Squats oder Tempo-Exzentrik als Drill.")

    if "torso_lean_high" in codes:
        cues.append("Starke Oberkörpervorlage → Brust stolz/Lat anspannen, Core aktiv. Sprunggelenks-Mobilität & Core-Stabilität prüfen.")

    if not cues:
        cues.append("Solide Basis! Achte weiter auf stabile Knieachse, gleichmäßigen Druck über den ganzen Fuß und kontrollierte Tiefe.")

    for c in cues:
        print(f"- {c}")

    

    

if __name__ == "__main__":
    main()
