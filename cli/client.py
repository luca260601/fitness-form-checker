import argparse
import requests
from pathlib import Path

API_URL = "http://localhost:8000/analyze_squat"

def main():
    parser = argparse.ArgumentParser(description="CLI für Squat-Analyse (MVP Stub)")
    parser.add_argument("--video", required=True, help="Pfad zur Videodatei (mp4/mov)")
    parser.add_argument("--height", type=float, default=None, help="Körpergröße in cm")
    parser.add_argument("--fps", type=int, default=None, help="Ziel-FPS (optional)")
    args = parser.parse_args()

    video_path = Path(args.video)
    if not video_path.exists():
        raise SystemExit(f"Datei nicht gefunden: {video_path}")

    files = {"file": (video_path.name, open(video_path, "rb"), "video/mp4")}
    data = {}
    if args.height is not None:
        data["athlete_height_cm"] = str(args.height)
    if args.fps is not None:
        data["fps"] = str(args.fps)

    print("Sende Video zur Analyse ...")
    resp = requests.post(API_URL, files=files, data=data, timeout=60)
    resp.raise_for_status()
    result = resp.json()

    print("\nErgebnis (Stub):")
    print(f"- Winkel (Knie min/max): {result['angles']['knee_min']}° / {result['angles']['knee_max']}°")
    print(f"- Torso max: {result['angles']['torso_max']}°")
    print(f"- Flags: knee_over_toe_high={result['flags']['knee_over_toe_high']}, low_depth={result['flags']['low_depth']}, torso_lean_high={result['flags']['torso_lean_high']}")
    if result.get("notes"):
        print(f"- Hinweis: {result['notes']}")

if __name__ == "__main__":
    main()
