import os, json, argparse
from dotenv import load_dotenv
from rich import print, box
from rich.table import Table
from rich.prompt import Prompt
from pydantic import BaseModel, Field
from typing import Dict

from utils.parsing import parse_kg, slugify
from utils.file_ops import make_session_dir, save_text, save_json
from utils.new_knowlege import read_knowledge, read_system_prompt
from utils.openai_client import get_client
from utils.config_gen import generate_config_from_pdf
from utils.exercises import find_config, load_all_configs
from utils.estimation import estimate_angles_from_text, estimate_angles_from_pdf

from pose_service.engine import get_pose_vector, compute_angles_config
from pose_service.combined_visualization import create_combined_analysis_image
from utils.bootstrap_fs import ensure_picture_tree, init_refdb_if_missing

from utils.bootstrap_fs import ensure_picture_tree
from utils.register_flow import register_exercise_flow
from database.ref_gallery import init_db
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
load_dotenv()
client = get_client()

# ------------------ Models ------------------
class UserProfile(BaseModel):
    name: str
    body_mass_kg: float = Field(..., ge=20, le=300)
    height_cm: float = Field(..., ge=120, le=230)

class AnalysisInput(BaseModel):
    exercise: str
    image_path: str
    external_load_kg: float = 0.0


def resolve_image_path(user_path: str, cfg: dict) -> str:
    p = Path(user_path)
    candidates = [
        p,
        BASE_DIR / p,
    ]
    # aus der Übungs-Config: pictures_dir (+ good/)
    pics = (cfg.get("meta") or {}).get("pictures_dir")
    if pics:
        pics = Path(pics)
        candidates += [pics / p.name, pics / "good" / p.name, pics / "bad" / p.name]

    for c in candidates:
        if c.is_file():
            return str(c)
    # nichts gefunden → gib den Originalpfad zurück (führt dann zu sauberer Fehlermeldung)
    return user_path

# ------------------ AI feedback ------------------
def ai_feedback(profile: UserProfile, analysis: AnalysisInput,
                angles: Dict[str, float], moments: Dict[str, float], knowledge_text: str) -> str:
    system_prompt = read_system_prompt(BASE_DIR)
    user_content = f"""
[PROFIL]
Name: {profile.name}
Gewicht: {profile.body_mass_kg} kg
Größe: {profile.height_cm} cm

[ANALYSE]
Übung: {analysis.exercise}
Zusatzlast: {analysis.external_load_kg} kg
Winkel (Grad): {json.dumps(angles, ensure_ascii=False)}

[WISSEN]
{knowledge_text}
""".strip()
    
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_content}
        ],
        temperature=0.2,
        max_tokens=1500
    )
    
    return resp.choices[0].message.content.strip()

# ------------------ Commands ------------------
def cmd_analyze():
    print("\n[bold cyan]Fitness Form Assistant[/bold cyan] - Analyse\n")
    name = Prompt.ask("Dein Name", default="Alex")
    body_mass = float(Prompt.ask("Körpergewicht [kg]", default="70"))
    height_cm = float(Prompt.ask("Größe [cm]", default="175"))
    profile = UserProfile(name=name, body_mass_kg=body_mass, height_cm=height_cm)

    print("\n[bold]Analyse[/bold]")
    exercise = Prompt.ask("Übung (z. B. Squat, Bizepscurls, Overhead Press)", default="Squat")
    external_load = parse_kg(Prompt.ask("Externe Last [kg] (z. B. Stange + Scheiben)", default="0"))

    # YAML-Konfiguration MUSS existieren
    cfg = find_config(BASE_DIR, exercise)
    if not cfg:
        print(f"[red]Keine Übungs-Config gefunden für '{exercise}'.[/red]")
        print("Bitte zuerst registrieren mit Option 2 im Menü")
        return

    # Session-Ordner
    session_dir = make_session_dir(os.path.join(BASE_DIR, "output"), slugify(profile.name), slugify(exercise))
    print(f"[dim]Session:[/dim] {session_dir}")

    # Winkel bestimmen
    pose = None
    image_path = Prompt.ask("Pfad zum Bild (leer lassen für Schätzung)", default="").strip()
    
    if image_path:
        image_path = resolve_image_path(image_path, cfg)
        analysis = AnalysisInput(exercise=exercise, image_path=image_path, external_load_kg=external_load)
        
        print("\n[cyan]→ Lese Pose aus Bild...[/cyan]")
        pose = get_pose_vector(image_path, enhanced=True)
        
        # Show pose quality metrics
        quality = pose.get("quality_metrics", {})
        if quality.get("is_high_quality", False):
            print(f"[green]✓ Pose-Qualität: Hoch (Score: {quality.get('pose_quality_score', 0):.2f})[/green]")
        else:
            print(f"[yellow]⚠ Pose-Qualität: Mittel (Score: {quality.get('pose_quality_score', 0):.2f})[/yellow]")
        
        angles = compute_angles_config(pose, cfg, use_3d=False)
        
        # Show perspective detection
        perspective = angles.get("__perspective__", "unknown")
        use_3d = angles.get("__use_3d__", False)
        perspective_emoji = {"frontal": "📷", "lateral": "📐", "oblique": "📊"}.get(perspective, "❓")
        angle_type = "3D" if use_3d else "2D"
        print(f"[cyan]{perspective_emoji} Kamera-Perspektive: {perspective.upper()} → {angle_type}-Winkel[/cyan]")
    else:
        # Dummy path für analysis wenn kein Bild
        analysis = AnalysisInput(exercise=exercise, image_path="", external_load_kg=external_load)
        
        # aus PDF (falls registriert) oder aus kurzer Beschreibung schätzen
        src_pdf = (cfg.get("meta") or {}).get("source_pdf_path")
        if src_pdf and os.path.isfile(src_pdf):
            print("\n[cyan]→ Kein Bild - schätze Winkel aus PDF-Text...[/cyan]")
            angles = estimate_angles_from_pdf(cfg, src_pdf)
        else:
            print("\n[cyan]→ Kein Bild & keine PDF-Quelle - schätze Winkel aus Beschreibung...[/cyan]")
            desc = Prompt.ask("Kurzbeschreibung der Pose (z. B. Ellbogen stark gebeugt, Oberarm senkrecht, ...)")
            angles = estimate_angles_from_text(cfg, desc)

    # No moments calculation - show "No moment data available" for cleaner look
    moments = {}
    
    # Tabelle (nur Winkel - sauber und professionell)
    t = Table(title="Joint Angles Analysis", box=box.SIMPLE_HEAVY)
    t.add_column("Joint"); t.add_column("Angle")
    for k, v in angles.items():
        if k.endswith("_deg"):
            joint_name = k.replace("_deg","").replace("_", " ").title()
            t.add_row(joint_name, f"{v:.1f}°")
    print(t)

    # Feedback + Persistenz - ERST generieren für die Visualisierung
    print("[cyan]→ Generiere KI-Feedback...[/cyan]")
    knowledge_text = read_knowledge(exercise, BASE_DIR)
    feedback = ai_feedback(profile, analysis, angles, moments, knowledge_text)
    
    # Nur Visualisierung erstellen wenn wir ein Bild haben
    if pose and image_path:
        print("[cyan]→ Erstelle professionelle Analyse-Visualisierung...[/cyan]")
        
        # Create professional combined analysis image
        profile_dict = {
            'name': profile.name,
            'body_mass_kg': profile.body_mass_kg,
            'height_cm': profile.height_cm
        }
        
        combined_path = create_combined_analysis_image(
            image_path, pose, cfg, angles, moments, profile_dict, session_dir, feedback_text=feedback
        )

        print(f"[green]✓ Professionelle Analyse erstellt:[/green] {combined_path}")
        print(f"[dim]Alle Visualisierungen in einem hochwertigen Bild kombiniert[/dim]")
    
    save_text(session_dir, "feedback.txt", feedback)
    save_json(session_dir, "angles.json", angles)
    save_json(session_dir, "moments.json", moments)
    save_json(session_dir, "profile.json", profile.model_dump())
    save_json(session_dir, "analysis.json", analysis.model_dump())

    print("\n[bold]Dein Feedback:[/bold]\n")
    print(feedback)
    print(f"\n[dim]Gespeichert in:[/dim] {session_dir}\n")
    
def cmd_list_exercises():
    cfgs = load_all_configs(BASE_DIR)
    if not cfgs:
        print("[yellow]Keine registrierten Übungen gefunden. Registriere zuerst eine PDF.[/yellow]")
        return
    t = Table(title="Registrierte Übungen", box=box.SIMPLE_HEAVY)
    t.add_column("Name", style="bold")
    t.add_column("Aliases")
    t.add_column("Datei")
    for c in cfgs:
        aliases = ", ".join(c.get("aliases", [])) or "-"
        t.add_row(c.get("name","?"), aliases, c.get("__path__",""))
    print(t)


def run_menu():
    while True:
        print("\n[bold cyan]Fitness Form Assistant - Menü[/bold cyan]")
        print("[1] Analyse starten")
        print("[2] Neue Übung registrieren (PDF + Bilder)")
        print("[3] Registrierte Übungen anzeigen")
        print("[q] Beenden")

        choice = Prompt.ask("\nAuswahl", choices=["1","2","3","q"], default="1")
        if choice == "1":
            cmd_analyze()
        elif choice == "2":
            # interaktiver Flow (fragt innerhalb der Funktion alles ab)
            register_exercise_flow()
        elif choice == "3":
            cmd_list_exercises()
        elif choice == "q":
            print("Bis bald!"); break

# ------------------ main ------------------
if __name__ == "__main__":
    try:
        run_menu()
    except KeyboardInterrupt:
        print("\n[dim]Abgebrochen.[/dim]")