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
from utils.ingest import extract_knowledge_from_pdf
from utils.config_gen import generate_config_from_pdf
from utils.exercises import find_config, load_all_configs
from utils.estimation import estimate_angles_from_text, estimate_angles_from_pdf

from pose_service.engine import get_pose_vector, compute_angles_config, estimate_moments_config
from pose_service.overlay import draw_vector_body_config, draw_force_overlay

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv()
client = get_client()

# ------------------ Models ------------------
class UserProfile(BaseModel):
    name: str
    body_mass_kg: float = Field(..., ge=20, le=300)
    height_cm: float = Field(..., ge=120, le=230)
    experience_level: str = "Anfänger"

class AnalysisInput(BaseModel):
    exercise: str
    image_path: str
    external_load_kg: float = 0.0

# ------------------ AI feedback ------------------
def ai_feedback(profile: UserProfile, analysis: AnalysisInput,
                angles: Dict[str, float], moments: Dict[str, float],
                knowledge_text: str) -> str:
    system_prompt = read_system_prompt(BASE_DIR)
    user_content = f"""
[PROFIL]
Name: {profile.name}
Gewicht: {profile.body_mass_kg} kg
Größe: {profile.height_cm} cm
Level: {profile.experience_level}

[ANALYSE]
Übung: {analysis.exercise}
Zusatzlast: {analysis.external_load_kg} kg
Winkel (Grad): {json.dumps(angles, ensure_ascii=False)}
Momente (N·m, vereinfacht): {json.dumps(moments, ensure_ascii=False)}

[WISSEN]
{knowledge_text}
""".strip()
    resp = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": [{"type": "input_text", "text": user_content}]}
        ],
        temperature=0.2
    )
    out = []
    for item in getattr(resp, "output", []):
        if getattr(item, "type", "") == "message":
            for ct in item.content:
                if ct.type == "output_text":
                    out.append(ct.text)
    return "\n".join(out).strip()

# ------------------ Commands ------------------
def cmd_analyze():
    print("\n[bold cyan]Fitness Form Assistant[/bold cyan] – Analyse\n")
    name = Prompt.ask("Dein Name", default="Alex")
    body_mass = float(Prompt.ask("Körpergewicht [kg]", default="70"))
    height_cm = float(Prompt.ask("Größe [cm]", default="175"))
    level = Prompt.ask("Trainingserfahrung", choices=["Anfänger","Fortgeschritten","Pro"], default="Anfänger")
    profile = UserProfile(name=name, body_mass_kg=body_mass, height_cm=height_cm, experience_level=level)

    print("\n[bold]Analyse[/bold]")
    exercise = Prompt.ask("Übung (z. B. Squat, Bizepscurls, Overhead Press)", default="Squat")
    image_path = Prompt.ask("Pfad zum Bild (leer lassen für Schätzung aus PDF/Text)", default="").strip()
    external_load = parse_kg(Prompt.ask("Externe Last [kg] (z. B. Stange + Scheiben)", default="0"))
    analysis = AnalysisInput(exercise=exercise, image_path=image_path, external_load_kg=external_load)

    # YAML MUSS existieren
    cfg = find_config(BASE_DIR, exercise)
    if not cfg:
        print(f"[red]Keine Übungs-Config gefunden für '{exercise}'.[/red]")
        print("Bitte zuerst registrieren mit:\n  python app.py register-exercise-pdf --name \"<Übung>\" --pdf \"/Pfad/datei.pdf\"")
        return

    # Session-Ordner
    session_dir = make_session_dir(os.path.join(BASE_DIR, "output"), slugify(profile.name), slugify(exercise))
    print(f"[dim]Session:[/dim] {session_dir}")

    # Winkel bestimmen
    pose = None
    if image_path:
        if not os.path.isfile(image_path):
            print(f"[red]Bild nicht gefunden:[/red] {image_path}")
            return
        print("\n[cyan]→ Lese Pose aus Bild...[/cyan]")
        pose = get_pose_vector(image_path)
        angles = compute_angles_config(pose, cfg)
    else:
        # aus PDF (falls registriert) oder aus kurzer Beschreibung schätzen
        src_pdf = (cfg.get("meta") or {}).get("source_pdf_path")
        if src_pdf and os.path.isfile(src_pdf):
            print("\n[cyan]→ Kein Bild – schätze Winkel aus PDF-Text...[/cyan]")
            angles = estimate_angles_from_pdf(cfg, src_pdf)
        else:
            print("\n[cyan]→ Kein Bild & keine PDF-Quelle – schätze Winkel aus Beschreibung...[/cyan]")
            desc = Prompt.ask("Kurzbeschreibung der Pose (z. B. Ellbogen stark gebeugt, Oberarm senkrecht, ...)")
            angles = estimate_angles_from_text(cfg, desc)

    # Momente (immer berechnen)
    moments = estimate_moments_config(cfg, angles, profile.body_mass_kg, analysis.external_load_kg)

    # Tabelle
    t = Table(title="Winkel & Momente (vereinfacht)", box=box.SIMPLE_HEAVY)
    t.add_column("Größe"); t.add_column("Wert")
    for k, v in angles.items():
        if k.endswith("_deg"):
            t.add_row(k.replace("_deg",""), f"{v}°")
    for k, v in moments.items():
        t.add_row(k, f"{v} N·m")
    print(t)

    # Zeichnen NUR wenn echte Pose vorhanden ist
    if pose:
        print("[cyan]→ Vektorfigur & Overlays...[/cyan]")
        vfiles = draw_vector_body_config(cfg, pose, angles, moments, session_dir)
        print(f"[green]vector svg:[/green] {vfiles['svg']}")
        print(f"[green]vector png:[/green] {vfiles['png']}")
        for joint in cfg.get("overlays", {}).get("arrows_at", []):
            try:
                pth = draw_force_overlay(analysis.image_path, pose, moments, joint, session_dir)
                print(f"[green]overlay {joint}:[/green] {pth}")
            except Exception as e:
                print(f"[yellow]Overlay {joint} übersprungen: {e}[/yellow]")
    else:
        print("[yellow]Kein Bild → keine Pose-Grafik. (Werte & Feedback wurden dennoch erstellt.)[/yellow]")

    # Feedback + Persistenz
    print("[cyan]→ Generiere KI-Feedback...[/cyan]")
    knowledge_text = read_knowledge(exercise, BASE_DIR)
    feedback = ai_feedback(profile, analysis, angles, moments, knowledge_text)
    save_text(session_dir, "feedback.txt", feedback)
    save_json(session_dir, "angles.json", angles)
    save_json(session_dir, "moments.json", moments)
    save_json(session_dir, "profile.json", profile.model_dump())
    save_json(session_dir, "analysis.json", analysis.model_dump())

    print("\n[bold]Dein Feedback:[/bold]\n")
    print(feedback)
    print(f"\n[dim]Gespeichert in:[/dim] {session_dir}\n")

def cmd_register_exercise_pdf(args):
    print("\n[bold cyan]Übung registrieren (mit PDF-Ingest) – atomar[/bold cyan]\n")
    name = args.name or Prompt.ask("Name der Übung")
    pdf  = args.pdf or Prompt.ask("Pfad zur PDF")

    # 0) Preflight: PDF muss existieren
    if not os.path.isfile(pdf):
        print(f"[red]PDF nicht gefunden:[/red] {pdf}")
        print("Abbruch – es wurde nichts angelegt.")
        return 2  # non-zero exit

    created_paths = []  # zum Aufräumen bei Fehlern
    try:
        # 1) Knowledge aus PDF
        md_path = extract_knowledge_from_pdf(BASE_DIR, name, pdf)
        created_paths.append(md_path)

        # 2) YAML-Config aus PDF (legt auch meta.source_pdf_path in der YAML ab)
        yml_path = generate_config_from_pdf(BASE_DIR, name, pdf)
        created_paths.append(yml_path)

        print(f"[green]Knowledge erstellt:[/green] {md_path}")
        print(f"[green]Übungs-Config erstellt:[/green] {yml_path}")
        print("[green]Registrierung abgeschlossen.[/green]")
        return 0

    except Exception as e:
        # Rollback
        for p in created_paths:
            try:
                if os.path.isfile(p):
                    os.remove(p)
            except Exception:
                pass
        print(f"[red]Registrierung fehlgeschlagen:[/red] {e}")
        print("[yellow]Es wurden keine Artefakte zurückgelassen.[/yellow]")
        return 1
    
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
        aliases = ", ".join(c.get("aliases", [])) or "–"
        t.add_row(c.get("name","?"), aliases, c.get("__path__",""))
    print(t)

def run_register_exercise_pdf_interactive():
    """Atomare Registrierung: Knowledge + YAML. Bei Fehler -> Rollback."""
    from utils.ingest import extract_knowledge_from_pdf
    from utils.config_gen import generate_config_from_pdf

    print("\n[bold cyan]Neue Übung registrieren (PDF) – atomar[/bold cyan]\n")
    name = Prompt.ask("Name der Übung")
    pdf  = Prompt.ask("Pfad zur PDF")

    if not os.path.isfile(pdf):
        print(f"[red]PDF nicht gefunden:[/red] {pdf}\n[yellow]Abbruch – nichts wurde angelegt.[/yellow]")
        return

    created_paths = []
    try:
        md_path  = extract_knowledge_from_pdf(BASE_DIR, name, pdf); created_paths.append(md_path)
        yml_path = generate_config_from_pdf(BASE_DIR, name, pdf);   created_paths.append(yml_path)
        print(f"[green]Knowledge erstellt:[/green] {md_path}")
        print(f"[green]Übungs-Config erstellt:[/green] {yml_path}")
        print("[green]Registrierung abgeschlossen.[/green]")
    except Exception as e:
        # Rollback
        for p in created_paths:
            try:
                if os.path.isfile(p): os.remove(p)
            except Exception:
                pass
        print(f"[red]Registrierung fehlgeschlagen:[/red] {e}")
        print("[yellow]Es wurden keine Artefakte zurückgelassen.[/yellow]")

def run_menu():
    while True:
        print("\n[bold cyan]Fitness Form Assistant – Menü[/bold cyan]")
        print("[1] Analyse starten")
        print("[2] Neue Übung registrieren (PDF)")
        print("[3] Registrierte Übungen anzeigen")
        print("[q] Beenden")

        choice = Prompt.ask("\nAuswahl", choices=["1","2","3","q"], default="1")
        if choice == "1":
            cmd_analyze()
        elif choice == "2":
            run_register_exercise_pdf_interactive()
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
