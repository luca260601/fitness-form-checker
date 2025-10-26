#!/usr/bin/env python3
"""
Fitness Form Checker - Verbesserte Web App
Mit Bild-Entfernen und besserer Fehlerbehandlung
"""

import os
import json
import uuid
import shutil
import traceback
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from fastapi import FastAPI, File, UploadFile, HTTPException, Form, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Import existing modules
from utils.parsing import parse_kg, slugify
from utils.file_ops import make_session_dir, save_text, save_json
from utils.new_knowlege import read_knowledge, read_system_prompt
from utils.openai_client import get_client
from utils.exercises import find_config, load_all_configs
from pose_service.engine import get_pose_vector, compute_angles_config
from pose_service.combined_visualization import create_combined_analysis_image

# Initialize
BASE_DIR = Path(__file__).resolve().parent
app = FastAPI(title="Fitness Form Checker", version="2.0.0")

# Create simple web directory
simple_web_dir = BASE_DIR / "simple_web"
simple_web_dir.mkdir(exist_ok=True)

# ============ MODELS ============

class UserProfile(BaseModel):
    name: str
    body_mass_kg: float
    height_cm: float

# ============ API ENDPOINTS ============

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve simple HTML page"""
    return HTMLResponse(content=get_html_content())

@app.get("/api/exercises")
async def get_exercises():
    """Get all available exercises"""
    try:
        configs = load_all_configs(BASE_DIR)
        exercises = []
        for cfg in configs:
            exercises.append({
                "name": cfg.get("name", "Unknown"),
                "aliases": cfg.get("aliases", []),
                "description": f"Analyse für {cfg.get('name', 'Unknown')}"
            })
        return {"exercises": exercises}
    except Exception as e:
        print(f"Error loading exercises: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Laden der Übungen: {str(e)}")

@app.post("/api/analyze")
async def analyze_form(
    profile_name: str = Form(...),
    body_mass_kg: float = Form(...),
    height_cm: float = Form(...),
    exercise: str = Form(...),
    external_load_kg: float = Form(0.0),
    image: UploadFile = File(...)
):
    """Analyze fitness form from uploaded image"""
    try:
        print(f"Starting analysis for {profile_name}, exercise: {exercise}")
        
        # Validate inputs
        if not profile_name or not exercise:
            raise HTTPException(status_code=400, detail="Name und Übung sind erforderlich")
        
        if not image or not image.filename:
            raise HTTPException(status_code=400, detail="Bild ist erforderlich")
        
        # Create session
        session_dir = make_session_dir(
            os.path.join(BASE_DIR, "output"), 
            slugify(profile_name), 
            slugify(exercise)
        )
        # Use session directory name as ID for consistent image serving
        session_id = Path(session_dir).name
        print(f"Created session directory: {session_dir}")
        
        # Save uploaded image
        image_path = os.path.join(session_dir, f"upload_{image.filename}")
        with open(image_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)
        print(f"Saved image: {image_path}")
        
        # Create profile
        profile = UserProfile(
            name=profile_name,
            body_mass_kg=body_mass_kg,
            height_cm=height_cm
        )
        
        # Find exercise config
        cfg = find_config(BASE_DIR, exercise)
        if not cfg:
            raise HTTPException(status_code=404, detail=f"Übung '{exercise}' nicht gefunden")
        print(f"Found exercise config: {cfg.get('name')}")
        
        # Analyze pose
        print("Analyzing pose...")
        pose = get_pose_vector(image_path, enhanced=True)
        if not pose or "landmarks" not in pose:
            raise HTTPException(status_code=400, detail="Keine Pose im Bild erkannt. Bitte verwende ein Bild mit einer Person.")
        
        angles = compute_angles_config(pose, cfg, use_3d=False)
        print(f"Computed angles: {list(angles.keys())}")
        
        # Generate AI feedback
        print("Generating AI feedback...")
        client = get_client()
        knowledge_text = read_knowledge(exercise, BASE_DIR)
        feedback = await generate_ai_feedback(profile, exercise, external_load_kg, angles, knowledge_text, client)
        
        # Create visualization
        print("Creating visualization...")
        profile_dict = profile.dict()
        combined_path = create_combined_analysis_image(
            image_path, pose, cfg, angles, {}, profile_dict, session_dir, feedback_text=feedback
        )
        
        # Save results
        save_json(session_dir, "angles.json", angles)
        save_json(session_dir, "profile.json", profile_dict)
        save_text(session_dir, "feedback.txt", feedback)
        
        # Get quality metrics
        quality = pose.get("quality_metrics", {})
        
        print("Analysis completed successfully")
        result = {
            "session_id": session_id,
            "angles": angles,
            "feedback": feedback,
            "image_path": f"/api/image/{session_id}/upload",
            "combined_analysis_path": f"/api/image/{session_id}/analysis",
            "perspective": angles.get("__perspective__", "unknown"),
            "quality_score": quality.get("pose_quality_score", 0.0)
        }
        
        print(f"Returning result with session_id: {session_id}")
        print(f"Image paths: upload={result['image_path']}, analysis={result['combined_analysis_path']}")
        
        return result
        
    except HTTPException:
        raise
    except Exception as e:
        error_msg = str(e)
        print(f"Analysis error: {error_msg}")
        print(f"Traceback: {traceback.format_exc()}")
        raise HTTPException(status_code=500, detail=f"Analyse-Fehler: {error_msg}")

@app.get("/api/image/{session_id}/{image_type}")
async def get_image(session_id: str, image_type: str):
    """Serve analysis images"""
    try:
        print(f"Looking for image: session_id={session_id}, type={image_type}")
        
        # Find session directory
        output_dir = BASE_DIR / "output" / "sessions"
        print(f"Searching in: {output_dir}")
        
        if not output_dir.exists():
            print("Output directory does not exist")
            raise HTTPException(status_code=404, detail="Output-Verzeichnis nicht gefunden")
        
        for session_dir in output_dir.glob("*"):
            print(f"Checking session dir: {session_dir}")
            if session_id in str(session_dir):
                print(f"Found matching session: {session_dir}")
                
                if image_type == "upload":
                    # Find uploaded image
                    upload_files = list(session_dir.glob("upload_*"))
                    print(f"Upload files found: {upload_files}")
                    if upload_files:
                        return FileResponse(upload_files[0])
                        
                elif image_type == "analysis":
                    # Find analysis image
                    analysis_files = list(session_dir.glob("combined_analysis_*.png"))
                    print(f"Analysis files found: {analysis_files}")
                    if analysis_files:
                        return FileResponse(analysis_files[0])
                    
                    # Also check for other analysis image patterns
                    other_files = list(session_dir.glob("*.png"))
                    print(f"Other PNG files found: {other_files}")
                    if other_files:
                        return FileResponse(other_files[0])
        
        print(f"No matching session found for {session_id}")
        raise HTTPException(status_code=404, detail="Bild nicht gefunden")
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Image serving error: {e}")
        raise HTTPException(status_code=500, detail=f"Fehler beim Laden des Bildes: {str(e)}")

# ============ HELPER FUNCTIONS ============

async def generate_ai_feedback(profile: UserProfile, exercise: str, external_load_kg: float, 
                             angles: Dict, knowledge_text: str, client) -> str:
    """Generate AI feedback (async version)"""
    system_prompt = read_system_prompt(BASE_DIR)
    user_content = f"""
[PROFIL]
Name: {profile.name}
Gewicht: {profile.body_mass_kg} kg
Größe: {profile.height_cm} cm

[ANALYSE]
Übung: {exercise}
Zusatzlast: {external_load_kg} kg
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

def get_html_content():
    """Get HTML content for the web interface"""
    return '''<!DOCTYPE html>
<html lang="de">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Fitness Form Checker</title>
    <script src="https://cdn.tailwindcss.com"></script>
    <script src="https://unpkg.com/lucide@latest/dist/umd/lucide.js"></script>
</head>
<body class="bg-gray-50">
    <div class="min-h-screen">
        <!-- Header -->
        <header class="bg-white shadow-sm border-b">
            <div class="max-w-7xl mx-auto px-4 py-4">
                <div class="flex items-center space-x-3">
                    <div class="bg-blue-600 p-2 rounded-lg">
                        <i data-lucide="activity" class="h-6 w-6 text-white"></i>
                    </div>
                    <div>
                        <h1 class="text-xl font-bold text-gray-900">Fitness Form Checker</h1>
                        <p class="text-sm text-gray-500">KI-gestützte Bewegungsanalyse</p>
                    </div>
                </div>
            </div>
        </header>

        <!-- Main Content -->
        <main class="max-w-4xl mx-auto px-4 py-8">
            <!-- Hero -->
            <div class="text-center mb-8">
                <h2 class="text-3xl font-bold text-gray-900 mb-4">
                    Analysiere deine Trainingsform
                </h2>
                <p class="text-lg text-gray-600">
                    Lade ein Foto deiner Übung hoch und erhalte sofortiges KI-Feedback
                </p>
            </div>

            <!-- Form -->
            <div class="bg-white rounded-xl shadow-sm border p-6 mb-8">
                <form id="analysisForm" enctype="multipart/form-data">
                    <!-- Profile Section -->
                    <div class="mb-6">
                        <h3 class="text-lg font-semibold mb-4">Persönliche Daten</h3>
                        <div class="grid md:grid-cols-3 gap-4">
                            <input type="text" name="profile_name" placeholder="Name" required
                                   class="border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500">
                            <input type="number" name="body_mass_kg" placeholder="Gewicht (kg)" required min="30" max="200"
                                   class="border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500">
                            <input type="number" name="height_cm" placeholder="Größe (cm)" required min="120" max="230"
                                   class="border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500">
                        </div>
                    </div>

                    <!-- Exercise Section -->
                    <div class="mb-6">
                        <h3 class="text-lg font-semibold mb-4">Übung</h3>
                        <div class="grid md:grid-cols-2 gap-4">
                            <select name="exercise" required
                                    class="border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500">
                                <option value="">Wähle eine Übung</option>
                            </select>
                            <input type="number" name="external_load_kg" placeholder="Zusatzgewicht (kg)" value="0" min="0" max="300"
                                   class="border rounded-lg px-3 py-2 focus:ring-2 focus:ring-blue-500">
                        </div>
                    </div>

                    <!-- Image Upload -->
                    <div class="mb-6">
                        <h3 class="text-lg font-semibold mb-4">Bild Upload</h3>
                        <div class="border-2 border-dashed border-gray-300 rounded-lg p-8 text-center">
                            <input type="file" name="image" accept="image/*" required
                                   class="hidden" id="imageInput">
                            <div id="uploadArea">
                                <label for="imageInput" class="cursor-pointer">
                                    <i data-lucide="upload" class="h-12 w-12 mx-auto text-gray-400 mb-4"></i>
                                    <p class="text-lg font-medium text-gray-900">Bild hochladen</p>
                                    <p class="text-sm text-gray-600">Klicke zum Auswählen</p>
                                </label>
                            </div>
                            <div id="imagePreview" class="mt-4 hidden">
                                <img id="previewImg" class="max-h-64 mx-auto rounded-lg mb-4">
                                <button type="button" id="removeImage" 
                                        class="bg-red-500 hover:bg-red-600 text-white px-4 py-2 rounded-lg">
                                    <i data-lucide="x" class="h-4 w-4 inline mr-1"></i>
                                    Bild entfernen
                                </button>
                            </div>
                        </div>
                        
                        <!-- Photo Tips -->
                        <div class="mt-4 p-4 bg-blue-50 rounded-lg">
                            <h4 class="font-medium text-blue-900 mb-2">📸 Foto-Tipps:</h4>
                            <ul class="text-sm text-blue-800 space-y-1">
                                <li>• <strong>Seitlich:</strong> Für Squats, Deadlifts, Lunges</li>
                                <li>• <strong>Frontal:</strong> Für Bizepscurls, Overhead Press</li>
                                <li>• <strong>Ganzer Körper sichtbar</strong> von Kopf bis Füße</li>
                                <li>• <strong>Gute Beleuchtung</strong> ohne Schatten</li>
                            </ul>
                        </div>
                    </div>

                    <!-- Submit Button -->
                    <div class="text-center">
                        <button type="submit" 
                                class="bg-blue-600 hover:bg-blue-700 text-white font-medium py-3 px-8 rounded-lg disabled:opacity-50">
                            <span id="submitText">Form analysieren</span>
                            <span id="loadingText" class="hidden">
                                <i data-lucide="loader-2" class="h-4 w-4 inline mr-2 animate-spin"></i>
                                Analysiere...
                            </span>
                        </button>
                    </div>
                </form>
            </div>

            <!-- Results -->
            <div id="results" class="hidden">
                <div class="bg-white rounded-xl shadow-sm border p-6">
                    <h3 class="text-lg font-semibold mb-4">Analyse-Ergebnisse</h3>
                    <div id="resultsContent"></div>
                </div>
            </div>
        </main>
    </div>

    <script>
        // Initialize Lucide icons
        lucide.createIcons();

        // Load exercises
        fetch('/api/exercises')
            .then(response => response.json())
            .then(data => {
                const select = document.querySelector('select[name="exercise"]');
                data.exercises.forEach(exercise => {
                    const option = document.createElement('option');
                    option.value = exercise.name;
                    option.textContent = exercise.name;
                    select.appendChild(option);
                });
            })
            .catch(error => {
                console.error('Error loading exercises:', error);
            });

        // Image preview and remove functionality
        document.getElementById('imageInput').addEventListener('change', function(e) {
            const file = e.target.files[0];
            if (file) {
                // Validate file size (max 10MB)
                if (file.size > 10 * 1024 * 1024) {
                    alert('Datei zu groß! Maximal 10MB erlaubt.');
                    this.value = '';
                    return;
                }
                
                const reader = new FileReader();
                reader.onload = function(e) {
                    document.getElementById('previewImg').src = e.target.result;
                    document.getElementById('imagePreview').classList.remove('hidden');
                    document.getElementById('uploadArea').classList.add('hidden');
                };
                reader.readAsDataURL(file);
            }
        });

        // Remove image functionality
        document.getElementById('removeImage').addEventListener('click', function() {
            document.getElementById('imageInput').value = '';
            document.getElementById('imagePreview').classList.add('hidden');
            document.getElementById('uploadArea').classList.remove('hidden');
            document.getElementById('previewImg').src = '';
        });

        // Form submission
        document.getElementById('analysisForm').addEventListener('submit', async function(e) {
            e.preventDefault();
            
            const submitBtn = e.target.querySelector('button[type="submit"]');
            const submitText = document.getElementById('submitText');
            const loadingText = document.getElementById('loadingText');
            
            // Show loading state
            submitBtn.disabled = true;
            submitText.classList.add('hidden');
            loadingText.classList.remove('hidden');
            lucide.createIcons();
            
            try {
                const formData = new FormData(e.target);
                
                // Validate form
                if (!formData.get('profile_name') || !formData.get('exercise') || !formData.get('image')) {
                    throw new Error('Bitte fülle alle Pflichtfelder aus');
                }
                
                const response = await fetch('/api/analyze', {
                    method: 'POST',
                    body: formData
                });
                
                if (!response.ok) {
                    const errorData = await response.json().catch(() => ({}));
                    throw new Error(errorData.detail || `HTTP ${response.status}: Analyse fehlgeschlagen`);
                }
                
                const result = await response.json();
                showResults(result);
                
            } catch (error) {
                console.error('Analysis error:', error);
                showError(error.message || 'Unbekannter Fehler bei der Analyse');
            } finally {
                // Reset button state
                submitBtn.disabled = false;
                submitText.classList.remove('hidden');
                loadingText.classList.add('hidden');
            }
        });

        function showError(errorMsg) {
            const resultsDiv = document.getElementById('results');
            const contentDiv = document.getElementById('resultsContent');
            contentDiv.innerHTML = `
                <div class="bg-red-50 border border-red-200 rounded-lg p-4">
                    <div class="flex items-start">
                        <i data-lucide="alert-circle" class="h-5 w-5 text-red-600 mr-3 mt-0.5"></i>
                        <div>
                            <h4 class="font-semibold text-red-800">Analyse fehlgeschlagen</h4>
                            <p class="text-red-700 text-sm mt-1">${errorMsg}</p>
                            <div class="text-red-600 text-xs mt-3">
                                <p class="font-medium mb-1">Bitte überprüfe:</p>
                                <ul class="list-disc list-inside space-y-1">
                                    <li>Alle Felder sind ausgefüllt</li>
                                    <li>Ein gültiges Bild ist hochgeladen (JPG, PNG)</li>
                                    <li>Das Bild zeigt eine Person in Trainingsposition</li>
                                    <li>Die Internetverbindung funktioniert</li>
                                </ul>
                            </div>
                        </div>
                    </div>
                </div>
            `;
            resultsDiv.classList.remove('hidden');
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
            lucide.createIcons();
        }

        function showResults(result) {
            const resultsDiv = document.getElementById('results');
            const contentDiv = document.getElementById('resultsContent');
            
            // Show perspective info
            const perspectiveHtml = `
                <div class="mb-6">
                    <h4 class="font-semibold mb-3 flex items-center"><i data-lucide="camera" class="h-4 w-4 mr-2"></i>Analyse-Info:</h4>
                    <div class="bg-blue-50 p-3 rounded-lg">
                        <p class="text-sm text-blue-800">
                            <strong>Perspektive:</strong> ${result.perspective} • 
                            <strong>Qualität:</strong> ${(result.quality_score * 100).toFixed(1)}%
                        </p>
                    </div>
                </div>
            `;
            
            // Show angles with German names and explanations
            const angleTranslations = {
                'front_knee_angle': { name: 'Vorderes Knie', desc: 'Beugung des vorderen Beins' },
                'rear_knee_angle': { name: 'Hinteres Knie', desc: 'Beugung des hinteren Beins' },
                'hip_angle': { name: 'Hüftwinkel', desc: 'Öffnung der Hüfte' },
                'trunk_inclination': { name: 'Rumpfneigung', desc: 'Neigung des Oberkörpers' },
                'knee': { name: 'Kniewinkel', desc: 'Beugung im Kniegelenk' },
                'hip': { name: 'Hüftwinkel', desc: 'Beugung im Hüftgelenk' },
                'ankle': { name: 'Sprunggelenk', desc: 'Winkel im Sprunggelenk' },
                'armwinkel_links': { name: 'Linker Arm', desc: 'Ellbogenwinkel links' },
                'armwinkel_rechts': { name: 'Rechter Arm', desc: 'Ellbogenwinkel rechts' },
                'rueckenwinkel': { name: 'Rückenwinkel', desc: 'Neigung der Wirbelsäule' }
            };
            
            let anglesHtml = '<div class="mb-6"><h4 class="font-semibold mb-3 flex items-center"><i data-lucide="ruler" class="h-4 w-4 mr-2"></i>Gelenkwinkel:</h4><div class="grid gap-3">';
            for (const [key, value] of Object.entries(result.angles)) {
                if (key.endsWith('_deg')) {
                    const cleanKey = key.replace('_deg', '').toLowerCase();
                    const translation = angleTranslations[cleanKey] || { 
                        name: key.replace('_deg', '').replace('_', ' ').replace(/\\b\\w/g, l => l.toUpperCase()),
                        desc: 'Gelenkwinkel'
                    };
                    
                    const isEstimated = result.angles.__trunk_estimated__ && key.includes('trunk');
                    
                    // Color coding based on angle ranges
                    let colorClass = 'text-blue-600';
                    if (cleanKey.includes('knee')) {
                        if (value >= 80 && value <= 110) colorClass = 'text-green-600';
                        else if (value >= 60 && value <= 130) colorClass = 'text-yellow-600';
                        else colorClass = 'text-red-600';
                    } else if (cleanKey.includes('hip')) {
                        if (value >= 70 && value <= 100) colorClass = 'text-green-600';
                        else if (value >= 50 && value <= 120) colorClass = 'text-yellow-600';
                        else colorClass = 'text-red-600';
                    }
                    
                    anglesHtml += `<div class="p-4 bg-white border border-gray-200 rounded-lg shadow-sm">
                        <div class="flex justify-between items-start">
                            <div>
                                <h5 class="font-medium text-gray-900">${translation.name}</h5>
                                <p class="text-xs text-gray-500 mt-1">${translation.desc}</p>
                            </div>
                            <div class="text-right">
                                <span class="font-bold text-lg ${colorClass}">${isEstimated ? '~' : ''}${value.toFixed(1)}°</span>
                                ${isEstimated ? '<p class="text-xs text-orange-500">geschätzt</p>' : ''}
                            </div>
                        </div>
                    </div>`;
                }
            }
            anglesHtml += '</div></div>';
            
            // Show images with error handling
            const imagesHtml = `
                <div class="mb-6">
                    <h4 class="font-semibold mb-3 flex items-center"><i data-lucide="image" class="h-4 w-4 mr-2"></i>Analyse-Visualisierung:</h4>
                    <img src="${result.combined_analysis_path}" 
                         class="w-full rounded-lg shadow-sm" 
                         alt="Analyse-Ergebnis"
                         onerror="this.style.display='none'; this.nextElementSibling.style.display='block';">
                    <div class="bg-yellow-50 border border-yellow-200 rounded-lg p-4 text-center hidden">
                        <i data-lucide="image-off" class="h-8 w-8 mx-auto text-yellow-600 mb-2"></i>
                        <p class="text-yellow-800">Visualisierung wird erstellt...</p>
                    </div>
                </div>
            `;
            
            // Parse and format feedback professionally
            const feedbackHtml = formatFeedback(result.feedback);
            
            contentDiv.innerHTML = perspectiveHtml + anglesHtml + imagesHtml + feedbackHtml;
            resultsDiv.classList.remove('hidden');
            
            // Scroll to results
            resultsDiv.scrollIntoView({ behavior: 'smooth' });
            lucide.createIcons();
        }

        function formatFeedback(feedbackText) {
            // Parse feedback sections
            const sections = {
                pros: [],
                cons: [],
                cues: [],
                general: []
            };
            
            const lines = feedbackText.split('\\n');
            let currentSection = 'general';
            
            for (const line of lines) {
                const trimmed = line.trim();
                if (!trimmed) continue;
                
                // Remove markdown formatting
                const cleanLine = trimmed.replace(/\\*\\*/g, '').replace(/\\*/g, '');
                
                // Detect sections
                if (cleanLine.toLowerCase().includes('was gut ist') || cleanLine.toLowerCase().includes('positiv')) {
                    currentSection = 'pros';
                    continue;
                } else if (cleanLine.toLowerCase().includes('was zu verbessern') || cleanLine.toLowerCase().includes('negativ')) {
                    currentSection = 'cons';
                    continue;
                } else if (cleanLine.toLowerCase().includes('konkrete cues') || cleanLine.toLowerCase().includes('tipps')) {
                    currentSection = 'cues';
                    continue;
                }
                
                // Add content to sections
                if (cleanLine.startsWith('-') || cleanLine.startsWith('•')) {
                    const content = cleanLine.substring(1).trim();
                    if (content) {
                        sections[currentSection].push(content);
                    }
                } else if (cleanLine.length > 10) {
                    sections.general.push(cleanLine);
                }
            }
            
            let html = '<div class="mb-6"><h4 class="font-semibold mb-4 flex items-center"><i data-lucide="brain" class="h-4 w-4 mr-2"></i>KI-Feedback:</h4>';
            
            // Positive feedback
            if (sections.pros.length > 0) {
                html += `
                    <div class="mb-4 bg-green-50 border border-green-200 rounded-lg p-4">
                        <h5 class="font-medium text-green-800 mb-2 flex items-center">
                            <i data-lucide="check-circle" class="h-4 w-4 mr-2"></i>
                            Was gut ist
                        </h5>
                        <ul class="space-y-1">
                `;
                sections.pros.forEach(item => {
                    html += `<li class="text-green-700 text-sm flex items-start">
                        <span class="text-green-500 mr-2">•</span>
                        ${item}
                    </li>`;
                });
                html += '</ul></div>';
            }
            
            // Areas for improvement
            if (sections.cons.length > 0) {
                html += `
                    <div class="mb-4 bg-orange-50 border border-orange-200 rounded-lg p-4">
                        <h5 class="font-medium text-orange-800 mb-2 flex items-center">
                            <i data-lucide="alert-triangle" class="h-4 w-4 mr-2"></i>
                            Verbesserungsmöglichkeiten
                        </h5>
                        <ul class="space-y-1">
                `;
                sections.cons.forEach(item => {
                    html += `<li class="text-orange-700 text-sm flex items-start">
                        <span class="text-orange-500 mr-2">•</span>
                        ${item}
                    </li>`;
                });
                html += '</ul></div>';
            }
            
            // Concrete tips
            if (sections.cues.length > 0) {
                html += `
                    <div class="mb-4 bg-blue-50 border border-blue-200 rounded-lg p-4">
                        <h5 class="font-medium text-blue-800 mb-2 flex items-center">
                            <i data-lucide="lightbulb" class="h-4 w-4 mr-2"></i>
                            Konkrete Tipps
                        </h5>
                        <ul class="space-y-1">
                `;
                sections.cues.forEach(item => {
                    html += `<li class="text-blue-700 text-sm flex items-start">
                        <span class="text-blue-500 mr-2">💡</span>
                        ${item}
                    </li>`;
                });
                html += '</ul></div>';
            }
            
            // General feedback
            if (sections.general.length > 0) {
                html += `
                    <div class="bg-gray-50 border border-gray-200 rounded-lg p-4">
                        <h5 class="font-medium text-gray-800 mb-2 flex items-center">
                            <i data-lucide="file-text" class="h-4 w-4 mr-2"></i>
                            Zusätzliche Hinweise
                        </h5>
                        <div class="text-gray-700 text-sm space-y-2">
                `;
                sections.general.forEach(item => {
                    html += `<p>${item}</p>`;
                });
                html += '</div></div>';
            }
            
            html += '</div>';
            return html;
        }
    </script>
</body>
</html>'''

# ============ STARTUP ============

if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Improved Fitness Form Checker Web App...")
    print("📱 Frontend: http://localhost:8000")
    print("🔧 API Docs: http://localhost:8000/docs")
    print("\n💡 Drücke Ctrl+C zum Beenden\n")
    
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
