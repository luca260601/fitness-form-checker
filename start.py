#!/usr/bin/env python3
"""
Fitness Form Checker - Web Application Starter
Professionelle Web-UI für KI-gestützte Bewegungsanalyse
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """Check if all required dependencies are installed"""
    print("🔍 Überprüfe Dependencies...")
    
    try:
        import fastapi, uvicorn, openai, cv2, mediapipe, matplotlib
        from dotenv import load_dotenv
        print("✅ Alle Dependencies installiert")
        return True
    except ImportError as e:
        print(f"❌ Fehlende Dependency: {e}")
        print("💡 Installiere mit: pip install -r requirements-web.txt")
        return False

def check_api_key():
    """Check if OpenAI API key is configured"""
    print("🔑 Überprüfe OpenAI API-Schlüssel...")
    
    from dotenv import load_dotenv
    load_dotenv()
    
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key and api_key.startswith('sk-'):
        print("✅ OpenAI API-Schlüssel gefunden")
        return True
    else:
        print("❌ OpenAI API-Schlüssel fehlt")
        print("💡 Erstelle .env Datei mit: OPENAI_API_KEY=dein_schluessel")
        return False

def start_application():
    """Start the web application"""
    print("\n🚀 Starte Fitness Form Checker Web App...")
    print("📱 Frontend: http://localhost:8000")
    print("🔧 API Docs: http://localhost:8000/docs")
    print("\n💡 Drücke Ctrl+C zum Beenden\n")
    
    try:
        import uvicorn
        from improved_web import app
        uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
    except KeyboardInterrupt:
        print("\n👋 Anwendung beendet")
    except Exception as e:
        print(f"❌ Fehler beim Starten: {e}")

def main():
    print("🏋️ Fitness Form Checker - Professional Web Application")
    print("=" * 60)
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Check API key
    if not check_api_key():
        print("\n⚠️ Warnung: Ohne API-Schlüssel funktioniert nur die Pose-Analyse,")
        print("   aber kein KI-Feedback. Trotzdem starten? (j/n): ", end="")
        if input().lower() not in ['j', 'ja', 'y', 'yes']:
            sys.exit(1)
    
    # Start application
    start_application()

if __name__ == "__main__":
    main()
