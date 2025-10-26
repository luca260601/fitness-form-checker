# 🏋️ Fitness Form Checker - Professional Web Application

Ein **professionelles KI-gestütztes System** zur Analyse von Trainingsformen mit Computer Vision und OpenAI. 

**Moderne Web-Oberfläche** für benutzerfreundliche Bewegungsanalyse.

## ✨ Features

- **🎯 Moderne Web-UI**: Professionelle, responsive Benutzeroberfläche
- **📷 Drag & Drop Upload**: Einfaches Hochladen von Trainingsbildern
- **🤖 KI-Pose-Erkennung**: Automatische Körperhaltungs-Analyse mit MediaPipe
- **📐 Winkel-Anzeige**: Verständliche Gelenkwinkel mit Erklärungen
- **🧠 Intelligentes Feedback**: Personalisierte Verbesserungsvorschläge mit OpenAI
- **🎨 Professionelle Visualisierung**: Saubere Analyse-Berichte ohne Redundanz
- **🔄 Live-Analyse**: Sofortige Ergebnisse in der Web-Oberfläche

## 🚀 Schnellstart

### 1. Dependencies installieren
```bash
pip install -r requirements-web.txt
```

### 3. Web-App starten
```bash
python start.py
```

**Dann öffne:** http://localhost:8000

## 🎯 Verwendung

### Web-Interface (Empfohlen)
1. **🌐 Browser öffnen:** http://localhost:8000
2. **👤 Profil eingeben:** Name, Gewicht, Größe
3. **🏋️ Übung wählen:** Aus Dropdown-Liste
4. **📷 Bild hochladen:** Drag & Drop oder Klick
5. **⚡ Analysieren:** Button klicken
6. **📊 Ergebnisse ansehen:** Strukturierte Analyse

### Kommandozeile (Legacy)
```bash
python app.py --profile "Max" --exercise "Squats" --image "squat.jpg"
```

## 🏋️ Unterstützte Übungen

| Übung | Deutsche Bezeichnung | Analyse-Typ |
|-------|---------------------|-------------|
| **Squats** | Kniebeugen | Bilateral (beide Beine) |
| **Lunges** | Ausfallschritte | Unilateral (Front/Rear) |
| **Bizepscurls** | Bizeps-Training | Armwinkel-Analyse |
| **Lattziehen** | Rückentraining | Rumpf + Arme |

## 📊 Was du bekommst

### **🎨 Professionelle Web-UI:**
- **Winkel-Namen** mit Erklärungen
- **Farb-Kodierung** (Grün/Gelb/Rot) für sofortige Bewertung
- **Strukturiertes KI-Feedback** in Kategorien
- **Saubere Visualisierung** ohne redundante Panels

### **📐 Intelligente Winkel-Analyse:**
```
🦵 Vorderes Knie          88.0°
   Beugung des vorderen Beins

🏃 Hüftwinkel            132.6°
   Öffnung der Hüfte

🏋️ Rumpfneigung           0.6°
   Neigung des Oberkörpers
```

### **🤖 KI-Feedback-Kategorien:**
- 🟢 **Was gut ist** - Positive Aspekte
- 🟠 **Verbesserungsmöglichkeiten** - Optimierungspotential  
- 🔵 **Konkrete Tipps** - Handlungsempfehlungen

## 🛠️ Technische Details

- **Frontend**: HTML5 + TailwindCSS + Lucide Icons
- **Backend**: FastAPI + Python 3.8+
- **Computer Vision**: MediaPipe für Pose-Erkennung
- **KI-Analyse**: OpenAI GPT-4 für Feedback
- **Visualisierung**: Matplotlib (optimiert)

## 📁 Projektstruktur

```
fitness-form-checker/
├── start.py              # 🚀 Haupt-Startdatei
├── improved_web.py       # 🌐 Web-Anwendung
├── pose_service/         # 🎯 Pose-Analyse
├── utils/               # 🔧 Hilfsfunktionen  
├── knowledge/           # 📚 Übungs-Wissen
├── data/               # ⚙️ Konfigurationen
└── output/             # 📊 Ergebnisse
```

## 🔧 Systemanforderungen

- **Python**: 3.8 oder höher
- **RAM**: Mindestens 4GB
- **OpenAI API**: Gültiger Schlüssel erforderlich
- **Browser**: Chrome, Firefox, Safari, Edge

## 💡 Tipps für beste Ergebnisse

### **📸 Foto-Qualität:**
- **Ganzer Körper sichtbar** von Kopf bis Füße
- **Gute Beleuchtung** ohne starke Schatten
- **Klarer Hintergrund** für bessere Erkennung

### **📐 Kamera-Perspektive:**
- **Seitlich**: Für Squats, Deadlifts, Lunges
- **Frontal**: Für Bizepscurls, Overhead Press
- **Augenhöhe**: Kamera auf Hüfthöhe positionieren

## 🎉 Entwickelt für Professionalität

Diese Web-Anwendung wurde speziell entwickelt für:
- **Fitness-Trainer** - Kunden-Analyse
- **Physiotherapeuten** - Bewegungsanalyse  
- **Sportler** - Technik-Optimierung
- **Fitness-Enthusiasten** - Selbst-Coaching

---

**🚀 Starte jetzt mit `python start.py` und erlebe professionelle Bewegungsanalyse!**
