# scripts/download_yt.py
import sys, pathlib
from yt_dlp import YoutubeDL

OUTDIR = pathlib.Path("assets/sample_clips")
OUTDIR.mkdir(parents=True, exist_ok=True)

if len(sys.argv) < 2:
    print("Usage: python scripts/download_yt.py <YOUTUBE_URL> [--maxres 1080]")
    sys.exit(1)

url = sys.argv[1]
maxres = 1080
if "--maxres" in sys.argv:
    i = sys.argv.index("--maxres")
    if i+1 < len(sys.argv):
        try: maxres = int(sys.argv[i+1])
        except: pass

# Format-Strategie:
# 1) bestvideo (mp4, <=maxres) + m4a; 2) sonst best mp4; 3) sonst best
fmt = f"bestvideo[ext=mp4][height<={maxres}]+bestaudio[ext=m4a]/best[ext=mp4]/best"

ydl_opts = {
    "format": f"bestvideo[ext=mp4][height<={maxres}]+bestaudio[ext=m4a]/best[ext=mp4]/best",
    "outtmpl": str(OUTDIR / "%(title).80s.%(ext)s"),
    "merge_output_format": "mp4",   # ffmpeg mergen/remuxen
    "quiet": False,
    "forceipv4": True,
}


with YoutubeDL(ydl_opts) as ydl:
    info = ydl.extract_info(url, download=True)
    print("\nSaved to:", OUTDIR.resolve())
