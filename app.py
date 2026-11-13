from pathlib import Path
import shutil, subprocess, uuid, json, re
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).parent
UPLOADS = ROOT / "uploads"
OUTPUTS = ROOT / "outputs"
UPLOADS.mkdir(exist_ok=True)
OUTPUTS.mkdir(exist_ok=True)

app = FastAPI(title="CLIP LUY")
app.mount("/outputs", StaticFiles(directory=OUTPUTS), name="outputs")

def run(cmd):
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode:
        raise RuntimeError(p.stderr[-2500:])
    return p.stdout

def duration(path):
    out = run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", str(path)])
    return float(out.strip())

def transcribe(path):
    try:
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("Falta faster-whisper. Ejecuta instalar.bat.")
    # CPU int8 funciona en equipos Windows sin GPU dedicada.
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, info = model.transcribe(str(path), vad_filter=True, word_timestamps=False)
    result = []
    for s in segments:
        txt = (s.text or "").strip()
        if txt:
            result.append({"start": float(s.start), "end": float(s.end), "text": txt})
    return result, info.language

def choose_moments(segments, total, clip_len=30, max_clips=8):
    """Puntuación inicial basada en señales lingüísticas sencillas; no es juicio semántico."""
    candidates = []
    hooks = ["pero", "porque", "increíble", "nunca", "siempre", "verdad", "error",
             "secreto", "mira", "wow", "atención", "resulta", "finalmente",
             "no puede", "imagina", "problema", "solución"]
    for s in segments:
        text = s["text"].lower()
        words = re.findall(r"\w+", text)
        score = min(len(words), 22) / 5
        score += sum(1.5 for h in hooks if h in text)
        score += 1.0 if "?" in text or "!" in text else 0
        center = (s["start"] + s["end"]) / 2
        start = max(0, center - clip_len / 2)
        end = min(total, start + clip_len)
        start = max(0, end - clip_len)
        if end - start >= 8:
            candidates.append({"start": start, "end": end, "score": score, "text": s["text"]})
    candidates.sort(key=lambda x: x["score"], reverse=True)
    chosen = []
    for c in candidates:
        if all(abs(c["start"] - x["start"]) > clip_len * .75 for x in chosen):
            chosen.append(c)
        if len(chosen) >= max_clips:
            break
    return sorted(chosen, key=lambda x: x["start"])

@app.get("/", response_class=HTMLResponse)
def home():
    return (ROOT / "static" / "index.html").read_text(encoding="utf-8")

@app.post("/process")
async def process(video: UploadFile = File(...), clip_seconds: int = 30):
    if not video.filename:
        raise HTTPException(400, "Selecciona un video.")
    ext = Path(video.filename).suffix.lower()
    if ext not in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
        raise HTTPException(400, "Formato no compatible. Usa MP4, MOV, MKV, WEBM o AVI.")
    job = uuid.uuid4().hex[:10]
    source = UPLOADS / f"{job}{ext}"
    with source.open("wb") as f:
        shutil.copyfileobj(video.file, f)
    try:
        total = duration(source)
        segments, language = transcribe(source)
        if not segments:
            raise RuntimeError("No se detectó voz. Prueba con un video que tenga diálogo claro.")
        moments = choose_moments(segments, total, clip_seconds)
        if not moments:
            raise RuntimeError("No se encontraron fragmentos adecuados.")
        clips = []
        for i, m in enumerate(moments, 1):
            out = OUTPUTS / f"CLIP_LUY_{job}_{i:02d}.mp4"
            # Escala y recorta al centro para 9:16. Sin marca de agua.
            vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
            run(["ffmpeg", "-y", "-ss", f"{m['start']:.2f}", "-i", str(source),
                 "-t", f"{m['end']-m['start']:.2f}", "-vf", vf,
                 "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                 "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(out)])
            clips.append({"name": out.name, "url": f"/outputs/{out.name}",
                          "start": round(m["start"], 1), "end": round(m["end"], 1),
                          "score": round(m["score"], 1), "transcript": m["text"]})
        return {"language": language, "clips": clips,
                "note": "Selección automática preliminar. Revisa cada clip antes de publicarlo."}
    except Exception as e:
        raise HTTPException(500, str(e))
    finally:
        try:
            source.unlink(missing_ok=True)
        except Exception:
            pass
