# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2", "faster-whisper>=1.1", "av>=11,<16"]
# ///
"""
Whisper MCP: transcripción local de audio y video para Claude Desktop y Claude Code.

Usa faster-whisper (CTranslate2). Acepta mp4, mov, mkv, mp3, wav, m4a, ogg, etc.
El modelo se descarga de Hugging Face la primera vez y queda en caché local.
Las transcripciones corren en segundo plano: `transcribe` devuelve un job_id al
instante y `transcribe_status` entrega el avance y el resultado. Así no importa
que el cliente MCP tenga un límite de tiempo por llamada.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid
from datetime import timedelta
from pathlib import Path

os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

try:  # SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("whisper")

MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
FORMATS = ("timestamps", "text", "srt", "vtt", "json")
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "cpu")          # cpu | cuda (cuda exige las DLL de CUDA 12 + cuDNN instaladas)
COMPUTE = os.environ.get("WHISPER_COMPUTE", "int8" if DEVICE == "cpu" else "float16")  # int8 | float16 | default

_models: dict[str, object] = {}
_models_lock = threading.Lock()
_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def _load(model: str):
    from faster_whisper import WhisperModel
    with _models_lock:
        if model not in _models:
            _models[model] = WhisperModel(model, device=DEVICE, compute_type=COMPUTE)
        return _models[model]


def _ts(seconds: float, sep: str) -> str:
    td = timedelta(seconds=max(0.0, float(seconds)))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{td.days * 24 + h:02d}:{m:02d}:{s:02d}{sep}{td.microseconds // 1000:03d}"


def _render(segments: list[dict], fmt: str) -> str:
    if fmt == "text":
        return "\n".join(s["text"].strip() for s in segments)
    if fmt == "srt":
        out = []
        for i, s in enumerate(segments, 1):
            out.append(f"{i}\n{_ts(s['start'], ',')} --> {_ts(s['end'], ',')}\n{s['text'].strip()}\n")
        return "\n".join(out)
    if fmt == "vtt":
        out = ["WEBVTT", ""]
        for s in segments:
            out.append(f"{_ts(s['start'], '.')} --> {_ts(s['end'], '.')}\n{s['text'].strip()}\n")
        return "\n".join(out)
    if fmt == "timestamps":
        return "\n".join(f"[{_ts(s['start'], '.')[:-4]}] {s['text'].strip()}" for s in segments)
    return json.dumps(segments, ensure_ascii=False, indent=2)


def _set(job_id: str, **kw) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kw)


def _run_job(job_id: str, p: Path, language: str | None, model: str, fmt: str, save: bool) -> None:
    try:
        _set(job_id, stage="cargando modelo (la primera vez lo descarga)")
        wm = _load(model)
        _set(job_id, stage="transcribiendo", progress=0.0)
        seg_iter, info = wm.transcribe(str(p), language=language, vad_filter=True, beam_size=5)
        total = float(info.duration) or 1.0
        segments: list[dict] = []
        for s in seg_iter:
            segments.append({"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text})
            _set(job_id, progress=min(0.99, s.end / total), segments=len(segments))
        body = _render(segments, fmt)
        header = (f"Archivo: {p.name} | Idioma: {info.language} ({info.language_probability:.0%}) "
                  f"| Duración: {_ts(info.duration, '.')[:-4]} | Modelo: {model} | Segmentos: {len(segments)}")
        saved = ""
        if save:
            ext = {"timestamps": ".txt", "text": ".txt", "srt": ".srt", "vtt": ".vtt", "json": ".json"}[fmt]
            out = p.with_suffix(p.suffix + ext)
            out.write_text(body, encoding="utf-8")
            saved = f"\nGuardado en: {out}"
        _set(job_id, status="done", stage="terminado", progress=1.0,
             result=f"{header}{saved}\n\n{body}", finished=time.time())
    except Exception:
        _set(job_id, status="error", stage="error", result="ERROR:\n" + traceback.format_exc(),
             finished=time.time())


def _status_text(job_id: str) -> str:
    with _jobs_lock:
        j = dict(_jobs.get(job_id) or {})
    if not j:
        return f"ERROR: no existe el job {job_id}"
    if j["status"] in ("done", "error"):
        return j["result"]
    elapsed = int(time.time() - j["started"])
    return (f"job {job_id} | estado: {j['status']} | etapa: {j['stage']} | "
            f"avance: {j.get('progress', 0.0):.0%} | segmentos: {j.get('segments', 0)} | "
            f"{elapsed} s transcurridos | archivo: {j['file']}\n"
            f"Vuelve a llamar transcribe_status(\"{job_id}\") en unos segundos.")


@mcp.tool()
def transcribe(
    path: str,
    language: str | None = None,
    model: str = DEFAULT_MODEL,
    format: str = "timestamps",
    save: bool = True,
    wait_seconds: int = 45,
) -> str:
    """Transcribe un archivo de audio o video local a texto. Corre en segundo plano.

    Devuelve el resultado si termina dentro de wait_seconds; si no, devuelve un job_id
    para consultar con transcribe_status(job_id).

    path: ruta completa del archivo (mp4, mov, mkv, mp3, wav, m4a, ogg...).
    language: código ISO como "es" o "en". Si se omite, se detecta.
    model: tiny | base | small | medium | large-v3 | large-v3-turbo. "small" equilibra
        velocidad y calidad en CPU; "medium" o "large-v3-turbo" si hay GPU.
    format: timestamps (por defecto) | text | srt | vtt | json.
    save: si es True, guarda el resultado junto al archivo original (.txt, .srt, .vtt o .json).
    wait_seconds: cuánto esperar antes de devolver el job_id (máximo 50).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return f"ERROR: no existe el archivo {p}"
    if model not in MODELS:
        return f"ERROR: modelo desconocido {model}. Opciones: {', '.join(MODELS)}"
    if format not in FORMATS:
        return f"ERROR: format debe ser uno de {', '.join(FORMATS)}"

    job_id = uuid.uuid4().hex[:8]
    with _jobs_lock:
        _jobs[job_id] = {"status": "running", "stage": "en cola", "progress": 0.0, "segments": 0,
                         "file": p.name, "started": time.time()}
    threading.Thread(target=_run_job, args=(job_id, p, language, model, format, save), daemon=True).start()

    deadline = time.time() + max(0, min(int(wait_seconds), 50))
    while time.time() < deadline:
        with _jobs_lock:
            if _jobs[job_id]["status"] != "running":
                break
        time.sleep(0.5)
    return _status_text(job_id)


@mcp.tool()
def transcribe_status(job_id: str, wait_seconds: int = 30) -> str:
    """Consulta el avance de una transcripción iniciada con transcribe. Devuelve el
    resultado completo cuando termina. Espera hasta wait_seconds (máximo 50) antes de responder."""
    deadline = time.time() + max(0, min(int(wait_seconds), 50))
    while time.time() < deadline:
        with _jobs_lock:
            j = _jobs.get(job_id)
            if not j or j["status"] != "running":
                break
        time.sleep(0.5)
    return _status_text(job_id)


@mcp.tool()
def list_jobs() -> str:
    """Lista las transcripciones en curso o terminadas en esta sesión."""
    with _jobs_lock:
        if not _jobs:
            return "No hay transcripciones en esta sesión."
        rows = [f"{k} | {v['status']} | {v['stage']} | {v.get('progress', 0.0):.0%} | {v['file']}"
                for k, v in _jobs.items()]
    return "\n".join(rows)


@mcp.tool()
def detect_language(path: str, model: str = "base") -> str:
    """Detecta el idioma hablado en un archivo de audio o video (usa el modelo base; la
    primera vez lo descarga, ~150 MB)."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"ERROR: no existe el archivo {p}"
    try:
        wm = _load(model)
        _, info = wm.transcribe(str(p), vad_filter=True)
        return f"{info.language} ({info.language_probability:.0%})"
    except Exception:
        return "ERROR:\n" + traceback.format_exc()


@mcp.tool()
def list_models() -> str:
    """Lista los modelos Whisper disponibles, cuál es el predeterminado y el dispositivo."""
    loaded = ", ".join(_models) or "ninguno todavía"
    return "\n".join(f"{'* ' if m == DEFAULT_MODEL else '  '}{m}" for m in MODELS) + \
        f"\n\nPredeterminado: {DEFAULT_MODEL} (variable WHISPER_MODEL). Dispositivo: {DEVICE}. " \
        f"Cargados en memoria: {loaded}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
