# /// script
# requires-python = ">=3.10"
# dependencies = ["mcp>=1.2", "faster-whisper>=1.1"]
# ///
"""
Whisper MCP: transcripción local de audio y video para Claude Desktop.

Usa faster-whisper (CTranslate2). Acepta mp4, mov, mkv, mp3, wav, m4a, ogg, etc.
El modelo se descarga de Hugging Face la primera vez y queda en caché local.
No sube nada a internet después de eso.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import timedelta
from pathlib import Path

try:  # SDK 2.x
    from mcp.server.mcpserver import MCPServer as _Server
except ImportError:  # SDK 1.x
    from mcp.server.fastmcp import FastMCP as _Server

mcp = _Server("whisper")

MODELS = ["tiny", "base", "small", "medium", "large-v3", "large-v3-turbo"]
DEFAULT_MODEL = os.environ.get("WHISPER_MODEL", "small")
DEVICE = os.environ.get("WHISPER_DEVICE", "auto")        # auto | cpu | cuda
COMPUTE = os.environ.get("WHISPER_COMPUTE", "default")    # default | int8 | float16
_cache: dict[str, object] = {}


def _load(model: str):
    from faster_whisper import WhisperModel
    if model not in _cache:
        _cache[model] = WhisperModel(model, device=DEVICE, compute_type=COMPUTE)
    return _cache[model]


def _ts(seconds: float, sep: str) -> str:
    td = timedelta(seconds=max(0.0, seconds))
    h, rem = divmod(td.seconds, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}{sep}{td.microseconds // 1000:03d}"


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


@mcp.tool()
def transcribe(
    path: str,
    language: str | None = None,
    model: str = DEFAULT_MODEL,
    format: str = "timestamps",
    save: bool = True,
) -> str:
    """Transcribe un archivo de audio o video local a texto.

    path: ruta completa del archivo (mp4, mov, mkv, mp3, wav, m4a, ogg...).
    language: código ISO como "es" o "en". Si se omite, se detecta.
    model: tiny | base | small | medium | large-v3 | large-v3-turbo. "small" equilibra
        velocidad y calidad en CPU; "medium" o "large-v3-turbo" si hay GPU o se quiere más precisión.
    format: timestamps (por defecto) | text | srt | vtt | json.
    save: si es True, guarda el resultado junto al archivo original (.txt, .srt, .vtt o .json).
    """
    p = Path(path).expanduser()
    if not p.is_file():
        return f"ERROR: no existe el archivo {p}"
    if model not in MODELS:
        return f"ERROR: modelo desconocido {model}. Opciones: {', '.join(MODELS)}"
    if format not in ("timestamps", "text", "srt", "vtt", "json"):
        return "ERROR: format debe ser timestamps, text, srt, vtt o json"

    wm = _load(model)
    seg_iter, info = wm.transcribe(str(p), language=language, vad_filter=True, beam_size=5)
    segments = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text} for s in seg_iter]
    body = _render(segments, format)

    header = (f"Archivo: {p.name} | Idioma: {info.language} ({info.language_probability:.0%}) "
              f"| Duración: {_ts(info.duration, '.')[:-4]} | Modelo: {model} | Segmentos: {len(segments)}")
    saved = ""
    if save:
        ext = {"timestamps": ".txt", "text": ".txt", "srt": ".srt", "vtt": ".vtt", "json": ".json"}[format]
        out = p.with_suffix(p.suffix + ext)
        out.write_text(body, encoding="utf-8")
        saved = f"\nGuardado en: {out}"
    return f"{header}{saved}\n\n{body}"


@mcp.tool()
def detect_language(path: str, model: str = "base") -> str:
    """Detecta el idioma hablado en un archivo de audio o video sin transcribirlo completo."""
    p = Path(path).expanduser()
    if not p.is_file():
        return f"ERROR: no existe el archivo {p}"
    wm = _load(model)
    _, info = wm.transcribe(str(p), vad_filter=True)
    return f"{info.language} ({info.language_probability:.0%})"


@mcp.tool()
def list_models() -> str:
    """Lista los modelos Whisper disponibles y cuál es el predeterminado."""
    return "\n".join(f"{'* ' if m == DEFAULT_MODEL else '  '}{m}" for m in MODELS) + \
        f"\n\nPredeterminado: {DEFAULT_MODEL} (variable WHISPER_MODEL). Dispositivo: {DEVICE}."


if __name__ == "__main__":
    mcp.run(transport="stdio")
