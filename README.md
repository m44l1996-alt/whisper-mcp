# whisper-mcp

Servidor MCP local que transcribe audio y video a texto para Claude Desktop y Claude Code.
Funciona sin internet después de la primera descarga del modelo, no necesita API keys y
corre en CPU por defecto; puede usar GPU NVIDIA si instalas CUDA 12 y cuDNN (`WHISPER_DEVICE=cuda`).

Basado en [faster-whisper](https://github.com/SYSTRAN/faster-whisper), la implementación
más usada de Whisper para CPU/GPU, y en el SDK oficial de MCP para Python.

## Herramientas que expone

| Herramienta | Qué hace |
|---|---|
| `transcribe(path, language?, model?, format?, save?, wait_seconds?)` | Inicia la transcripción en segundo plano. Si termina en `wait_seconds` (máx. 50) devuelve el texto; si no, devuelve un `job_id`. Formatos: `timestamps` (por defecto), `text`, `srt`, `vtt`, `json`. Guarda el resultado junto al archivo. |
| `transcribe_status(job_id, wait_seconds?)` | Avance (etapa, porcentaje) y resultado final de un job. |
| `list_jobs()` | Jobs de la sesión. |
| `detect_language(path)` | Detecta el idioma hablado. |
| `list_models()` | Lista modelos disponibles y el predeterminado. |

Acepta mp4, mov, mkv, webm, mp3, wav, m4a, ogg, flac y cualquier formato que entienda FFmpeg
(se decodifica con PyAV, no hace falta instalar FFmpeg aparte).

## Requisitos

- [uv](https://docs.astral.sh/uv/) instalado (si ya usas `uvx` para otro servidor MCP, ya lo tienes).
- Conexión a internet la primera vez: `uv` instala las dependencias y se descarga el modelo
  desde Hugging Face (small ≈ 460 MB; se guarda en caché para siempre).

## Instalación en Claude Desktop (Windows)

Agrega este bloque dentro de `"mcpServers"` en `%APPDATA%\Claude\claude_desktop_config.json`
(Configuración → Desarrollador → Editar configuración), ajustando la ruta al `.py`:

```json
"whisper": {
  "command": "cmd",
  "args": ["/c", "uv", "run", "--script", "D:\\ruta\\a\\whisper-mcp\\whisper_mcp.py"],
  "env": {
    "WHISPER_MODEL": "small"
  }
}
```

Reinicia Claude Desktop (bandeja → Salir, volver a abrir). El primer arranque tarda unos minutos
mientras `uv` instala dependencias; los siguientes son inmediatos.

## Instalación en Claude Code

```bash
claude mcp add whisper -e WHISPER_MODEL=small -- uv run --script /ruta/a/whisper_mcp.py
```

## Variables de entorno

| Variable | Valores | Predeterminado |
|---|---|---|
| `WHISPER_MODEL` | tiny, base, small, medium, large-v3, large-v3-turbo | small |
| `WHISPER_DEVICE` | cpu, cuda | cpu |
| `WHISPER_COMPUTE` | int8, float16, default | int8 en CPU, float16 en GPU |

Guía rápida: `small` para CPU (español e inglés con buena calidad), `medium` o `large-v3-turbo`
si tienes GPU NVIDIA o necesitas máxima precisión, `tiny`/`base` solo para pruebas.

## GPU

Por defecto corre en CPU, que funciona en cualquier equipo. Para usar una GPU NVIDIA hace falta
tener instaladas las librerías de CUDA 12 y cuDNN 9 (si falta alguna, verás un error tipo
`cublas64_12.dll is not found`); después pon `"WHISPER_DEVICE": "cuda"` en el bloque `env`.

## Por qué en segundo plano

Algunos clientes MCP (por ejemplo Claude Desktop cuando reenvía herramientas a una sesión
remota) cortan las llamadas que pasan de ~60 s. Por eso `transcribe` devuelve enseguida un
`job_id` cuando el archivo es largo o el modelo aún se está descargando, y Claude consulta
`transcribe_status` hasta que termina. El archivo de salida se escribe igual aunque nadie consulte.

## Uso desde Claude

> Transcribe `D:\videos\entrevista.mp4` en español y guárdalo como SRT.

> ¿En qué idioma está `C:\audio\nota.m4a`?

## Licencia

MIT.
