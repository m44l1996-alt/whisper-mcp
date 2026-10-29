# Historial de cambios

Todas las versiones son del 29 de octubre de 2026 y corresponden a la puesta en marcha en
Windows 11 con Claude Desktop 4.x, `uv` y una sesión remota de Claude que reenvía las
herramientas del equipo. Cada versión es un commit en `main`; aquí se explica qué falló y cómo se corrigió.

## v4 · `76f40df` — CPU por defecto, GPU opcional

**Síntoma:** `RuntimeError: Library cublas64_12.dll is not found or cannot be loaded` al transcribir.

**Causa:** `WHISPER_DEVICE=auto` detectaba una GPU NVIDIA y CTranslate2 intentaba usar CUDA,
pero el equipo no tiene instaladas las DLL de CUDA 12 ni cuDNN 9.

**Corrección:** el predeterminado pasa a `cpu` con cuantización `int8` (rápida y sin dependencias
externas). La GPU se activa de forma explícita con `WHISPER_DEVICE=cuda` cuando CUDA está instalado.
Documentado en el README (sección GPU).

**Resultado:** primera transcripción exitosa. 34 s de video en una sola llamada; 1:27 y 1:51 en dos.

## v3 · `758e46d` — Fijar `av<16`

**Síntoma:** `TypeError: open() got an unexpected keyword argument 'metadata_errors'` al decodificar el video.

**Causa:** `uv` instaló PyAV 19. `faster-whisper` 1.2.x llama `av.open(..., metadata_errors="ignore")`,
parámetro que PyAV eliminó a partir de la versión 16, y su requisito `av>=11` no lo impide.

**Corrección:** dependencia `av>=11,<16` en la cabecera PEP 723 del script. PyAV 15.1 tiene wheels
para Windows/Python 3.13 y conserva el parámetro.

## v2 · `12eddaf` — Transcripción en segundo plano y errores visibles

**Síntoma 1:** `Device did not respond within 60s` en la primera transcripción. El puente que reenvía
las herramientas del equipo a una sesión remota de Claude corta cualquier llamada que pase de 60 s,
y la primera ejecución descarga el modelo (~460 MB).

**Síntoma 2:** las fallas devolvían solo `Error executing tool transcribe`, sin detalle.

**Corrección:** `transcribe` lanza un hilo y devuelve el resultado si termina dentro de `wait_seconds`
(máx. 50) o un `job_id` para consultar con `transcribe_status`; nuevo `list_jobs`. Toda excepción se
devuelve con su traceback completo. El archivo de salida se escribe aunque nadie consulte el job.

## v1 · `3507eed` — Versión inicial

Servidor MCP con `transcribe`, `detect_language` y `list_models` sobre faster-whisper; compatible con
el SDK de MCP 1.x (`FastMCP`) y 2.x (`MCPServer`). Funcionaba en pruebas locales (Linux) pero no se
había ejecutado todavía en el equipo destino.

## Lecciones

- Fijar versiones de dependencias nativas (PyAV, CTranslate2) aunque la librería principal no lo exija.
- Nunca asumir GPU: `auto` falla en equipos con tarjeta NVIDIA pero sin toolkit CUDA.
- Cualquier herramienta MCP que pueda tardar más de un minuto debe ser asíncrona (job + status).
- Devolver siempre el traceback: un error mudo cuesta un reinicio completo por cada hipótesis.
