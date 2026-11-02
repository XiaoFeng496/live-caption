@echo off
chcp 65001 >nul
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo [live-caption] .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
REM llama.cpp: use the build on PATH when llama\ is not bundled.
if not exist "llama\llama-server.exe" if not defined LLAMA_SERVER for %%I in (llama-server.exe) do set "LLAMA_SERVER=%%~$PATH:I"
.venv\Scripts\python.exe live_caption.py --lang zh --out transcript.txt %*
