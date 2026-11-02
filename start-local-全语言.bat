@echo off
chcp 65001 >nul
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo [live-caption] .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
REM Reuse the q5_k LLM already present on this machine + the Q8_0 mmproj.
.venv\Scripts\python.exe live_caption.py --out transcript.txt --model models\Qwen3-ASR-1.7B-q5_k.gguf --mmproj models\mmproj-Qwen3-ASR-1.7B-Q8_0.gguf %*
