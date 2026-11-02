@echo off
chcp 65001 >nul
cd /d %~dp0

set "PY=.venv\Scripts\python.exe"
set "HAVE_UV=0"
where uv >nul 2>nul
if not errorlevel 1 set "HAVE_UV=1"

echo [1/3] Python venv ...
if "%HAVE_UV%"=="1" (
  echo       uv found - using uv venv, much faster and no pip needed
  if not exist "%PY%" uv venv .venv
)
if not exist "%PY%" (
  echo       uv missing or unusable - falling back to python -m venv
  python -m venv .venv
)
if not exist "%PY%" (
  echo       FAILED: no usable Python. Install Python 3.12 or uv, then rerun this script.
  pause
  exit /b 1
)

if "%HAVE_UV%"=="1" (
  set "INSTALL=uv pip install -q --python %PY%"
) else (
  set "INSTALL=%PY% -m pip install -q"
  %PY% -m pip install -q --upgrade pip
)

%INSTALL% -r requirements.txt
if errorlevel 1 (
  echo       FAILED: dependency install. See the message above.
  pause
  exit /b 1
)

echo [2/3] Model (Qwen3-ASR-1.7B GGUF, ~2.8GB) ...
%INSTALL% huggingface_hub
"%PY%" -c "from huggingface_hub import hf_hub_download as d; [d('ggml-org/Qwen3-ASR-1.7B-GGUF', f, local_dir='models') for f in ('Qwen3-ASR-1.7B-Q8_0.gguf','mmproj-Qwen3-ASR-1.7B-bf16.gguf')]"
if errorlevel 1 (
  echo       FAILED: model download. Check your network and rerun this script.
  pause
  exit /b 1
)

echo [3/3] llama.cpp: put a recent Windows CUDA build (b10900+ / after 2026-04-12) into the llama\ folder next to this script.
echo       https://github.com/ggml-org/llama.cpp/releases  ^(llama-bXXXX-bin-win-cuda-13.x-x64.zip + cudart-llama-bin-win-cuda-13.x-x64.zip^)
if not exist llama\llama-server.exe echo       llama\llama-server.exe NOT found yet.
if not exist hotwords.txt copy hotwords.example.txt hotwords.txt >nul
echo Done. Run start-zh.bat (Chinese) or start.bat (auto language).
echo Note: if uv created this venv it intentionally has no pip inside, so "python -m pip" will NOT
echo       work. Install packages with "uv pip install <pkg>" instead. To get pip back it costs
echo       about 6 MB: "uv pip install pip"  ^(uv venv --seed does nothing on uv 0.13.6^).
pause
