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
rem Pure-English stream (English podcast / English game stream / no Japanese at all).
rem Kept pure ASCII: cmd.exe reads .bat with the OEM codepage, so UTF-8 CJK in comments can
rem break line parsing and print "'x' is not recognized as an internal or external command".
rem Uses Index-Translate-2B alone: VRAM 1409 MiB instead of 7047 MiB for the three-model setup.
rem Index is a dedicated 150-language translation model (2.3B, based on Qwen3.5-2B). Measured on
rem the 34-sentence English set: same 94% as Hy-MT2-1.8B, but noticeably more natural on idioms:
rem   "I'm not gonna lie" -> honest opener (Hy: "I am really not lying")
rem   "piece of cake" -> a breeze;  "arm and a leg" -> costs a fortune
rem   "Nothing like buying things to make you happy" -> fully idiomatic
rem   "Go fuck yourself" -> keeps the profanity, does not sanitize it
rem IMPORTANT: --tr-style index means ONE user message, no system prompt, no context/history.
rem Appending anything (even the previous translation) makes it treat that text as the body
rem and start copying it (measured: repeats the previous sentence, double quotes leak out).
rem So --tr-pair / context features do not apply here; keep it stateless.
rem Glossary: works only partly here. Index has no system prompt, so terms can only be
rem PRE-SUBSTITUTED in the source. Single-word entries are safe (a player handle mapped to its
rem Chinese name tested clean, rack=bed fixed the sentence), but multi-word English idioms are
rem skipped on purpose ("break a leg" -> "good luck" would break the grammar of the source line).
rem English-only routing is automatic: with no --tr-model set, every sentence goes to Index,
rem so do not feed Japanese with this bat.
.venv\Scripts\python.exe live_caption.py --out transcript.txt --translate --model models\Qwen3-ASR-1.7B-q5_k.gguf --mmproj models\mmproj-Qwen3-ASR-1.7B-Q8_0.gguf --tr-model models\Index-Translate-2B-Q4_K_M.gguf --tr-style index --no-tr-context %*
