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
rem Pure-Japanese stream (JP game / JP talk / JP stream). One translation model only.
rem Kept pure ASCII: cmd.exe reads .bat with the OEM codepage, so UTF-8 CJK in comments can
rem break line parsing and print "'x' is not recognized as an internal or external command".
rem Hy-MT2-1.8B is the best Japanese translator measured here: 13/15 on the JP set, vs 10/15 for
rem Index-Translate-2B and 9-10/15 for the Qwen 2B models. It gets the pragmatics right:
rem   "he has a wide face" (idiom = well-connected) -> well-connected; Index: his face is wide
rem   "the week after next" -> correct; Index says "next week"
rem   "at the end of last month" -> correct; Index says "at the end of this month"
rem   "that is just too much" (complaint) -> too much; Index: "that really is too serious"
rem VRAM (RTX 4060 Laptop 8GB, desktop baseline 1757): ASR 2693 + Hy-MT2-1.8B 1188 = 3881
rem   increment, 5638 MiB total on card, ~2320 MiB free. Plenty of headroom.
rem --tr-style v3 splits by script internally: kana sentences use the v2 rules (two JP examples),
rem others use the v3 rules. For a pure-JP stream only the JP branch ever fires.
rem Japanese sentences deliberately get NO context (measured: context breaks "riding on the
rem wind" and "round"; the v2 baseline is 15/15 without it). Do not add --tr-context here.
rem Glossary: glossary.txt works here too (--tr-style v3 can inject it). Measured both JP time
rem words at 5/5, with English 31/34 unchanged and latency 98->95ms.
rem Do NOT swap --tr-model to a Qwen 2B: with v1/v2/v3 prompts long sentences collapse into repeats.
rem Language is pinned with --lang ja: it writes "language Japanese<asr_text>" as the assistant
rem prefix, which is the only thing that actually forces the output language. Without it the
rem hotwords list (LPL, CUDA, PyTorch...) drags the model into English: on the same 64.6s JP clip
rem 4/11 segments came out as romanized English ("Mizuki-chan, what's wrong?"), with the prefix
rem it is 11/11 Japanese and nothing is dropped.
.venv\Scripts\python.exe live_caption.py --out transcript.txt --translate --model models\Qwen3-ASR-1.7B-q5_k.gguf --mmproj models\mmproj-Qwen3-ASR-1.7B-Q8_0.gguf --lang ja --tr-model models\Hy-MT2-1.8B-Q4_K_M.gguf --tr-style v3 %*
