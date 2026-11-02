@echo off
chcp 65001 >nul
cd /d %~dp0
if not exist .venv\Scripts\python.exe (
  echo [live-caption] .venv not found. Run setup.bat first.
  pause
  exit /b 1
)
rem Real-time translate to Chinese (Japanese + English mixed stream).
rem Kept pure ASCII on purpose: cmd.exe parses .bat with the OEM codepage (GBK on a zh-CN box),
rem so UTF-8 CJK comments get mis-split and can swallow a newline or the "rem" keyword, which
rem surfaces as "'x' is not recognized as an internal or external command".
rem Default = Hy-MT2-1.8B (1.06GB, ~60-100ms/sentence) + v3 prompt (best all-round).
rem v3 auto-splits by language: Japanese -> v2 rules (lyrics/idioms, two JP examples),
rem others (English etc.) -> v3 rules (idioms: "pulling my leg" = teasing, "Not bad" = pretty
rem good; software jargon: ship = release, NOT physically shipping goods; deprecate = retire;
rem edge case = corner case).
rem Measured: English 34 sentences cur 88.2% / v2 88.2% / v3 94.1%; split 33/33.
rem Best wording, slower and +1.5GB VRAM: swap --tr-model to models\Qwen3.5-4B-Q4_K_M.gguf
rem Do NOT pair Qwen3.5-2B with --tr-style v1/v2/v3 (long sentences collapse into repeats).
rem Two subtitle lines refresh independently by default: top = live original, bottom = last
rem translation, which stays until the next translation arrives. Add --tr-pair if you want the
rem pair to switch together (both lines always the same sentence, but original is no longer live).
rem Context is ON by default: English sentences carry the last 2 translations in the system
rem prompt (fixes term consistency: blood money -> hard-earned money, "stick that on the credit
rem card" -> pay by card).
rem Japanese sentences deliberately carry none (context makes "riding on the wind" and "round"
rem come out wrong). Disable all with --no-tr-context.
rem ASR VRAM: default --asr-ctx 4096 (-np 2, so 2048 per slot). A 6s sentence only needs ~117
rem tokens, so 4096 is already generous. Use --asr-ctx 2048 to free another 222MB (measured
rem 2726 -> 2504 MiB, translations identical). Do NOT drop --slots to 1: saves 8MB but makes
rem the live/interim line queue behind the final one (+129ms).
rem Glossary (--glossary, default glossary.txt): one "source=target" per line, only entries that
rem actually appear in a sentence are used. Latin entries (rack=bed, blood money=hard-earned
rem money, character names) go into the system prompt: 5 formats x 5 runs = 10/10 hit, 0
rem pollution. CJK entries (e.g. a Japanese "week after next") are PRE-SUBSTITUTED in the source
rem instead, because injecting them makes 1.8B emit Korean (a Korean particle showed up inside
rem the Chinese output); no wording fixes that (3/3 still drifted), pre-substitution is clean
rem (3/3). Measured: rack 5/5, both JP time words 5/5; English 31->31, Japanese 13->15.
rem Do not put multi-word English idioms in it: pre-substitution breaks grammar, and injection
rem only works on v3/v1/v2/qwen.
rem Per-language routing: Japanese (any text with kana) -> Hy-MT2-1.8B, everything else
rem (English/Latin) -> Index-Translate-2B. Both servers stay resident, so switching costs 0ms.
rem Measured: Index is better at English idioms ("Break a leg" -> good luck, "Go fuck yourself"
rem -> literal profanity) and Hy is far better at Japanese (13/15 vs 10/15).
rem VRAM measured per server (RTX 4060 Laptop 8GB, desktop baseline 1757 MiB):
rem   ASR 2693 + Hy-MT2-1.8B 1188 + Index-Translate-2B 1409 = 7047 MiB, 911 MiB free.
rem Shrinking the English model ctx 2048->512 saves only 7 MiB (KV is not the cost), not worth it.
rem Drop --tr-model-en for a pure-Japanese stream (-1409 MiB). For pure English use the
rem English-only bat in this folder: Index alone is 1409 MiB, no Hy needed.
.venv\Scripts\python.exe live_caption.py --out transcript.txt --translate --model models\Qwen3-ASR-1.7B-q5_k.gguf --mmproj models\mmproj-Qwen3-ASR-1.7B-Q8_0.gguf --tr-model models\Hy-MT2-1.8B-Q4_K_M.gguf --tr-style v3 --tr-model-en models\Index-Translate-2B-Q4_K_M.gguf %*
