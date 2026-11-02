"""Hy-MT2-7B 质量快测：同一批语料（英语34 / 日语15 / 用户20），v3 提示词，带上下文。
与 Hy-MT2-1.8B 的既有结果可直接对比（_bench_index.json 里的 Hy 有ctx 组）。
"""
import importlib.util, os, sys, json, time, subprocess

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
sys.path.insert(0, HERE)
from bench_index import EN, JA, USER20, judge, _clean_tr, _untranslated  # noqa: E402

SRV = os.path.join(HERE, "llama", "llama-server.exe")
M7 = os.path.join(HERE, "models", "Hy-MT2-7B-Q4_K_M.gguf")

subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
time.sleep(1.5)
tr = None
try:
    import live_caption as lc
    tr = lc.Translator(SRV, M7, port=8818, ctx=2048, use_context=True,
                       timeout=90.0, max_tokens=160, style="v3")
    tr.warmup()
except Exception as e:
    print("7B 启动失败：", type(e).__name__, str(e)[:200])
    raise SystemExit(1)

res = {}
for name, cases in (("英语", EN), ("日语", JA)):
    st = {"ok": 0, "miss": 0, "bad": 0, "n": 0}
    print(f"\n-- {name} --", flush=True)
    for src, tag, want, forbid in cases:
        zh = tr.translate(src)
        j = judge(zh, want, forbid)
        if want:
            st["n"] += 1
            st[{"OK": "ok", "MISS": "miss", "BAD": "bad"}[j]] += 1
        print(f"  [{tag:6s}][{j:4s}] {zh or '(空)'}")
    print(f"  => {st['ok']}/{st['n']}  错译 {st['bad']}")
    res[name] = st

print("\n-- 用户素材 20 句 --", flush=True)
rows = []
for cid, src in USER20:
    zh = tr.translate(src)
    rows.append({"tag": cid, "zh": zh})
    print(f"  [{cid:4s}] {zh or '(空)'}")
res["用户"] = {"rows": rows}
tr.close()

with open(os.path.join(TMP, "_bench_7b.json"), "w", encoding="utf-8") as fh:
    json.dump(res, fh, ensure_ascii=False, indent=1)
print("\n完成")
