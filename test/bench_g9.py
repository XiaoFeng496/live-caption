"""补测 G9：把 ASR 听坏的那句还原成正确英文，看模型本身到底行不行。

原句（ASR 输出，坏了两处）：
  I got the pills, Amanda. Jesus! I don't know how you get out of the fucking rack
  in the morning. They got you thoped up like at the pressed elephant.
正确原文：
  ... They got you doped up like a depressed elephant.

四个配置各跑「无上下文 3 遍 / 有上下文 3 遍」：
上下文固定喂 G7、G8 的官方译文，排除历史质量波动的干扰。
用法: python bench_g9.py
"""
import importlib.util, os, sys, json, time, subprocess, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MODELS_DIR = os.path.join(HERE, "models")

_spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "live_caption.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

G9 = ("I got the pills, Amanda. Jesus! I don't know how you get out of the "
      "fucking rack in the morning. They got you doped up like a depressed elephant.")
REF = "阿曼达，我帮你拿药了，天呐。真不知道你他妈早上是怎么起床的。他们给你开了麻醉一头忧郁症大象的剂量。"

# 固定上下文（G7 / G8 的官方译文），只对吃上下文的配置有意义
CTX = [
    ("Darling, you just stick that on the credit card. I gotta run.",
     "亲爱的，用信用卡付账吧。我得走了。"),
    ("Finally, feel like I can really make a difference in the world.",
     "我总算感觉到我能为这世界带来些影响了。"),
]

MODELS = [
    ("Hy-MT2-1.8B", "Hy-MT2-1.8B-Q4_K_M.gguf", "v3"),
    ("Index-2B", "Index-Translate-2B-Q4_K_M.gguf", "index"),
    ("Qwen3.8-4B-D", "Qwen3.8-4B-Q4_K_M.gguf", "qwen"),
    ("Qwen3.5-4B", "Qwen3.5-4B-Q4_K_M.gguf", "qwen"),
]
N = 3


def kill_all():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True, text=True)
    time.sleep(1.5)


def run(name, fn, style):
    path = os.path.join(MODELS_DIR, fn)
    if not os.path.exists(path):
        print(f"跳过 {name}（缺 {fn}）", flush=True)
        return None
    print("=" * 92)
    print(f"### {name}  style={style}", flush=True)
    print("=" * 92, flush=True)
    tr = lc.Translator(SRV, path, port=8803, ctx=2048, use_context=False,
                       timeout=90.0, max_tokens=320, style=style)
    tr.warmup()
    rec = {"style": style, "off": [], "on": [], "lat": []}

    for _ in range(N):
        t0 = time.time()
        zh = tr.translate(G9)
        rec["lat"].append(round((time.time() - t0) * 1000))
        rec["off"].append(zh)
        print(f"  [无ctx] {zh or '(空)'}", flush=True)

    # 固定上下文：直接写 hist，避免上一批的结果污染
    tr.use_context = True
    tr.multiturn = style == "qwen"
    tr.use_sys_ctx = style in ("v1", "v2", "v3")
    for _ in range(N):
        tr.hist = list(CTX)
        tr.prev = CTX[-1] if tr.multiturn else None
        t0 = time.time()
        zh = tr.translate(G9)
        rec["lat"].append(round((time.time() - t0) * 1000))
        rec["on"].append(zh)
        print(f"  [有ctx] {zh or '(空)'}", flush=True)

    tr.close()
    kill_all()
    return rec


def main():
    kill_all()
    res = {}
    for name, fn, style in MODELS:
        r = run(name, fn, style)
        if r:
            res[name] = r
    with open(os.path.join(TMP, "_bench_g9.json"), "w", encoding="utf-8") as f:
        json.dump({"src": G9, "ref": REF, "runs": res}, f, ensure_ascii=False, indent=1)

    print("\n\n" + "#" * 92)
    print("# G9 正确原文补测")
    print("#" * 92)
    print(f"原文: {G9}")
    print(f"参考: {REF}")
    for name, r in res.items():
        print(f"\n── {name} ({r['style']}) 延迟中位 {statistics.median(r['lat'])}ms")
        print(f"   无ctx×{N}: " + " | ".join(x or "(空)" for x in r["off"]))
        print(f"   有ctx×{N}: " + " | ".join(x or "(空)" for x in r["on"]))
    print("\n结果写入 _tmp/_bench_g9.json")


if __name__ == "__main__":
    main()
