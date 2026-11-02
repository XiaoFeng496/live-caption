"""删模型前的最终定论：同口径跑 5 个候选。

语料：英语 34（含俚语/技术/长句/注入）+ 日语 15（泛化与污染）+ 真实素材 20（G9 用
还原后的正确英文）。全部用各模型自己的最优 style，带上下文（Index 除外，它不能带）。

判分：英语/日语走关键词（先认对再认错），真实素材打印原文供人工复核。
另外记录显存占用与延迟分布，用来判断「多花的显存值不值」。
用法: python bench_final.py
"""
import ast, importlib.util, os, sys, json, time, subprocess, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MDIR = os.path.join(HERE, "models")
DEL = os.path.join(MDIR, "准备删的")

G9_OK = ("I got the pills, Amanda. Jesus! I don't know how you get out of the fucking rack "
         "in the morning. They got you doped up like a depressed elephant.")

MODELS = [
    ("Hy-1.8B", os.path.join(MDIR, "Hy-MT2-1.8B-Q4_K_M.gguf"), "v3"),
    ("Index-2B", os.path.join(MDIR, "Index-Translate-2B-Q4_K_M.gguf"), "index"),
    ("Qwen3.5-2B", os.path.join(DEL, "Qwen3.5-2B-Q4_K_M.gguf"), "qwen"),
    ("Qwen3.8-2B", os.path.join(DEL, "Qwen3.8-2B-Q4_K_M.gguf"), "qwen"),
    ("Hy-7B", os.path.join(DEL, "Hy-MT2-7B-Q4_K_M.gguf"), "v3"),
]


def extract(path, names):
    """从脚本源码静态取出顶层列表常量，不执行文件"""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            t = getattr(node.targets[0], "id", None)
            if t in names:
                try:
                    out[t] = ast.literal_eval(node.value)
                except Exception:
                    pass
    return out


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    u, f = o.stdout.strip().split(",")
    return int(u), int(f)


def judge(zh, want, forbid):
    if not want:
        return "?"
    if any(w in zh for w in want.split("|")):
        return "OK"
    if forbid and any(f in zh for f in forbid.split("|")):
        return "BAD"
    return "MISS"


def main():
    en = extract(os.path.join(HERE, "bench_en.py"), {"EN", "JA"})
    EN, JA = en["EN"], en["JA"]
    u = extract(os.path.join(HERE, "bench_user20.py"), {"CASES"})["CASES"]
    USER = [(c[0], G9_OK if c[0] == "G9" else c[1], c[2], c[3]) for c in u]
    print(f"语料：英语 {len(EN)} + 日语 {len(JA)} + 真实素材 {len(USER)}\n", flush=True)

    _spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "live_caption.py"))
    lc = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(lc)

    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1.5)
    base, _ = vram()
    print(f"桌面基线 {base} MiB\n", flush=True)

    res = {}
    for name, path, style in MODELS:
        if not os.path.exists(path):
            print(f"跳过 {name}：缺 {path}", flush=True)
            continue
        print("=" * 96)
        print(f"### {name}   style={style}", flush=True)
        t0 = time.time()
        tr = lc.Translator(SRV, path, port=8820, ctx=2048, use_context=True,
                           timeout=120.0, max_tokens=200, style=style)
        load_s = round(time.time() - t0, 1)
        tr.warmup()
        tr.warmup()
        time.sleep(0.8)
        used, free = vram()
        rec = {"style": style, "load_s": load_s, "vram_delta": used - base,
               "vram_total": used, "free": free, "en": [], "ja": [], "user": [],
               "lat": {"en": [], "ja": [], "user": []}}
        print(f"加载 {load_s}s  显存 +{used-base} MiB（整卡 {used}）", flush=True)

        def batch(cases, key, judged=True):
            tr.hist = []          # 每批之间清空，批内保持连续（模拟真实句流）
            tr.prev = None
            for c in cases:
                cid, src = (c[0], c[1]) if key == "user" else (None, c[0])
                tag = c[1] if key == "user" else c[1]
                try:
                    t1 = time.time()
                    zh = tr.translate(src)
                    ms = (time.time() - t1) * 1000
                except Exception as e:
                    zh, ms = f"<{type(e).__name__}>", 0
                zh = zh or "(空)"
                rec["lat"][key].append(round(ms))
                row = {"id": cid, "src": src, "zh": zh, "ms": round(ms)}
                if judged:
                    want, forbid = c[2], c[3]
                    row["judge"] = judge(zh, want, forbid)
                    row["want"] = want
                rec[key].append(row)
                m = row.get("judge", "")
                print(f"  [{cid or tag[:4]}|{m:4s}] {zh}", flush=True)

        batch(EN, "en")
        batch(JA, "ja")
        batch(USER, "user", judged=False)
        tr.close()
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        time.sleep(1.2)
        res[name] = rec

        for k in ("en", "ja"):
            rows = rec[k]
            ok = sum(1 for r in rows if r["judge"] == "OK")
            bad = sum(1 for r in rows if r["judge"] == "BAD")
            rec[f"{k}_ok"], rec[f"{k}_bad"] = ok, bad
            print(f"  >> {k}: {ok}/{len(rows)} = {ok/len(rows)*100:.0f}%  明显错译 {bad}  "
                  f"延迟中位 {statistics.median(rec['lat'][k]):.0f}ms", flush=True)

    json.dump(res, open(os.path.join(TMP, "_bench_final.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)

    print("\n\n" + "#" * 96)
    print("# 汇总")
    print("#" * 96)
    hdr = f"{'模型':<12}{'英语':>10}{'日语':>10}{'加载':>8}{'显存+':>9}{'整卡':>8}{'剩余':>8}{'延迟中位':>10}"
    print(hdr)
    for n, r in res.items():
        lat = statistics.median(r["lat"]["en"] + r["lat"]["ja"])
        print(f"{n:<12}{r['en_ok']:>7}/{len(r['en']):<2}"
              f"{r['ja_ok']:>8}/{len(r['ja']):<2}"
              f"{r['load_s']:>7}s{r['vram_delta']:>8}M{r['vram_total']:>7}M{r['free']:>7}M"
              f"{lat:>8.0f}ms")
    print("\n真实素材 20 句（原文 + 各模型译文）已写入 _tmp/_bench_final.json")


if __name__ == "__main__":
    main()
