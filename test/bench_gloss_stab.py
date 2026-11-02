"""术语表稳定性 + 跨风格行为验证。

1. Hy-1.8B + v3：3 个目标句各 5 遍，看命中率与污染（韩文/假名/拉丁残留）
2. Index + index：验证「拉丁词条无效、中日文预替换有效」这个断言
3. Hy 官方模板 + hymt2：同上
用法: python bench_gloss_stab.py
"""
import ast, importlib.util, os, re, sys, json, time, subprocess, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MD = os.path.join(HERE, "models")

_spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "live_caption.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

TBL = [("rack", "床"), ("doped up", "被灌药"), ("blood money", "血汗钱"),
       ("再来週", "下下周"), ("先月末", "上个月底")]

CASES = [
    ("rack", "I got the pills, Amanda. Jesus! I don't know how you get out of the "
             "fucking rack in the morning. They got you doped up like a depressed elephant.",
     "床", "架子"),
    ("再来週", "再来週の打ち合わせは中止になった", "下下周", ""),
    ("先月末", "先月末に契約が切れた", "上个月底", ""),
]

HANGUL = re.compile(r"[\u1100-\u11ff\uac00-\ud7af]")
KANA = re.compile(r"[\u3040-\u30ff]")


def kill():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1.2)


def start(model, style, port, gloss):
    kill()
    tr = lc.Translator(SRV, model, port=port, ctx=2048, use_context=True,
                       timeout=90.0, max_tokens=200, style=style, glossary=gloss)
    tr.warmup()
    return tr


def probe(tr, n=5, label=""):
    print(f"\n-- {label or tr.style} --", flush=True)
    rows = []
    for name, src, want, forbid in CASES:
        outs = []
        for _ in range(n):
            tr.hist, tr.prev = [], None
            outs.append(tr.translate(src) or "(空)")
        ok = sum(1 for z in outs if want in z and not (forbid and forbid in z))
        hang = sum(1 for z in outs if HANGUL.search(z))
        kana = sum(1 for z in outs if KANA.search(z) and not KANA.search(src))
        rows.append({"name": name, "ok": ok, "hangul": hang, "kana": kana,
                     "uniq": list(dict.fromkeys(outs))})
        print(f"  [{name:6s}] 命中 {ok}/{n}  韩文 {hang}/{n}  假名溢出 {kana}/{n}", flush=True)
        for z in dict.fromkeys(outs):
            flag = "  <韩文>" if HANGUL.search(z) else ("  <假名>" if KANA.search(z) and not KANA.search(src) else "")
            print(f"      {z}{flag}", flush=True)
    return rows


def main():
    out = {}
    hy = start(os.path.join(MD, "Hy-MT2-1.8B-Q4_K_M.gguf"), "v3", 8870, TBL)
    out["hy_v3"] = probe(hy, 5, "Hy-1.8B + v3（基准配置）")
    hy.close()

    ix = start(os.path.join(MD, "Index-Translate-2B-Q4_K_M.gguf"), "index", 8871, TBL)
    out["index"] = probe(ix, 3, "Index-2B + index（无 system，只有预替换能生效）")
    ix.close()

    hm = start(os.path.join(MD, "Hy-MT2-1.8B-Q4_K_M.gguf"), "hymt2", 8872, TBL)
    out["hymt2"] = probe(hm, 3, "Hy 官方模板 hymt2（同上）")
    hm.close()

    kill()
    json.dump(out, open(os.path.join(TMP, "_bench_gloss_stab.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
