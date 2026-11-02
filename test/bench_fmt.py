"""术语表注入格式对照：找出不污染（不出韩文/乱码）且能钉住译法的写法。

现象：`再来週 = 下下周` 这种等号式写法，1.8B 有时会输出「下下周의会议」——韩语助词。
这里对 5 种注入格式各跑 5 遍，统计：
  - 目标译法是否命中（下下周 / 上个月底 / 床）
  - 是否出现韩文、日文假名残留、其它异常字符
用法: python bench_fmt.py
"""
import importlib.util, os, re, sys, time, subprocess, json, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
SRV = os.path.join(HERE, "llama", "llama-server.exe")
HY = os.path.join(HERE, "models", "Hy-MT2-1.8B-Q4_K_M.gguf")

_spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "live_caption.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

# 每个格式是一个函数 (src, dst) -> 单行文本
FORMATS = {
    "A 等号":      lambda s, d: f"{s} = {d}",
    "B 箭头":      lambda s, d: f"{s} → {d}",
    "C 引号+译作": lambda s, d: f"「{s}」译作「{d}」",
    "D 破折号":    lambda s, d: f"- {s}：{d}",
    "E 带语种":    lambda s, d: f"日语「{s}」必须译为中文「{d}」",
}

HDR = "术语表（本句出现下列词时，必须按此译法，优先于以上所有规则）："

CASES = [
    ("再来週", "再来週の打ち合わせは中止になった", "下下周", ""),
    ("先月末", "先月末に契約が切れた", "上个月底", ""),
    ("rack", "They got you doped up like a depressed elephant, and you can't even "
             "get out of the rack in the morning.", "床", "架子"),
]
TERMS = {
    "再来週": "下下周",
    "先月末": "上个月底",
    "rack": "床",
    "doped up": "被灌药",
}

KANA = re.compile(r"[\u3040-\u30ff]")
HANGUL = re.compile(r"[\u1100-\u11ff\uac00-\ud7af]")
CJK = re.compile(r"[\u4e00-\u9fff]")


def main():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1.2)
    tr = lc.Translator(SRV, HY, port=8840, ctx=2048, use_context=False,
                       timeout=90.0, max_tokens=200, style="v3")
    tr.warmup()

    # 用 monkeypatch 控制注入格式（正常代码里 _glossary_block 固定用等号式）
    cur = [FORMATS["A 等号"]]

    def block(hits):
        if not hits:
            return ""
        return "\n\n" + HDR + "\n" + "\n".join(cur[0](s, d) for s, d in hits)

    lc._glossary_block = block
    table = [(k, v) for k, v in TERMS.items()]

    out = {}
    for fname, fmt in FORMATS.items():
        cur[0] = fmt
        print("=" * 88)
        print(f"### {fname}", flush=True)
        rec = {"cases": []}
        for key, src, want, forbid in CASES:
            tr.glossary = table
            outs = []
            for _ in range(5):
                zh = tr.translate(src) or "(空)"
                outs.append(zh)
            good = sum(1 for z in outs if want in z and not (forbid and forbid in z))
            hang = sum(1 for z in outs if HANGUL.search(z))
            kana = sum(1 for z in outs if KANA.search(z))
            rec["cases"].append({"key": key, "outs": outs, "good": good,
                                 "hangul": hang, "kana": kana})
            print(f"  [{key}] 命中 {good}/5  韩文 {hang}/5  假名 {kana}/5", flush=True)
            for z in dict.fromkeys(outs):
                mark = ""
                if HANGUL.search(z):
                    mark = "  <韩文>"
                elif KANA.search(z):
                    mark = "  <假名>"
                print(f"     {z}{mark}", flush=True)
        out[fname] = rec
        tr.glossary = []

    tr.close()
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)

    print("\n" + "=" * 88)
    print(f"{'格式':<12}{'命中':<10}{'韩文':<8}{'假名':<8}")
    for fname, rec in out.items():
        g = sum(c["good"] for c in rec["cases"])
        h = sum(c["hangul"] for c in rec["cases"])
        k = sum(c["kana"] for c in rec["cases"])
        print(f"{fname:<12}{g}/{len(CASES)*5:<9}{h:<8}{k:<8}")

    json.dump(out, open(os.path.join(TMP, "_bench_fmt.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
