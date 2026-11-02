"""--glossary 的注入效果与副作用验证。

三件事：
1. 命中逻辑单测（词边界、大小写、多词短语、中日文子串）—— 不起模型
2. 同一模型同一提示词，加/不加术语表对照：目标句是否修好
3. 副作用：英语 34 + 日语 15 全量跑一遍，看注入后原有规则有没有被挤掉
用法: python bench_glossary.py
"""
import ast, importlib.util, os, re, sys, json, time, subprocess, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
HY = os.path.join(HERE, "models", "Hy-MT2-1.8B-Q4_K_M.gguf")

_spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "live_caption.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)


def extract(path, names):
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


TBL = [
    ("rack", "床"),
    ("doped up", "被灌药"),
    ("blood money", "血汗钱"),
    ("再来週", "下下周"),
    ("先月末", "上个月底"),
]

# (标签, 原文, 期望出现, 期望不出现)
TARGETS = [
    ("rack", "I got the pills, Amanda. Jesus! I don't know how you get out of the "
             "fucking rack in the morning. They got you doped up like a depressed elephant.",
     "床", "架子"),
    ("再来週", "再来週の打ち合わせは中止になった", "下下周", ""),
    ("先月末", "先月末に契約が切れた", "上个月底", ""),
    ("blood money", "Remember, baby, you're spending blood money.", "血汗钱", ""),
    # 不含任何术语的干扰句：不该被注入，也不该退化
    ("干扰-无命中", "I need to track the package before noon.", "包裹|快递", ""),
]


def unit_tests():
    print("=" * 90)
    print("1) 命中逻辑")
    print("=" * 90)
    cases = [
        ("the fucking rack in the morning", ["rack"], True),
        ("I need to track the package", [], False),
        ("mount the bracket", [], False),
        ("RACK it up", ["rack"], True),           # 大小写
        ("they got you doped up like", ["doped up"], True),
        ("doped", [], False),                      # 单词不该被 dop up 命中
        ("it's blood money", ["blood money"], True),
        ("再来週の打ち合わせ", ["再来週"], True),
        ("先月末に契約", ["先月末"], True),
        ("彼は顔が広い", [], False),
    ]
    ok = 0
    for text, want, expect in cases:
        got = [s for s, _ in lc.glossary_hits(text, TBL)]
        good = (bool(got) == expect) and all(w in got for w in want)
        ok += good
        print(f"  {'OK ' if good else 'BAD'} {text[:44]:<46} -> {got}")
    print(f"  单测 {ok}/{len(cases)}\n")
    return ok == len(cases)


def main():
    if not unit_tests():
        print("命中逻辑有问题，先停下。")
        return

    EN = extract(os.path.join(HERE, "bench_en.py"), {"EN", "JA"})
    EN, JA = EN["EN"], EN["JA"]

    print("=" * 90)
    print("2) 目标句：无术语表 vs 有术语表（Hy-MT2-1.8B + v3）")
    print("=" * 90)

    def run(tag, glossary):
        tr = lc.Translator(SRV, HY, port=8830, ctx=2048, use_context=True,
                           timeout=90.0, max_tokens=200, style="v3", glossary=glossary)
        tr.warmup()
        res = {"targets": [], "en": [], "ja": [], "lat": []}

        print(f"\n-- {tag} --", flush=True)
        for name, src, want, forbid in TARGETS:
            t0 = time.time()
            zh = tr.translate(src) or "(空)"
            res["lat"].append(round((time.time() - t0) * 1000))
            good = (any(w in zh for w in want.split("|"))) and not (
                forbid and any(f in zh for f in forbid.split("|")))
            res["targets"].append({"name": name, "zh": zh, "ok": good})
            print(f"  [{name:10s}|{'OK ' if good else 'BAD'}] {zh}", flush=True)

        print("  英语 34 …", flush=True)
        tr.hist, tr.prev = [], None
        for src, tg, want, forbid in EN:
            zh = tr.translate(src) or "(空)"
            j = lc_judge(zh, want, forbid)
            res["en"].append({"src": src, "zh": zh, "judge": j})
        print("  日语 15 …", flush=True)
        tr.hist, tr.prev = [], None
        for src, tg, want, forbid in JA:
            zh = tr.translate(src) or "(空)"
            j = lc_judge(zh, want, forbid)
            res["ja"].append({"src": src, "zh": zh, "judge": j})
        tr.close()
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
        time.sleep(1.2)
        return res

    a = run("无术语表（基线）", None)
    b = run("有术语表（5 条）", TBL)

    def acc(rows):
        return sum(1 for r in rows if r["judge"] == "OK")

    print("\n" + "=" * 90)
    print("3) 结果")
    print("=" * 90)
    print(f"{'目标句':<14}{'基线':<8}{'有术语表'}")
    for x, y in zip(a["targets"], b["targets"]):
        print(f"{x['name']:<14}{'OK' if x['ok'] else 'BAD':<8}{'OK' if y['ok'] else 'BAD'}")
        if not x["ok"] and y["ok"]:
            print(f"      ← 修好了：{y['zh']}")
        elif x["ok"] and not y["ok"]:
            print(f"      ← 退化了：{y['zh']}")
    print(f"\n英语 34  基线 {acc(a['en'])}/34 → 有术语表 {acc(b['en'])}/34")
    print(f"日语 15  基线 {acc(a['ja'])}/15 → 有术语表 {acc(b['ja'])}/15")
    print(f"延迟中位  基线 {statistics.median(a['lat']):.0f}ms → "
          f"有术语表 {statistics.median(b['lat']):.0f}ms")

    for key, label in (("en", "英语"), ("ja", "日语")):
        diff = [(x, y) for x, y in zip(a[key], b[key]) if x["judge"] != y["judge"]]
        if diff:
            print(f"\n{label}判分变化：")
            for x, y in diff:
                print(f"  [{x['judge']}→{y['judge']}] {x['src'][:56]}")
                print(f"      基线 {x['zh']}")
                print(f"      术语 {y['zh']}")

    json.dump({"base": a, "gloss": b}, open(os.path.join(TMP, "_bench_gloss.json"), "w",
              encoding="utf-8"), ensure_ascii=False, indent=1)


def lc_judge(zh, want, forbid):
    if not want:
        return "?"
    if any(w in zh for w in want.split("|")):
        return "OK"
    if forbid and any(f in zh for f in forbid.split("|")):
        return "BAD"
    return "MISS"


if __name__ == "__main__":
    main()
