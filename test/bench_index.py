"""Index-Translate-2B（B站 Index 团队，专业翻译模型）vs Hy-MT2-1.8B（现役）

三组语料：英语 34 / 日语 15 / 用户真实素材 20（与 bench_38.py 完全一致）。
  Hy  走现有实机配置 style=v3（日英分流 + 英语带两句译文上下文）
  Idx 走官方 prompt：单条 user、无 system、enable_thinking=False
  Idx+ 额外把最近两句译文拼进 user 开头（官方没有上下文机制，试能不能救）
用法: python bench_index.py
"""
import importlib.util, os, sys, json, time, subprocess, urllib.request as ur, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MODELS_DIR = os.path.join(HERE, "models")
PORT = 8814
op = ur.build_opener(ur.ProxyHandler({}))

_spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "live_caption.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)
_clean_tr, _untranslated = lc._clean_tr, lc._untranslated

HY = os.path.join(MODELS_DIR, "Hy-MT2-1.8B-Q4_K_M.gguf")
IDX = os.path.join(MODELS_DIR, "Index-Translate-2B-Q4_K_M.gguf")

# Index 官方标准 prompt（--source auto，省略源语言）
IDX_P = "请将以下文本翻译为简体中文，直接输出翻译结果，不要进行任何解释。\n\n{}"
IDX_CTX = ("请将以下文本翻译为简体中文，直接输出翻译结果，不要进行任何解释。\n"
           "上文译文（仅供保持术语、人称和语气一致，不要翻译也不要重复）：{}\n\n{}")

EN = [
    ("Hey, can you grab me a coffee while you're out?", "口语", "咖啡", ""),
    ("I'm not gonna lie, that was kind of a mess.", "口语", "说实话|老实说|不骗", ""),
    ("Wait, what? Say that again?", "口语", "再", ""),
    ("No way, you actually did it!", "口语", "不会吧|不可能|真的|居然", ""),
    ("Yeah, I'm down for that.", "口语", "可以|行|愿意|好啊|没问题|同意|参加", "下来|向下"),
    ("It's not a big deal, don't worry about it.", "口语", "没什么|没关系|大事", ""),
    ("I'm just gonna crash early tonight.", "口语", "睡|躺|休息", "崩溃"),
    ("Could you speak up a bit? I can't hear you.", "口语", "大声|听不|听不清", ""),
    ("That's a piece of cake.", "俚语", "小菜一碟|容易|简单|轻松|易如反掌|手到擒来", "蛋糕|一块"),
    ("He really dropped the ball on this one.", "俚语", "搞砸|失误|没做好|失职|掉链子", "球"),
    ("Let's call it a day.", "俚语", "今天|收工|结束|到此为止", "叫它一天"),
    ("I'm all ears.", "俚语", "洗耳恭听|听着|请说|耳朵", ""),
    ("It costs an arm and a leg.", "俚语", "贵|天价|不少钱|成本极高|巨资|大价钱|破费", "胳膊|腿"),
    ("Break a leg!", "俚语", "好运|顺利|成功|加油", "断腿|摔断"),
    ("You're pulling my leg.", "俚语", "开玩笑|骗|逗|耍", "拉|腿"),
    ("We need to get our act together.", "俚语", "振作|认真|准备|团结|齐心", "表演"),
    ("Can you push the fix to main and let the CI run?", "技术", "main|主分支|主干", ""),
    ("The build is failing on the latest commit, I'll take a look.", "技术", "构建|编译|打包", ""),
    ("Let's schedule a quick sync tomorrow morning.", "技术", "同步|对齐|会议|碰", ""),
    ("I think we should deprecate that endpoint.", "技术", "废弃|弃用|淘汰|停用", ""),
    ("Make sure the migration is backward compatible.", "技术", "兼容", ""),
    ("So basically we need to make sure the whole pipeline doesn't break when the network drops out.", "技术", "流程|管线|链路", ""),
    ("The build passed, but the tests are flaky.", "技术", "不稳定|偶发|时好时坏|不稳", "薄|片状"),
    ("If it turns out that the data wasn't actually saved, we're going to have to redo the whole thing from scratch.", "长句", "从头|重做|重新|重来", ""),
    ("The reason we didn't ship it last week was that nobody had tested the edge cases.", "长句", "边界|边缘|极端", ""),
    ("Despite what everyone expected, the launch went off without a hitch.", "长句", "顺利|没问题|一帆风顺", ""),
    ("so i was like okay fine whatever and then he just walked out", "无标点", "行吧|随便|算了|走出去|离开|走了", ""),
    ("um so the thing is we might not be able to finish it by friday", "无标点", "周五|星期五", ""),
    ("Really?", "短句", "真|是|吗", ""),
    ("Not bad.", "短句", "不错|还行|可以", ""),
    ("Fair enough.", "短句", "有道理|合理|可以|行|说得通", ""),
    ("Never mind.", "短句", "算了|没关系|当我没说|不用", ""),
    ("I could care less.", "其他", "不在乎|无所谓|不关心|不在意", ""),
    ("That's the last thing I want.", "其他", "最不|不想|不要|最不愿意", ""),
]

JA = [
    ("まるい月が空に浮かんでいる", "泛化-圆", "圆|满", "弯"),
    ("テーブルの上に丸い皿が三枚ある", "泛化-圆", "圆", "弯"),
    ("赤いりんごを二つ買ってきた", "泛化-具象", "红|苹果", ""),
    ("細長い箱に入っている", "泛化-具象", "细长|长条|狭长", "宽|扁"),
    ("猫の手も借りたいほど忙しい", "泛化-惯用", "忙|不可开交|分身", "猫"),
    ("彼は顔が広いから紹介してもらおう", "泛化-惯用", "人脉|关系|路子|面子|人缘", "脸宽|脸大"),
    ("その言葉は耳が痛い", "泛化-惯用", "扎心|刺耳|说中|难受|痛|心痛", "耳朵疼"),
    ("彼が足を引っ張っている", "泛化-惯用", "拖后腿|拖累|扯后腿", "拉脚|腿"),
    ("再来週の打ち合わせは中止になった", "泛化-时间", "下下周|下下个", "下周"),
    ("先月末に契約が切れた", "泛化-时间", "上个月|上月", "本月|下月底"),
    ("さすがですね、やっぱりプロは違います", "污染-褒义", "不愧是|果然|专业", "过分|离谱|不行|没道理"),
    ("さすがはベテランだ、手際がいい", "污染-褒义", "不愧是|果然|老手|熟练", "过分|离谱|不行"),
    ("さすがにそれはひどすぎる", "对照-贬义", "过分|离谱|太过", ""),
    ("ちょっと何言ってるかわかんない", "口语", "不懂|不明白|听不懂|什么", ""),
    ("それな、ほんとそれ", "口语", "就是|确实|对啊|同意", ""),
]

USER20 = [
    ("G1", "I don't only talk to the kids when I'm drunk. I don't think it gets kind of fuzzy."),
    ("G2", "You're never fucking good enough with those people."),
    ("G3", "Fuck you, you wasp pricks!"),
    ("G4a", "Remember, baby, you're spending blood money."),
    ("G4b", "My blood money."),
    ("G5", "Nothing like buying things to make you happy."),
    ("G6", "Everything hurts."),
    ("G7", "Darling, you just stick that on the credit card. I gotta run."),
    ("G8", "Finally, feel like I can really make a difference in the world."),
    ("G9", "I got the pills, Amanda. Jesus! I don't know how you get out of the fucking rack in the morning. They got you thoped up like at the pressed elephant."),
    ("G10", "Yeah, yeah. Just because the doctor's writing your script doesn't mean they're good for you."),
    ("G11", "That ain't an excuse, so love of my life. You write these cockroaches a check for enough money, they'll write you a prescription for anything."),
    ("V1", "Hey, how's it going, guys? I have one quick question, and then another one. That's two. Yeah, two."),
    ("V2", "No, it's they're short. Now that you're in the biggest game in history, currently."),
    ("V3", "Are you gonna work in any other video games later on?"),
    ("V4", "Or are you done for like ever?"),
    ("V5", "That's up to Steven Spielberg."),
    ("V6", "And second question."),
    ("V7", "Steven, can you curse at me? Like say Diego, go fuck yourself or something."),
    ("V8", "Go fuck yourself."),
]


def wait(port, proc, limit=240):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            with op.open(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.25)
    return False


def post(msgs, max_tokens=160):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0.0,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    req = ur.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                     data=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
    with op.open(req, timeout=180) as r:
        out = json.loads(r.read().decode())
    ms = (time.time() - t) * 1000
    return out["choices"][0]["message"]["content"], ms, (out.get("usage") or {}).get("completion_tokens", 0)


def judge(zh, want, forbid):
    if not want:
        return "?"
    if any(w in zh for w in want.split("|")):
        return "OK"
    if forbid and any(f in zh for f in forbid.split("|")):
        return "BAD"
    return "MISS"


def run(cases, fn):
    st = {"ok": 0, "miss": 0, "bad": 0, "n": 0}
    lat, rows = [], []
    for src, tag, want, forbid in cases:
        zh, ms, ct = fn(src)
        lat.append(ms)
        j = judge(zh, want, forbid)
        if want:
            st["n"] += 1
            st[{"OK": "ok", "MISS": "miss", "BAD": "bad"}[j]] += 1
        rows.append({"src": src, "tag": tag, "zh": zh, "j": j, "ms": round(ms)})
    return {"ok": st["ok"], "miss": st["miss"], "bad": st["bad"], "n": st["n"],
            "lat": round(statistics.median(lat)), "rows": rows}


def start(path):
    p = subprocess.Popen([SRV, "-m", path, "-ngl", "99", "--port", str(PORT), "-c", "4096",
                          "--no-webui", "--log-disable", "-np", "1"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return p if wait(PORT, p) else None


def stop(p):
    p.terminate()
    try:
        p.wait(timeout=15)
    except Exception:
        p.kill()
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1)


def main():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1.5)
    res = {}

    # ---------- Index-Translate-2B ----------
    print("=" * 92)
    print("### Index-Translate-2B（官方 prompt，单条 user）")
    print("=" * 92, flush=True)
    p = start(IDX)
    if not p:
        print("启动失败")
        return
    raw, ms, ct = post([{"role": "user", "content": IDX_P.format("The build is failing on the latest commit.")}], 512)
    print(f"探针 out={ct}tok {ms:.0f}ms -> {raw.strip()[:120]}\n")
    post([{"role": "user", "content": IDX_P.format("こんにちは")}], 64)

    idx = {}
    for name, cases in (("英语", EN), ("日语", JA)):
        print(f"-- {name} --", flush=True)
        r = run(cases, lambda s: post([{"role": "user", "content": IDX_P.format(s)}]))
        for x in r["rows"]:
            print(f"  [{x['tag']:6s}][{x['j']:4s}] {x['zh'] or '(空)'}")
        print(f"  => {r['ok']}/{r['n']}  错译 {r['bad']}  延迟 {r['lat']}ms")
        idx[name] = r

    # 用户素材：无上下文 / 带上下文两版
    print("\n-- 用户素材（无上下文）--", flush=True)
    r = run([(c[1], c[0], "", "") for c in USER20],
            lambda s: post([{"role": "user", "content": IDX_P.format(s)}]))
    for x in r["rows"]:
        print(f"  [{x['tag']:4s}] {x['zh'] or '(空)'}")
    idx["用户无ctx"] = r

    print("\n-- 用户素材（带前两句译文）--", flush=True)
    hist = []
    rows = []
    for cid, src in USER20:
        ctx = "「" + "」「".join(z for _, z in hist[-2:]) + "」" if hist else ""
        prompt = IDX_CTX.format(ctx, src) if ctx else IDX_P.format(src)
        zh, ms, ct = post([{"role": "user", "content": prompt}])
        zh = _clean_tr(zh)
        rows.append({"src": src, "tag": cid, "zh": zh, "j": "", "ms": round(ms)})
        print(f"  [{cid:4s}] {zh or '(空)'}")
        if zh and not _untranslated(zh, src):
            hist.append((src, zh))
    idx["用户带ctx"] = {"rows": rows, "lat": round(statistics.median([r["ms"] for r in rows])), "ok": 0, "n": 0, "bad": 0}
    print(f"  => 延迟 {idx['用户带ctx']['lat']}ms")
    stop(p)
    res["Index-Translate-2B"] = idx

    # ---------- Hy-MT2-1.8B 基线（v3 实机配置） ----------
    print("\n" + "=" * 92)
    print("### Hy-MT2-1.8B（v3，现役配置）")
    print("=" * 92, flush=True)
    hy = {}
    for ctx_on in (False, True):
        tag = "有ctx" if ctx_on else "无ctx"
        # Translator 自己起 server（8815），避免和 Index 的 8814 抢显存
        tr = lc.Translator(SRV, HY, port=8815, ctx=2048, use_context=ctx_on,
                           timeout=90.0, max_tokens=160, style="v3")
        tr.warmup()
        print(f"\n-- {tag} --", flush=True)
        hy[tag] = {}
        for name, cases in (("英语", EN), ("日语", JA)):
            st = {"ok": 0, "miss": 0, "bad": 0, "n": 0}
            rows0 = []
            for src, t2, want, forbid in cases:
                zh = tr.translate(src)
                j = judge(zh, want, forbid)
                if want:
                    st["n"] += 1
                    st[{"OK": "ok", "MISS": "miss", "BAD": "bad"}[j]] += 1
                rows0.append({"src": src, "zh": zh, "j": j})
            st["rows"] = rows0
            print(f"  {name}: {st['ok']}/{st['n']}  错译 {st['bad']}")
            hy[tag][name] = st
        rows0 = []
        for cid, src in USER20:
            zh = tr.translate(src)
            rows0.append({"src": src, "tag": cid, "zh": zh})
            print(f"  [{cid:4s}] {zh or '(空)'}")
        hy[tag]["用户"] = {"rows": rows0}
        tr.close()
        time.sleep(1)
    res["Hy-MT2-1.8B"] = hy

    with open(os.path.join(TMP, "_bench_index.json"), "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)

    print("\n\n" + "#" * 92)
    print("# 汇总（英语 34 / 日语 15，自动判定）")
    print("#" * 92)
    i = res["Index-Translate-2B"]
    for lbl, e, jq in (("Index 官方prompt", i["英语"], i["日语"]),
                       ("Hy 无ctx", res["Hy-MT2-1.8B"]["无ctx"]["英语"], res["Hy-MT2-1.8B"]["无ctx"]["日语"]),
                       ("Hy 有ctx", res["Hy-MT2-1.8B"]["有ctx"]["英语"], res["Hy-MT2-1.8B"]["有ctx"]["日语"])):
        print(f"{lbl:20s} 英语 {e['ok']:2d}/{e['n']} ({e['ok']/e['n']*100:3.0f}%)   "
              f"日语 {jq['ok']:2d}/{jq['n']} ({jq['ok']/jq['n']*100:3.0f}%)")
    print("\n结果已写入 _bench_index.json")


if __name__ == "__main__":
    main()
