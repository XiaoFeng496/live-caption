"""Qwen3.5-4B vs Qwen3.8-4B-Distill 细粒度对照（只留一个的决策依据）。

比 bench_38 更细的地方：
  1. 每句跑多遍（英语/日语 3 遍，真实素材 2 遍）—— temperature=0 仍会因 KV 复用抖动
  2. 按类别（口语/俚语/技术/长句/短句/无标点）分别统计，而不是只给总分
  3. 失败模式分类：空输出 / 英文泄漏 / 复读上一句 / 退化重复 / 拒答
  4. 延迟 p50/p90、输出 token 中位（啰嗦会拖慢字幕）
  5. 真实素材把每句的多遍输出全部留存，人工判分

用法: python bench_4b_detail.py
"""
import importlib.util, os, re, sys, json, time, subprocess, statistics
import urllib.request as ur

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MD = os.path.join(HERE, "models")
PORT = 8821
op = ur.build_opener(ur.ProxyHandler({}))

_spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "live_caption.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)
TR_SYS, TR_WRAP_QWEN = lc.TR_SYS, lc.TR_WRAP_QWEN
_clean_tr, _untranslated = lc._clean_tr, lc._untranslated

MODELS = [("Qwen3.5-4B", "Qwen3.5-4B-Q4_K_M.gguf"),
          ("Qwen3.8-4B-D", "Qwen3.8-4B-Q4_K_M.gguf")]

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

CJK = re.compile(r"[\u4e00-\u9fff\u3040-\u30ff]")
LATIN = re.compile(r"[A-Za-z]")
DEGEN = re.compile(r"(.{2,10})\1{2,}")


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()[0]
    return int(o)


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


def post(msgs, max_tokens=256):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0.0,
            "cache_prompt": True, "chat_template_kwargs": {"enable_thinking": False}}
    t = time.time()
    req = ur.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                     data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with op.open(req, timeout=180) as r:
        out = json.loads(r.read().decode())
    ms = (time.time() - t) * 1000
    raw = out["choices"][0]["message"]["content"]
    u = out.get("usage", {}) or {}
    return raw, ms, u.get("completion_tokens", 0)


def translate(text, prev):
    msgs = [{"role": "system", "content": TR_SYS}]
    if prev:
        msgs.append({"role": "user", "content": prev[0]})
        msgs.append({"role": "assistant", "content": prev[1]})
    msgs.append({"role": "user", "content": TR_WRAP_QWEN.format(text)})
    raw, ms, ct = post(msgs)
    return _clean_tr(raw), ms, ct


def flag(zh, src, prev_zh):
    """失败模式分类（与质量无关的那几类）"""
    f = []
    if not zh:
        f.append("空")
        return f
    if prev_zh and zh.strip() == prev_zh.strip():
        f.append("复读上句")
    if _untranslated(zh, src):
        f.append("英文泄漏")
    if DEGEN.search(zh):
        f.append("退化重复")
    if not CJK.search(zh):
        f.append("无中文")
    return f


def judge(zh, want, forbid):
    if not want:
        return "?"
    if any(w in zh for w in want.split("|")):
        return "OK"
    if forbid and any(f in zh for f in forbid.split("|")):
        return "BAD"
    return "MISS"


def run_cases(cases, passes, with_ctx=True):
    prev = None
    rows = []
    for c in cases:
        src, tag = c[0], c[1]
        want, forbid = (c[2], c[3]) if len(c) > 2 else ("", "")
        outs, lats, toks, js, flags = [], [], [], [], []
        for _ in range(passes):
            zh, ms, ct = translate(src, prev if with_ctx else None)
            outs.append(zh)
            lats.append(ms)
            toks.append(ct)
            js.append(judge(zh, want, forbid))
            flags.extend(flag(zh, src, prev[1] if prev else None))
        if with_ctx:
            ok_out = next((o for o in outs if o and not _untranslated(o, src)), None)
            if ok_out:
                prev = (src, ok_out)
        # 多数票：3 遍取 ≥2，2 遍取 ≥1（2 遍平票按 OK 记）
        ok = js.count("OK")
        rows.append({"src": src, "tag": tag, "want": want, "forbid": forbid,
                     "outs": outs, "js": js, "flags": sorted(set(flags)),
                     "ok_votes": ok, "ms": round(statistics.median(lats)),
                     "tok": round(statistics.median(toks)),
                     "stable": len(set(outs)) == 1})
    return rows


def summarize(rows):
    n = len(rows)
    ok_any = sum(1 for r in rows if r["ok_votes"] > 0)          # 至少一遍对
    ok_all = sum(1 for r in rows if r["ok_votes"] == len(r["js"]))  # 每遍都对
    stable = sum(1 for r in rows if r["stable"])
    bad = sum(1 for r in rows if "BAD" in r["js"])
    flags = {}
    for r in rows:
        for f in r["flags"]:
            flags[f] = flags.get(f, 0) + 1
    lat = [r["ms"] for r in rows]
    by_tag = {}
    for r in rows:
        t = r["tag"]
        d = by_tag.setdefault(t, [0, 0])
        d[0] += 1 if r["ok_votes"] > 0 else 0
        d[1] += 1
    return {"n": n, "ok_any": ok_any, "ok_all": ok_all, "stable": stable, "bad": bad,
            "flags": flags, "lat_med": round(statistics.median(lat)),
            "lat_p90": round(sorted(lat)[int(len(lat) * 0.9) - 1]),
            "by_tag": by_tag, "rows": rows}


def run_model(name, fn, base_vram):
    path = os.path.join(MD, fn)
    print("\n" + "=" * 100)
    print(f"### {name}  ({os.path.getsize(path)/2**30:.2f} GiB)")
    print("=" * 100, flush=True)
    t0 = time.time()
    p = subprocess.Popen([SRV, "-m", path, "-ngl", "99", "--port", str(PORT), "-c", "4096",
                          "--no-webui", "--log-disable", "-np", "1"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait(PORT, p):
        print("启动失败")
        p.terminate()
        return None
    load_s = time.time() - t0
    vram_mib = vram() - base_vram
    print(f"加载 {load_s:.1f}s  显存占用 {vram_mib} MiB", flush=True)

    # thinking 探针
    raw, ms, ct = post([{"role": "system", "content": TR_SYS},
                        {"role": "user", "content": TR_WRAP_QWEN.format("こんにちは")}], 512)
    print(f"thinking 探针：{ct} tok / {ms:.0f}ms -> {raw.strip()[:60]!r}")
    for _ in range(2):
        post([{"role": "user", "content": TR_WRAP_QWEN.format("こんにちは")}], 64)

    rec = {"load_s": round(load_s, 1), "vram": vram_mib, "probe_tok": ct}
    print("\n-- 英语 34 句 ×3 遍（带上下文） --", flush=True)
    rec["en"] = summarize(run_cases(EN, 3, True))
    print("\n-- 日语 15 句 ×3 遍（带上下文） --", flush=True)
    rec["ja"] = summarize(run_cases(JA, 3, True))
    print("\n-- 英语 34 句 ×2 遍（无上下文） --", flush=True)
    rec["en_noctx"] = summarize(run_cases(EN, 2, False))
    print("-- 真实素材 20 句 ×2 遍（带上下文） --", flush=True)
    rec["user"] = summarize(run_cases([(txt, k, "", "") for k, txt in USER20], 2, True))
    p.terminate()
    try:
        p.wait(timeout=15)
    except Exception:
        p.kill()
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1)
    return rec


def main():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1.5)
    base = vram()
    print(f"空载显存 {base} MiB")
    results = {}
    for name, fn in MODELS:
        if not os.path.exists(os.path.join(MD, fn)):
            print(f"缺少 {fn}")
            continue
        results[name] = run_model(name, fn, base)

    with open(os.path.join(TMP, "_bench_4b_detail.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)
    print("\n结果写入 _tmp/_bench_4b_detail.json")


if __name__ == "__main__":
    main()
