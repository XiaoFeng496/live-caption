"""Qwen3.8 社区蒸馏版 (empero-ai) vs Qwen3.5 官方版 —— 同架构、同量化 Q4_K_M 对照。

三组语料：
  英语 34 句（口语/俚语/技术/长句/无标点/短句/注入）—— 关键词自动判定
  日语 15 句（泛化/惯用/时间/污染）—— 关键词自动判定
  用户真实素材 20 句（游戏语音 12 + 短视频 8）—— 存文本人工判

全部走 qwen 实机配置：TR_SYS few-shot + 上一句多轮 + 引号包裹，与 live_caption.py 一致。
开头先跑 thinking 探针：蒸馏版是推理模型，若 CoT 关不掉，实时字幕就用不了。
用法: python bench_38.py [--only 3.8]
"""
import importlib.util, os, sys, json, time, subprocess, urllib.request as ur, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MODELS_DIR = os.path.join(HERE, "models")
PORT = 8812
op = ur.build_opener(ur.ProxyHandler({}))

_spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "live_caption.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)
TR_SYS, TR_WRAP_QWEN = lc.TR_SYS, lc.TR_WRAP_QWEN
_clean_tr, _untranslated = lc._clean_tr, lc._untranslated

MODELS = [
    ("Qwen3.5-2B", "Qwen3.5-2B-Q4_K_M.gguf"),
    ("Qwen3.8-2B-D", "Qwen3.8-2B-Q4_K_M.gguf"),
    ("Qwen3.5-4B", "Qwen3.5-4B-Q4_K_M.gguf"),
    ("Qwen3.8-4B-D", "Qwen3.8-4B-Q4_K_M.gguf"),
]
if "--only" in sys.argv:
    key = sys.argv[sys.argv.index("--only") + 1]
    MODELS = [m for m in MODELS if key in m[0]]

# ---------------- 英语 34 句（原样取自 bench_en.py） ----------------
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

# ---------------- 日语 15 句（原样取自 bench_en.py） ----------------
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

# ---------------- 用户素材 20 句（与 bench_user20.py 一致） ----------------
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


def post(msgs, max_tokens, nothink=True, extra_user=""):
    body = {"messages": msgs, "max_tokens": max_tokens, "temperature": 0.0,
            "cache_prompt": True}
    if nothink:
        body["chat_template_kwargs"] = {"enable_thinking": False}
    t = time.time()
    req = ur.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                     data=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
    with op.open(req, timeout=180) as r:
        out = json.loads(r.read().decode())
    ms = (time.time() - t) * 1000
    raw = out["choices"][0]["message"]["content"]
    u = out.get("usage", {}) or {}
    return raw, ms, u.get("completion_tokens", 0), u.get("prompt_tokens", 0)


def probe(name):
    """蒸馏版是推理模型：确认 enable_thinking=False 是否真能关掉思维链"""
    print(f"\n-- thinking 探针 --")
    msgs = [{"role": "system", "content": TR_SYS},
            {"role": "user", "content": TR_WRAP_QWEN.format("The build is failing on the latest commit.")}]
    for label, kw in (("enable_thinking=False", True), ("默认(不加参数)", False)):
        raw, ms, ct, pt = post(msgs, 512, nothink=kw)
        head = raw.strip().replace("\n", " ")[:110]
        print(f"  [{label:22s}] out={ct:4d}tok  {ms:6.0f}ms  -> {head}")
    # Qwen3 系列的 /no_think 软指令
    m2 = [{"role": "system", "content": TR_SYS},
          {"role": "user", "content": TR_WRAP_QWEN.format("The build is failing on the latest commit.") + " /no_think"}]
    raw, ms, ct, pt = post(m2, 512, nothink=False)
    print(f"  [{'/no_think 软指令':22s}] out={ct:4d}tok  {ms:6.0f}ms  -> "
          f"{raw.strip().replace(chr(10),' ')[:110]}")


def translate(text, prev, note):
    msgs = [{"role": "system", "content": TR_SYS}]
    if prev:
        msgs.append({"role": "user", "content": prev[0]})
        msgs.append({"role": "assistant", "content": prev[1]})
    msgs.append({"role": "user", "content": TR_WRAP_QWEN.format(text)})
    raw, ms, ct, pt = post(msgs, 256, nothink=True)
    zh = _clean_tr(raw)
    return zh, ms, ct, raw


def judge(zh, want, forbid):
    if not want:
        return "?"
    if any(w in zh for w in want.split("|")):
        return "OK"
    if forbid and any(f in zh for f in forbid.split("|")):
        return "BAD"
    return "MISS"


def run_set(title, cases, tr_fn, with_ctx=True):
    """cases: (src, tag, want, forbid)"""
    st = {"ok": 0, "miss": 0, "bad": 0, "n": 0}
    lat, rows = [], []
    prev = None
    for src, tag, want, forbid in cases:
        zh, ms, ct, raw = tr_fn(src, prev)
        lat.append(ms)
        j = judge(zh, want, forbid)
        if want:
            st["n"] += 1
            st[{"OK": "ok", "MISS": "miss", "BAD": "bad"}[j]] += 1
        rows.append({"src": src, "tag": tag, "zh": zh, "ms": round(ms),
                     "tok": ct, "j": j})
        if zh and not _untranslated(zh, src) and not src.startswith(("Ignore", "Translate this", "You are now")):
            prev = (src, zh)
    st["lat_med"] = round(statistics.median(lat)) if lat else 0
    st["tok_med"] = round(statistics.median([r["tok"] for r in rows])) if rows else 0
    st["rows"] = rows
    return st


def main():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, text=True)
    time.sleep(1.5)
    results = {}
    for name, fn in MODELS:
        path = os.path.join(MODELS_DIR, fn)
        if not os.path.exists(path):
            print(f"跳过 {name}（缺文件）")
            continue
        print("\n" + "=" * 96)
        print(f"### {name}   ({os.path.getsize(path)/2**30:.2f} GiB)")
        print("=" * 96, flush=True)
        t0 = time.time()
        p = subprocess.Popen([SRV, "-m", path, "-ngl", "99", "--port", str(PORT), "-c", "4096",
                              "--no-webui", "--log-disable", "-np", "1"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if not wait(PORT, p):
            print("启动失败")
            p.terminate()
            continue
        load_s = time.time() - t0
        print(f"加载 {load_s:.1f}s")
        probe(name)
        # 预热
        for _ in range(2):
            post([{"role": "user", "content": TR_WRAP_QWEN.format("こんにちは")}], 64)

        rec = {"load_s": round(load_s, 1)}
        for title, cases in (("英语", EN), ("日语", JA)):
            print(f"\n-- {title} {len(cases)} 句 --", flush=True)
            s = run_set(title, cases, lambda t, pv: translate(t, pv, ""), True)
            for r in s["rows"]:
                print(f"  [{r['tag']:6s}][{r['j']:4s}] {r['zh'] or '(空)'}")
            rec[title] = s
            print(f"  => 命中 {s['ok']}/{s['n']}  错译 {s['bad']}  "
                  f"延迟中位 {s['lat_med']}ms  输出 {s['tok_med']}tok")

        print(f"\n-- 用户素材 20 句（带上下文） --", flush=True)
        s = run_set("用户", [(c[1], c[0], "", "") for c in USER20],
                    lambda t, pv: translate(t, pv, ""), True)
        for r in s["rows"]:
            print(f"  [{r['tag']:4s}] {r['zh'] or '(空)'}")
        rec["用户"] = s
        print(f"  => 延迟中位 {s['lat_med']}ms  输出 {s['tok_med']}tok")

        p.terminate()
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
        subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True, text=True)
        time.sleep(1)
        results[name] = rec

    with open(os.path.join(TMP, "_bench38.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    print("\n\n" + "#" * 96)
    print("# 汇总")
    print("#" * 96)
    print(f"{'模型':<14s}{'英语':>12s}{'日语':>12s}{'延迟中位':>10s}{'输出tok':>9s}")
    for n, r in results.items():
        e, j = r["英语"], r["日语"]
        print(f"{n:<14s}{e['ok']}/{e['n']} ({e['ok']/e['n']*100:4.0f}%)"
              f"{j['ok']:>5d}/{j['n']} ({j['ok']/j['n']*100:4.0f}%)"
              f"{e['lat_med']:>8d}ms{e['tok_med']:>8d}")
    print("\n结果已写入 _bench38.json")


if __name__ == "__main__":
    main()
