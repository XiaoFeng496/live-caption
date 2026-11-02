"""用真实素材（桌面两份 txt）的 20 句，对照测 3 个模型。

每句都按实机配置跑两组：
  无上下文 —— 单句隔离，看模型本身的语感
  有上下文 —— 连续句流，hist 自动累积（Hy 挂 system 最近两句译文，Qwen 走多轮）
Hy-MT2 用 style=v3（日英分流，英语带上下文），Qwen3.5 用 style=qwen。
用法: python bench_user20.py
"""
import importlib.util, os, sys, json, time, subprocess, urllib.request as ur, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MODELS_DIR = os.path.join(HERE, "models")

_spec = importlib.util.spec_from_file_location("lc", os.path.join(HERE, "live_caption.py"))
lc = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(lc)

MODELS = [
    ("Hy-MT2-1.8B", "Hy-MT2-1.8B-Q4_K_M.gguf", "v3"),
    ("Qwen3.5-2B", "Qwen3.5-2B-Q4_K_M.gguf", "qwen"),
    ("Qwen3.5-4B", "Qwen3.5-4B-Q4_K_M.gguf", "qwen"),
]

# (id, 原文, 官方参考译文(空=无), 判分要点)
CASES = [
    ("G1", "I don't only talk to the kids when I'm drunk. I don't think it gets kind of fuzzy.",
     "我不是只有在喝醉时跟孩子们说话。我觉得这不是。我有点混乱了。",
     "后半句原文含混，官方也在猜；宽松"),
    ("G2", "You're never fucking good enough with those people.",
     "你他妈和那些人永远相处不好。",
     "关键：never...good enough 是「永远处不好」而不是「不够好」；脏话强度要保留"),
    ("G3", "Fuck you, you wasp pricks!",
     "",
     "wasp 可能指 WASP（新教徒）骂人话；看是否硬译成「黄蜂」"),
    ("G4a", "Remember, baby, you're spending blood money.",
     "宝贝，记住，你花的可都是血汗钱。我的血汗钱。",
     "blood money 在这个语境=血汗钱/血钱，不是「血腥钱」"),
    ("G4b", "My blood money.",
     "",
     "承接上句，能不能保持一致"),
    ("G5", "Nothing like buying things to make you happy.",
     "没什么能比买东西更让你开心了。",
     "Nothing like 是「没有什么比…更」的感叹，别译成「没有像…那样的事情」"),
    ("G6", "Everything hurts.",
     "一切都不顺。",
     "官方是语境意译；字面「一切都疼/处处都疼」也可以接受，判宽松"),
    ("G7", "Darling, you just stick that on the credit card. I gotta run.",
     "亲爱的，用信用卡付账吧。我得走了。",
     "stick on the credit card = 刷/记信用卡账，不是「贴在信用卡上」"),
    ("G8", "Finally, feel like I can really make a difference in the world.",
     "我总算感觉到我能为这世界带来些影响了。",
     "make a difference = 带来改变/影响，不是「在世界上」"),
    ("G9", "I got the pills, Amanda. Jesus! I don't know how you get out of the fucking rack in the morning. They got you thoped up like at the pressed elephant.",
     "阿曼达，我帮你拿药了，天呐。真不知道你他妈早上是怎么起床的。他们给你开了麻醉一头忧郁症大象的剂量。",
     "原文 ASR 已错（doped up like a depressed elephant）；此句只记录不判分"),
    ("G10", "Yeah, yeah. Just because the doctor's writing your script doesn't mean they're good for you.",
     "对，对，医生给你开了张药方，并不表示吃这些药就会好。",
     "script = 处方，不是「剧本」"),
    ("G11", "That ain't an excuse, so love of my life. You write these cockroaches a check for enough money, they'll write you a prescription for anything.",
     "我不是在找借口，亲爱的。只要你给这些蟑螂的钱够多，他们什么药都敢开。",
     "so love of my life 是 ASR 断错，能接成「亲爱的」算加分"),
    ("V1", "Hey, how's it going, guys? I have one quick question, and then another one. That's two. Yeah, two.",
     "",
     "口语流畅度，That's two 别译成奇怪的「那是二」"),
    ("V2", "No, it's they're short. Now that you're in the biggest game in history, currently.",
     "",
     "这是游戏访谈，game=游戏（史上最牛的游戏），不是「比赛」；前半句原文已残"),
    ("V3", "Are you gonna work in any other video games later on?",
     "",
     "video games=电子游戏/游戏"),
    ("V4", "Or are you done for like ever?",
     "",
     "done for ever = 就此彻底不干了，别译成「你已经永远完了」"),
    ("V5", "That's up to Steven Spielberg.",
     "",
     "人名保留"),
    ("V6", "And second question.",
     "",
     "短句，看会不会加戏"),
    ("V7", "Steven, can you curse at me? Like say Diego, go fuck yourself or something.",
     "",
     "curse at me = 骂我；go fuck yourself 要保留脏话强度，别净化成「去你的」"),
    ("V8", "Go fuck yourself.",
     "",
     "同上，裸句测试脏话强度"),
]


def kill_all():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"],
                   capture_output=True, text=True)
    time.sleep(1.5)


def run_model(name, fn, style):
    path = os.path.join(MODELS_DIR, fn)
    if not os.path.exists(path):
        print(f"跳过 {name}（缺文件）")
        return None
    print("=" * 96)
    print(f"### {name}  style={style}")
    print("=" * 96, flush=True)
    t0 = time.time()
    tr = lc.Translator(SRV, path, port=8802, ctx=2048, use_context=False,
                       timeout=90.0, max_tokens=160, style=style)
    load_s = time.time() - t0
    tr.warmup()
    rec = {"style": style, "load_s": round(load_s, 1), "off": {}, "on": {}, "lat": {}}

    def batch(tag):
        out, lat = {}, []
        for cid, src, ref, note in CASES:
            try:
                t1 = time.time()
                zh = tr.translate(src)
                ms = (time.time() - t1) * 1000
                out[cid] = zh
                if zh:
                    lat.append(ms)
            except Exception as e:
                out[cid] = f"<异常 {type(e).__name__}>"
            print(f"  [{cid}] {out[cid] or '(空)'}", flush=True)
        rec["lat"][tag] = round(statistics.median(lat), 0) if lat else 0
        return out

    print(f"\n-- 无上下文 --", flush=True)
    rec["off"] = batch("off")
    # 切到实机配置：开上下文，清干净历史
    tr.use_context = True
    tr.multiturn = style == "qwen"
    tr.use_sys_ctx = style in ("v1", "v2", "v3")
    tr.warmup()
    print(f"\n-- 有上下文（实机配置）--", flush=True)
    rec["on"] = batch("on")
    tr.close()
    kill_all()
    return rec


def main():
    kill_all()
    results = {}
    for name, fn, style in MODELS:
        r = run_model(name, fn, style)
        if r:
            results[name] = r

    with open(os.path.join(TMP, "_bench_user20.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=1)

    names = list(results)
    for tag, label in (("off", "无上下文"), ("on", "有上下文")):
        print("\n\n" + "#" * 96)
        print(f"# 按句对照 —— {label}")
        print("#" * 96)
        for cid, src, ref, note in CASES:
            print(f"\n── {cid} ── {src}")
            if ref:
                print(f"   官方 : {ref}")
            print(f"   要点 : {note}")
            for n in names:
                print(f"   {n[-3:]:>4s} : {results[n][tag].get(cid) or '(空)'}")
    print("\n结果已写入 _bench_user20.json")


if __name__ == "__main__":
    main()
