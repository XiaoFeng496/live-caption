"""实测：8GB 卡在「浏览器常驻 + ASR 常驻」的真实场景下，能否带得动 Hy-MT2-7B。

逐级加压，每级记录显存与翻译延迟：
  A 只起 ASR（基线）
  B ASR + 7B 全量上卡（-ngl 99, ctx 2048）
  C 若失败：ctx 降到 1024 + KV 量化 q8_0
  D 若还失败：部分层留 CPU（-ngl 24 / 18 / 12）
每级都跑同一句翻译，报延迟和显存。
"""
import os, sys, json, time, subprocess, urllib.request as ur, statistics

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MD = os.path.join(HERE, "models")
ASR_M = os.path.join(MD, "Qwen3-ASR-1.7B-q5_k.gguf")
MMPROJ = os.path.join(MD, "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
TR_M = os.path.join(MD, "Hy-MT2-7B-Q4_K_M.gguf")
ASR_PORT, TR_PORT = 8799, 8816
op = ur.build_opener(ur.ProxyHandler({}))

TEST = "Darling, you just stick that on the credit card. I gotta run."


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()[0]
    u, f = o.split(",")
    return int(u), int(f)


def kill_all():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(2)


def wait(port, proc, limit=180):
    t0 = time.time()
    while time.time() - t0 < limit:
        if proc.poll() is not None:
            return False
        try:
            with op.open(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            time.sleep(0.3)
    return False


def translate(port, text):
    body = {"messages": [{"role": "system", "content": "你是一位简体中文译者，把用户提供的文本翻译成简体中文，只输出译文。"},
                         {"role": "user", "content": "「" + text + "」"}],
            "max_tokens": 160, "temperature": 0.0, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = ur.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                     data=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
    t = time.time()
    with op.open(req, timeout=120) as r:
        out = json.loads(r.read().decode())
    return out["choices"][0]["message"]["content"].strip(), (time.time() - t) * 1000


def start_tr(extra, label):
    """返回 (proc, ok)。extra 是追加的 llama-server 参数"""
    cmd = [SRV, "-m", TR_M, "--port", str(TR_PORT),
           "--no-webui", "--log-disable", "-np", "1"] + extra
    p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    ok = wait(TR_PORT, p)
    return p, ok


def main():
    kill_all()
    base_used, base_free = vram()
    print("=" * 88)
    print(f"起点（浏览器/桌面已占用）：used {base_used} MiB, free {base_free} MiB")
    print(f"7B 权重文件：{os.path.getsize(TR_M)/2**30:.2f} GiB")
    print("=" * 88, flush=True)

    # A. 起 ASR
    pa = subprocess.Popen([SRV, "-m", ASR_M, "--mmproj", MMPROJ, "-ngl", "99",
                           "--port", str(ASR_PORT), "-c", "8192", "--no-webui",
                           "--log-disable", "-np", "2"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if not wait(ASR_PORT, pa):
        print("ASR 起不来，测试中止")
        return
    a_used, a_free = vram()
    print(f"\n[A] ASR 常驻（-c 8192 -np 2）：used {a_used} MiB, free {a_free} MiB "
          f"(ASR 占 {a_used-base_used} MiB)", flush=True)

    results = []
    trials = [
        ("B 7B 全量上卡 ctx2048", ["-ngl", "99", "-c", "2048"]),
        ("C ctx1024 + KV q8_0", ["-ngl", "99", "-c", "1024",
                                 "--cache-type-k", "q8_0", "--cache-type-v", "q8_0"]),
        ("D 部分卸载 -ngl 24", ["-ngl", "24", "-c", "2048"]),
        ("E 部分卸载 -ngl 16", ["-ngl", "16", "-c", "2048"]),
        ("F 部分卸载 -ngl 8", ["-ngl", "8", "-c", "2048"]),
    ]
    for label, extra in trials:
        print(f"\n---- {label} ----", flush=True)
        p, ok = start_tr(extra, label)
        if not ok:
            p.terminate()
            try:
                p.wait(timeout=10)
            except Exception:
                p.kill()
            time.sleep(1)
            u, f = vram()
            results.append({"label": label, "ok": False, "used": u, "free": f,
                            "lat": None, "txt": ""})
            print(f"  启动失败（显存不足）。当前 used {u} MiB, free {f} MiB")
            continue
        u, f = vram()
        # 预热 + 3 次计时
        try:
            translate(TR_PORT, TEST)
            lats, txt = [], ""
            for _ in range(3):
                txt, ms = translate(TR_PORT, TEST)
                lats.append(ms)
            med = statistics.median(lats)
            results.append({"label": label, "ok": True, "used": u, "free": f,
                            "lat": round(med), "txt": txt})
            print(f"  成功：used {u} MiB, free {f} MiB | 延迟中位 {med:.0f}ms")
            print(f"  译文：{txt}")
        except Exception as e:
            results.append({"label": label, "ok": False, "used": u, "free": f,
                            "lat": None, "txt": f"请求异常 {type(e).__name__}"})
            print(f"  请求异常：{type(e).__name__}")
        p.terminate()
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
        time.sleep(1.5)

    pa.terminate()
    try:
        pa.wait(timeout=10)
    except Exception:
        pa.kill()
    kill_all()

    print("\n" + "=" * 88)
    print("汇总（ASR 常驻前提下）")
    print("=" * 88)
    print(f"{'方案':<26s}{'结果':>8s}{'显存used':>10s}{'剩余':>8s}{'延迟':>9s}")
    for r in results:
        print(f"{r['label']:<26s}{'成功' if r['ok'] else '失败':>8s}"
              f"{r['used']:>9d}M{r['free']:>7d}M"
              f"{(str(r['lat'])+'ms') if r['lat'] else '-':>9s}")
    with open(os.path.join(TMP, "_vram7b.json"), "w", encoding="utf-8") as f:
        json.dump({"base": base_used, "asr": a_used, "trials": results}, f,
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
