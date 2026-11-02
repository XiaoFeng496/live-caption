"""补测：ASR 降配能省多少显存，腾出来后 7B 全量上卡的余量与延迟稳定性。

现实意义：字幕是逐句串行送入，不需要 -np 2;每句音频几秒，不需要 -c 8192。
把 ASR 的 KV cache 压下来，7B 的 283 MiB 余量就可能变成 1G+。
"""
import os, sys, json, time, subprocess, statistics, urllib.request as ur

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MD = os.path.join(HERE, "models")
ASR_M = os.path.join(MD, "Qwen3-ASR-1.7B-q5_k.gguf")
MMPROJ = os.path.join(MD, "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
TR_M = os.path.join(MD, "Hy-MT2-7B-Q4_K_M.gguf")
PORT = 8817
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


def tr_once(text):
    body = {"messages": [{"role": "system", "content": "你是一位简体中文译者，把用户提供的文本翻译成简体中文，只输出译文。"},
                         {"role": "user", "content": "「" + text + "」"}],
            "max_tokens": 160, "temperature": 0.0, "cache_prompt": True,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = ur.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                     data=json.dumps(body).encode(),
                     headers={"Content-Type": "application/json"})
    t = time.time()
    with op.open(req, timeout=120) as r:
        out = json.loads(r.read().decode())
    return out["choices"][0]["message"]["content"].strip(), (time.time() - t) * 1000


def main():
    kill_all()
    base, _ = vram()
    print(f"起点 used {base} MiB\n")

    # ---- 1. ASR 三档配置的占用 ----
    cfgs = [("当前 -c 8192 -np 2", ["-c", "8192", "-np", "2"]),
            ("-c 4096 -np 1", ["-c", "4096", "-np", "1"]),
            ("-c 2048 -np 1", ["-c", "2048", "-np", "1"])]
    asr_cost = {}
    for label, extra in cfgs:
        p = subprocess.Popen([SRV, "-m", ASR_M, "--mmproj", MMPROJ, "-ngl", "99",
                              "--port", "8799", "--no-webui", "--log-disable"] + extra,
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ok = wait(8799, p)
        u, f = vram()
        asr_cost[label] = u - base
        print(f"ASR {label:22s} -> used {u} MiB (占 {u-base} MiB), free {f}")
        p.terminate()
        try:
            p.wait(timeout=10)
        except Exception:
            p.kill()
        time.sleep(1.5)

    # ---- 2. 最优 ASR 配置 + 7B 全量，测延迟抖动 ----
    print("\n---- ASR(-c 2048 -np 1) + 7B(-ngl 99 -c 2048) 全量上卡，跑 10 次 ----")
    pa = subprocess.Popen([SRV, "-m", ASR_M, "--mmproj", MMPROJ, "-ngl", "99",
                           "--port", "8799", "-c", "2048", "-np", "1",
                           "--no-webui", "--log-disable"],
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    wait(8799, pa)
    a_used, _ = vram()
    p = subprocess.Popen([SRV, "-m", TR_M, "--port", str(PORT), "-c", "2048",
                          "-ngl", "99", "--no-webui", "--log-disable", "-np", "1"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = wait(PORT, p)
    u, free_mib = vram()
    print(f"7B 启动{'成功' if ok else '失败'}：used {u} MiB, free {free_mib} MiB "
          f"(ASR 占 {a_used-base}, 7B 占 {u-a_used})")
    lats, txt = [], ""
    if ok:
        tr_once(TEST)
        for i in range(10):
            txt, ms = tr_once(TEST)
            lats.append(ms)
        s = sorted(lats)
        print(f"延迟 10 次：min {s[0]:.0f}  中位 {statistics.median(lats):.0f}  "
              f"max {s[-1]:.0f}  抖动 {s[-1]-s[0]:.0f}ms")
        print(f"译文：{txt}")
        p.terminate()
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
    pa.terminate()
    try:
        pa.wait(timeout=10)
    except Exception:
        pa.kill()
    kill_all()

    print("\n--- ASR 配置占用对比 ---")
    for k, v in asr_cost.items():
        print(f"  {k:24s} {v} MiB")
    with open(os.path.join(TMP, "_vram7b2.json"), "w", encoding="utf-8") as fh:
        json.dump({"asr": asr_cost, "lat": lats, "used": u, "free": free_mib}, fh,
                  ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
