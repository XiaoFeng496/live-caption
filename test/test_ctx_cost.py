"""验证 ASR 的 -c / -np 降配：真的省显存吗？有没有隐藏代价？

三组对照：
  A  -c 8192 -np 2   现状（每槽 4096）
  B  -c 4096 -np 2   每槽 2048，保留双槽并行  ← 候选
  C  -c 2048 -np 1   每槽 2048，单槽          ← 上一轮的建议

测四件事：
  1. 启动后真实显存占用，以及 llama.cpp 自己报的 KV/计算缓冲区大小
  2. 音频时长 -> prompt token 数（决定每句需要多少上下文）
  3. 溢出行为：超长音频在 2048 槽里是报错、静默截断，还是正常
  4. 双线程并发（final + partial 同时送）时，-np 1 是否把临时句排到最终句后面
"""
import os, sys, json, time, wave, base64, io, threading, subprocess, statistics
import urllib.request as ur
import numpy as np

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
MD = os.path.join(HERE, "models")
ASR_M = os.path.join(MD, "Qwen3-ASR-1.7B-q5_k.gguf")
MMPROJ = os.path.join(MD, "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
WAV = os.path.join(TMP, "_tts_src.wav")
PORT = 8831
op = ur.build_opener(ur.ProxyHandler({}))


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                        "--format=csv,noheader,nounits"],
                       capture_output=True, text=True).stdout.strip().splitlines()[0]
    u, f = o.split(",")
    return int(u), int(f)


def kill_all():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(2)


def load_wav():
    with wave.open(WAV, "rb") as w:
        sr = w.getframerate()
        a = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).astype(np.float32) / 32768.0
    assert sr == 16000, sr
    return a


def seg(a, sec, offset=0.0):
    i0 = int(offset * 16000)
    return a[i0:i0 + int(sec * 16000)]


def wav_b64(a):
    b = io.BytesIO()
    with wave.open(b, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())
    return base64.b64encode(b.getvalue()).decode()


def send(audio, timeout=120):
    """与 LlamaBackend.transcribe 完全同构的请求；返回 (text, usage, ms, truncated, err)"""
    msgs = [{"role": "system", "content": "热门词：信用卡 处方 血汗钱"},
            {"role": "user", "content": [{"type": "input_audio",
                                          "input_audio": {"data": wav_b64(audio), "format": "wav"}}]}]
    body = json.dumps({"messages": msgs, "max_tokens": 200, "temperature": 0.0,
                       "cache_prompt": True}).encode()
    req = ur.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions", data=body,
                     headers={"Content-Type": "application/json"})
    t = time.time()
    try:
        with op.open(req, timeout=timeout) as r:
            out = json.loads(r.read().decode())
        ms = (time.time() - t) * 1000
        ch = out["choices"][0]
        return (ch["message"]["content"].strip(), out.get("usage", {}),
                ms, ch.get("truncated"), None)
    except Exception as e:
        return "", {}, (time.time() - t) * 1000, None, f"{type(e).__name__}: {e}"


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


def parse_log(path):
    """从 llama.cpp 启动日志里抠出 KV / 计算缓冲区 / 槽位分配"""
    keys = ("n_ctx", "n_parallel", "n_ctx_per_seq", "KV self size", "kv cache",
            "compute buffer", "graph splits", "model buffer")
    out = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                low = line.lower()
                if any(k.lower() in low for k in keys):
                    out.append(line.strip())
    except OSError:
        pass
    return out[:14]


def run_cfg(label, cflag, npflag, audio, durs, long_audio, base):
    log = os.path.join(TMP, f"_asrlog_{npflag}_{cflag}.txt")
    with open(log, "w", encoding="utf-8") as lf:
        p = subprocess.Popen([SRV, "-m", ASR_M, "--mmproj", MMPROJ, "-ngl", "99",
                              "--port", str(PORT), "-c", str(cflag), "-np", str(npflag),
                              "--no-webui"],
                             stdout=lf, stderr=lf)
        ok = wait(PORT, p)
        u, f = vram()
        cost = u - base
        print(f"\n{'='*88}\n### {label}   (启动{'成功' if ok else '失败'})")
        print(f"显存：used {u} MiB，比空载多 {cost} MiB，free {f} MiB")
        for ln in parse_log(log):
            print("   ", ln)

        if not ok:
            p.kill()
            return {"label": label, "cost": cost, "ok": False}

        # warmup（首个请求要建图）
        send(seg(audio, 1.0))
        rows = []
        for d in durs:
            a = seg(audio, d, 20.0)
            txt, us, ms, trunc, err = send(a)
            pt = us.get("prompt_tokens")
            ct = us.get("completion_tokens")
            rows.append({"dur": d, "pt": pt, "ct": ct, "ms": round(ms),
                         "text": txt, "trunc": trunc, "err": err})
            print(f"  {d:5.1f}s  prompt {str(pt):>5} tok  out {str(ct):>4}  "
                  f"{ms:6.0f}ms  {'' if not err else err}")
            print(f"          {txt[:64]}")
            time.sleep(0.2)

        # 超长音频：看 2048 槽会不会溢出
        txt, us, ms, trunc, err = send(long_audio)
        print(f"  超长 {len(long_audio)/16000:5.1f}s  prompt {us.get('prompt_tokens')} tok  "
              f"{ms:6.0f}ms  truncated={trunc}  {'' if not err else err}")
        print(f"          {txt[:80]}")
        long_row = {"dur": len(long_audio) / 16000, "pt": us.get("prompt_tokens"),
                    "ms": round(ms), "text": txt, "trunc": trunc, "err": err}

        # 并发：模拟 worker_final(长句) 与 worker_partial(短句) 同时到达
        a_fin, a_par = seg(audio, 6.0, 20.0), seg(audio, 2.0, 40.0)
        res = {}

        def job(name, a):
            txt, us, ms, trunc, err = send(a)
            res[name] = {"ms": round(ms), "pt": us.get("prompt_tokens"), "text": txt, "err": err}

        t0 = time.time()
        th1 = threading.Thread(target=job, args=("final6s", a_fin))
        th2 = threading.Thread(target=job, args=("partial2s", a_par))
        th1.start(); time.sleep(0.02); th2.start()
        th1.join(); th2.join()
        wall = (time.time() - t0) * 1000
        print(f"  并发：总墙钟 {wall:.0f}ms | final6s {res['final6s']['ms']}ms | "
              f"partial2s {res['partial2s']['ms']}ms")

        # 串行对照：同样的两段依次发，算并发相对串行的节省
        s1 = send(a_fin)[2]
        s2 = send(a_par)[2]
        print(f"  串行对照：final6s {s1:.0f}ms + partial2s {s2:.0f}ms = {s1+s2:.0f}ms")

        p.terminate()
        try:
            p.wait(timeout=15)
        except Exception:
            p.kill()
        time.sleep(1.5)
        return {"label": label, "cost": cost, "ok": True, "rows": rows,
                "long": long_row, "conc": res, "wall": round(wall),
                "serial": [round(s1), round(s2)], "u": u, "free": f}


def main():
    kill_all()
    base, _ = vram()
    print(f"空载显存 used {base} MiB")

    audio = load_wav()
    print(f"测试音频 {len(audio)/16000:.1f}s")

    durs = [1.0, 2.0, 4.0, 6.0, 10.0]
    long_audio = seg(audio, 30.0, 30.0)      # 30s，远超 max_seg=6s

    cfgs = [("A 现状 -c 8192 -np 2", 8192, 2),
            ("B 候选 -c 4096 -np 2", 4096, 2),
            ("C 上轮建议 -c 2048 -np 1", 2048, 1)]
    out = []
    for label, c, n in cfgs:
        out.append(run_cfg(label, c, n, audio, durs, long_audio, base))

    kill_all()
    with open(os.path.join(TMP, "_ctx_cost.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=1)

    print(f"\n{'='*88}\n### 汇总")
    print(f"{'配置':26s} {'显存':>8s} {'1s':>6s} {'6s':>6s} {'6s token':>9s} {'并发墙钟':>9s} {'串行合计':>9s}")
    for r in out:
        if not r.get("ok"):
            print(f"{r['label']:26s} 启动失败")
            continue
        d = {x["dur"]: x for x in r["rows"]}
        print(f"{r['label']:26s} {r['cost']:6d}MiB {d[1.0]['ms']:5d}ms {d[6.0]['ms']:5d}ms "
              f"{str(d[6.0]['pt']):>9s} {r['wall']:7d}ms {sum(r['serial']):7d}ms")


if __name__ == "__main__":
    main()
