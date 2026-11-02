"""拆解三模型同时常驻时各自的显存占用，并测英语模型降 ctx 的收益。

步骤：ASR 单独起 -> 叠 Hy -> 叠 Index(ctx2048) -> 换成 Index(ctx512)
每一步用 nvidia-smi 按 pid 取该进程实占，同时记录整卡 free。
用法: python test_vram_split.py
"""
import os, sys, json, time, subprocess, urllib.request as ur

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
SRV = os.path.join(HERE, "llama", "llama-server.exe")
M = os.path.join(HERE, "models")
OP = ur.build_opener(ur.ProxyHandler({}))

ASR_M = os.path.join(M, "Qwen3-ASR-1.7B-q5_k.gguf")
ASR_P = os.path.join(M, "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf")
HY_M = os.path.join(M, "Hy-MT2-1.8B-Q4_K_M.gguf")
IDX_M = os.path.join(M, "Index-Translate-2B-Q4_K_M.gguf")


def vram():
    o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    u, f = o.stdout.strip().split(",")
    return int(u), int(f)


def proc_vram(pids):
    """按 pid 取 llama-server 占用的显存（MiB）"""
    o = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True)
    d = {}
    for line in o.stdout.strip().splitlines():
        try:
            pid, mem = line.split(",")
            d[int(pid.strip())] = int(mem.strip())
        except Exception:
            pass
    return {p: d.get(p, 0) for p in pids}


def wait(port, t=90):
    for _ in range(t * 4):
        try:
            with OP.open(f"http://127.0.0.1:{port}/health", timeout=2) as r:
                if r.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.25)
    return False


def start_asr(port=8770, ctx=4096, slots=2):
    cmd = [SRV, "-m", ASR_M, "--mmproj", ASR_P, "-ngl", "99", "--port", str(port),
           "-c", str(ctx), "--no-webui", "-np", str(slots), "--log-disable"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def start_tr(model, port, ctx):
    cmd = [SRV, "-m", model, "-ngl", "99", "--port", str(port), "-c", str(ctx),
           "--no-webui", "-np", "1", "--log-disable"]
    p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = wait(port)
    return p, ok


def kill(p):
    try:
        p.terminate(); p.wait(timeout=8)
    except Exception:
        try:
            p.kill()
        except Exception:
            pass
    time.sleep(1.2)


def main():
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)
    time.sleep(1.5)
    base, free0 = vram()
    print(f"桌面基线 {base} MiB，可用 {free0} MiB\n", flush=True)

    rec = {}
    procs = []

    # 1) ASR
    a = start_asr()
    procs.append(a)
    assert wait(8770), "ASR 起不来"
    time.sleep(1.0)
    u1, f1 = vram()
    rec["asr"] = {"delta": u1 - base, "pid": proc_vram([a.pid])}
    print(f"ASR 单独       整卡 {u1} (+{u1-base})  free {f1}  进程占 {proc_vram([a.pid])}", flush=True)

    # 2) + Hy-MT2-1.8B
    hy, ok = start_tr(HY_M, 8771, 2048)
    procs.append(hy)
    time.sleep(1.0)
    u2, f2 = vram()
    rec["hy"] = {"delta": u2 - u1, "pid": proc_vram([hy.pid])}
    print(f"+Hy-MT2-1.8B   整卡 {u2} (+{u2-u1})  free {f2}  进程占 {proc_vram([hy.pid])}", flush=True)

    # 3) + Index ctx2048
    ix, ok = start_tr(IDX_M, 8772, 2048)
    procs.append(ix)
    time.sleep(1.0)
    u3, f3 = vram()
    rec["idx2048"] = {"delta": u3 - u2, "pid": proc_vram([ix.pid])}
    print(f"+Index ctx2048 整卡 {u3} (+{u3-u2})  free {f3}  进程占 {proc_vram([ix.pid])}", flush=True)

    # 4) 换成 Index ctx512
    kill(ix)
    time.sleep(1.0)
    u4, f4 = vram()
    ix2, ok = start_tr(IDX_M, 8772, 512)
    procs.append(ix2)
    time.sleep(1.0)
    u5, f5 = vram()
    rec["idx512"] = {"total": u5, "free": f5, "pid": proc_vram([ix2.pid])}
    print(f"Index ctx512   整卡 {u5}  free {f5}  进程占 {proc_vram([ix2.pid])}"
          f"  (ctx2048->512 省 {u3-u4-(u5-u4)} MiB)", flush=True)

    for p in reversed(procs):
        kill(p)
    subprocess.run(["taskkill", "/F", "/IM", "llama-server.exe"], capture_output=True)

    json.dump(rec, open(os.path.join(TMP, "_vram_split.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=1)
    print("\n写入 _tmp/_vram_split.json")


if __name__ == "__main__":
    main()
