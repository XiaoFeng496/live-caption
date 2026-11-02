"""交替 A/B：在同一个 -c 下，-np 1 与 -np 2 到底差多少显存（消除基线漂移）。"""
import os, sys, time, subprocess, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_ctx_cost as m

audio = m.load_wav()
m.kill_all()
base, _ = m.vram()
print(f"空载 {base} MiB\n")

seq = [("np2", 2), ("np1", 1), ("np2", 2), ("np1", 1)]
got = {"np2": [], "np1": []}
for i, (tag, n) in enumerate(seq, 1):
    p = subprocess.Popen([m.SRV, "-m", m.ASR_M, "--mmproj", m.MMPROJ, "-ngl", "99",
                          "--port", str(m.PORT), "-c", "2048", "-np", str(n), "--no-webui"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    ok = m.wait(m.PORT, p)
    time.sleep(1.0)
    u, f = m.vram()
    got[tag].append(u - base)
    print(f"  第{i}次 {tag}  n_ctx_slot={2048//n:5d}  used {u} MiB  增量 {u-base} MiB")
    p.terminate()
    try:
        p.wait(timeout=15)
    except Exception:
        p.kill()
    time.sleep(2.0)

m.kill_all()
print()
for k, v in got.items():
    print(f"{k}: {v}  中位 {statistics.median(v):.0f} MiB")
print(f"\n差值（np1 - np2）= {statistics.median(got['np1']) - statistics.median(got['np2']):+.0f} MiB")
