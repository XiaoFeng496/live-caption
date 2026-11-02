"""交替 A/B：-c 8192 / 4096 / 2048（均 -np 2），A B C C B A 抵消基线漂移。"""
import os, sys, time, subprocess, statistics
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_ctx_cost as m

m.kill_all()
base, _ = m.vram()
print(f"空载 {base} MiB\n")

cfgs = {"8192": 8192, "4096": 4096, "2048": 2048}
seq = ["8192", "4096", "2048", "2048", "4096", "8192"]
got = {k: [] for k in cfgs}
for i, tag in enumerate(seq, 1):
    p = subprocess.Popen([m.SRV, "-m", m.ASR_M, "--mmproj", m.MMPROJ, "-ngl", "99",
                          "--port", str(m.PORT), "-c", str(cfgs[tag]), "-np", "2", "--no-webui"],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    m.wait(m.PORT, p)
    time.sleep(1.0)
    u, f = m.vram()
    got[tag].append(u - base)
    print(f"  第{i}次 -c {tag}  used {u} MiB  增量 {u-base} MiB  free {f}")
    p.terminate()
    try:
        p.wait(timeout=15)
    except Exception:
        p.kill()
    time.sleep(2.0)

m.kill_all()
print("\n--- 中位（每档 2 次） ---")
for k in ("8192", "4096", "2048"):
    print(f"  -c {k:5s}  {got[k]}  中位 {statistics.median(got[k]):.0f} MiB")
a, b, c = (statistics.median(got[k]) for k in ("8192", "4096", "2048"))
print(f"\n8192 -> 4096 省 {a-b:.0f} MiB   8192 -> 2048 省 {a-c:.0f} MiB")
