"""补测：继续压 -c 但保留 -np 2，找显存与并发的平衡点。"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import test_ctx_cost as m

audio = m.load_wav()
m.kill_all()
base, _ = m.vram()
print(f"空载 {base} MiB")

out = []
for label, c, n in [("-c 3072 -np 2", 3072, 2), ("-c 2048 -np 2", 2048, 2)]:
    out.append(m.run_cfg(label, c, n, audio, [6.0], m.seg(audio, 30.0, 30.0), base))

print("\n### 补充汇总")
for r in out:
    if not r.get("ok"):
        print(r["label"], "启动失败"); continue
    print(f"{r['label']:22s} {r['cost']:6d}MiB  6s {r['rows'][0]['ms']}ms "
          f"并发墙钟 {r['wall']}ms | final {r['conc']['final6s']['ms']}ms "
          f"partial {r['conc']['partial2s']['ms']}ms")
