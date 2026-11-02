"""合成一段英文测试语音（pyttsx3 -> SAPI5），供 ASR 上下文/显存测试复用。"""
import os, wave, numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TMP = os.path.join(HERE, "_tmp")
os.makedirs(TMP, exist_ok=True)
OUT = os.path.join(TMP, "_tts_src.wav")

TXT = (
    "亲爱的，你把那个刷到信用卡上就行了，我得赶紧走了。"
    "听着，医生给你开处方的时候，你要跟他说实话，到底有多疼。"
    "我那笔血汗钱全花在这台手术上了，我可不想白白浪费。"
    "没有什么比用没有的钱买不需要的东西更让人开心了。"
    "祝你今晚好运。你在开玩笑吗？那可是史上规模最大的一场比赛，你居然要错过。"
    "我全都听着呢，你跟我说说昨天会上到底发生了什么。"
    "她说截止日期改到下下周了，可谁也没通知设计组。"
    "说实话，我觉得这次干脆从头再来，把它做简单一点。"
    "天气越来越冷了，出门记得带件外套。"
    "他打开盒子，发现里面有一把生锈的钥匙，用一张一九九二年的旧报纸包着。"
    "火车又晚点了，我们就坐在站台上看雨。"
    "要我说，冬天来之前把那辆旧车卖掉算了。"
    "经理拿着文件走进来的时候，屋里所有人都安静了。"
    "就是那一刻，我知道整个计划要完蛋了。"
)


def read_wav(path):
    with wave.open(path, "rb") as w:
        sr, ch, sw, n = w.getframerate(), w.getnchannels(), w.getsampwidth(), w.getnframes()
        raw = w.readframes(n)
    if sw != 2:
        raise RuntimeError(f"只支持 16bit，当前 {sw*8}bit")
    a = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if ch > 1:
        a = a.reshape(-1, ch).mean(axis=1)
    return a, sr


def resample(a, sr_in, sr_out=16000):
    if sr_in == sr_out:
        return a
    n_out = int(round(len(a) * sr_out / sr_in))
    x = np.linspace(0, len(a) - 1, n_out)
    return np.interp(x, np.arange(len(a)), a).astype(np.float32)


def write_wav(path, a, sr=16000):
    with wave.open(path, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
        w.writeframes((np.clip(a, -1, 1) * 32767).astype(np.int16).tobytes())


def main():
    import pyttsx3
    tmp = os.path.join(TMP, "_tts_raw.wav")
    eng = pyttsx3.init()
    for v in eng.getProperty("voices"):
        if "en" in (v.id or "").lower() or "english" in (v.name or "").lower():
            eng.setProperty("voice", v.id)
            print("voice:", v.name)
            break
    eng.setProperty("rate", 165)
    eng.save_to_file(TXT, tmp)
    eng.runAndWait()
    for _ in range(40):
        if os.path.isfile(tmp) and os.path.getsize(tmp) > 1000:
            import time; time.sleep(0.3)
            if os.path.getsize(tmp) == os.path.getsize(tmp):
                break
        import time; time.sleep(0.3)
    a, sr = read_wav(tmp)
    a = resample(a, sr, 16000)
    write_wav(OUT, a)
    print(f"源 {sr}Hz {len(a)/16000:.1f}s -> {OUT}（16k 单声道）")
    try:
        os.remove(tmp)
    except OSError:
        pass


if __name__ == "__main__":
    main()
