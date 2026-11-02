# live-caption — 本地显卡实时字幕 / Local-GPU Realtime Captions 自用改版

**把电脑正在播放的任何声音，实时变成屏幕底部的字幕。全程本地显卡运行，不联网、不花钱。**
直播、会议、视频、网课、游戏解说……只要电脑在出声，就能出字。中文识别质量为主要目标，英文及中英混说亦可。

![overlay](docs/overlay.png)

- **延迟约半秒**：从一句话说完到字幕上屏，实测平均 0.54s（RTX 3090）
- **中文准确率**：Qwen3-ASR-1.7B，WenetSpeech 字错率 4.97%（whisper-large-v3 为 9.86%）；口语、快语速、游戏解说明显好于 whisper
- **不用虚拟声卡**：WASAPI loopback 直接抓系统输出，不需要 VB-Cable / 立体声混音
- **像视频字幕**：屏幕底部悬浮条，最新一句大字，上一句缩小变灰，没人说话自动渐隐；可拖动、滚轮调字号
- **热词**：把人名、队名、术语写进 `hotwords.txt`，专有名词识别率大幅提升
- **显存约 3GB**，显卡占用约 13%，可以一边打游戏 / 跑别的一边用

> English summary at the bottom.

## 与原版的区别

本仓库是 [TerryKiddy/live-caption](https://github.com/TerryKiddy/live-caption) 的自用改版：识别引擎、WASAPI 采集、悬浮窗骨架都来自原版，**原有命令行参数全部保留**，改动集中在下面这几处。

| | 原版 | 本改版 |
|---|---|---|
| 翻译 | 没有 | 独立翻译小模型（Hy-MT2-1.8B / Index-Translate-2B / Qwen3.8-4B，自动挑 models 里现存的那个），悬浮窗原文 + 译文双行，`--translate` 打开 |
| 语言 | `language X<asr_text>` 塞在提示词里，事后按标签过滤 | 改成 **末位 assistant 前缀**强制；`--lang` 严格丢句，新增 `--lang-hint`（只钉不丢） |
| 切句 | 纯 Silero VAD 决定这一窗要不要 | 起句「能量门控 or VAD」保召回，收句优先 VAD、命中率不够才回退能量；另加 AGC / pre-roll / 碎片地板 / 垃圾句过滤 |
| 悬浮窗 | 整窗一起 `-alpha`（字和黑底一起变淡），文字走 Tk 的 ClearType | 双层窗口：文字层抠底后**永远不透明**，只有独立黑底板受 `--alpha` 控制，**Ctrl+滚轮**实时调；文字改用 Pillow 灰度 AA + 描边渲染 |
| 安装 | venv + pip | 有 `uv` 就走 uv（秒级），没有自动回退 venv + pip |

具体差异：

- **翻译**：`--tr-partial` 让临时识别结果也带译文——一句话说完之前就能看到大意（原版没有翻译；只有 final 才有译文时，一行最短要活满 `--max-seg` 的 6 秒）；`glossary.txt` 钉人名/术语；`--tr-pair` 让原文译文成对冻结；`--tr-model-en` 按书写系统把英语句子分给另一个模型。
- **语言钉定**：原版在提示词里写 `language Japanese<asr_text>`，热词里一掺英文（LPL / CUDA / PyTorch）日语就被拽成英语。改成 assistant 前缀后，同一段素材实测日语 11/11 句稳定。
- **切句**：原版纯 VAD 在真实游戏/直播混音上只认 20% 时长（64.6 秒里 41 秒人声从没进过模型，表现就是「女声像没听见、连字都不出」）；但只用能量门控又会把几个短句粘成 6 秒硬切。现在双判据各管一头：240 秒混音里 ≥3 秒的长句从 20 句降到 11 句，而动漫女声仍然是完整句子。细节见「为什么改判据」和「为什么短句会被粘长」。
- **去复读**：`--context` 会把上一句顺下来并逐轮累积（第 7 段变成前 3 句拼接），改成识别后剥掉与上一句重合的前缀（`_strip_echo`）。
- **悬浮窗透明度**：原版是 `root.attributes("-alpha", 0.82)`，黑底和字一起变淡，降到 0.5 以下字已经开始发灰、还要跟桌面抢清晰度。现在拆成两层——文字层用 `-transparentcolor` 把底色抠成透明（键色 `#010203`，故意不用纯黑，免得文字抗锯齿边缘被一起抠掉），黑底板是独立窗口，**只有它受 `--alpha` 控制**：Ctrl+滚轮实时调 0~100%（步长 5%，0 = 只留文字、没有黑底），文字本身恒定不透明。
- **字体渲染**：文字不再交给 Tk 标签画，改用 Pillow 按 `msyhbd.ttc`（粗）/ `msyh.ttc`（常规）渲染成图再贴上去——Tk 标签走 GDI ClearType，字缘是次像素彩色像素，背景被抠掉后这些彩边会裸露成「忽有忽无的伪描边」。Pillow 是灰度 AA + 统一黑色描边（约字号的 4.5%，随机 1~3px），换行也自己按像素宽度算（英文/数字不断词，中文逐字，超行数加省略号）。没装 Pillow 或字体缺失时自动退回原生文字渲染，不会崩。
- **启动脚本**：原版 `start.bat` / `start-zh.bat` 保留，另加一组 `start-local-*.bat`（中文 / 全语言 / 日语翻译 / 英语翻译 / 日英混合翻译 / 纯 VAD 对照），全部纯 ASCII + CRLF，避免中文 bat 乱码和双击闪退。

想回到接近原版的行为：`--seg-mode vad --no-context`，不传 `--translate` 即不翻译。

## 怎么跑（Windows）

需要：NVIDIA 显卡（6GB+ 显存）、Python 3.10+。

1. 下载本仓库，双击 `setup.bat`：创建虚拟环境、装依赖、下载模型（约 2.8GB，国内可先 `set HF_ENDPOINT=https://hf-mirror.com`）。
   检测到 `uv` 时优先走 `uv venv` + `uv pip install`（秒级完成），没有 `uv` 则自动回退 `python -m venv` + `pip`。注意 uv 建的 venv 默认**不含 pip**，以后装包用 `uv pip install <pkg>`（想装回 pip 约 6MB：`uv pip install pip`；`uv venv --seed` 在 0.13.6 上实测无效）。
2. 下载 [llama.cpp 的 Windows CUDA 构建](https://github.com/ggml-org/llama.cpp/releases)（`llama-bXXXX-bin-win-cuda-13.x-x64.zip` 和同名 `cudart-...zip`），两个都解压到本仓库的 `llama\` 目录里，使 `llama\llama-server.exe` 存在。
   **机器上已经有 llama.cpp、且 `llama-server` 在 PATH 里的话可以跳过这步**：启动脚本在 `llama\` 不存在时会自动从 PATH 里取（也可以直接 `set LLAMA_SERVER=完整路径` 或用 `--llama-server` 指定），省下这 1GB。
   **必须是 2026-04-12 之后的版本**（Qwen3-ASR 支持在 [PR #19441](https://github.com/ggml-org/llama.cpp/pull/19441) 合并），b10941 实测可用。
3. 双击 `start-zh.bat`（中文）或 `start.bat`（自动检测语言）。字幕条出现在屏幕底部，终端同时打印，`transcript.txt` 追加保存。

字幕窗：鼠标拖动移动；滚轮调字号；右键或 Esc 退出。

## 常用参数

```
start.bat --lang zh                 中文（推荐；语言写进 assistant 前缀强制定语言，判成别的语言的段落如音乐会被丢掉）
start.bat --lang ja                 日语（同上，已写进 start-local-日语翻译.bat）
start.bat --lang-hint ja            只钉语言、不丢句（日语为主、偶尔有英语台词时用）
start.bat                           自动检测（中英混说）
start.bat --list                    列出输出设备；--device N 抓指定设备（默认跟随系统默认输出）
start.bat --console                 不开悬浮窗，只在终端显示
start.bat --srt out.srt             同时写 srt
start.bat --font-size 40 --bottom 120 --alpha 0.7 --width 0.7
start.bat --hotwords "A,B,C"        临时热词（也可写在 hotwords.txt）
start.bat --debug debug_dir         留存每段 wav + log.jsonl（时长/推理耗时/延迟/文本），调参用
```

| 参数 | 默认 | 说明 |
|---|---|---|
| `--silence` | 0.5 | 静音多少秒算一句结束。这是延迟下限，小了断句更碎 |
| `--max-seg` | 6 | 一句最长几秒强制切（决定一行字数，约 20~35 字） |
| `--min-seg` | 1.0 | 短于此秒数先不切，并入下一段 |
| `--partial-every` | 0.4 | 说话中多久刷新一次临时结果 |
| `--vad-win` | 0.2 | 判定窗口 |
| `--seg-mode` | both | 切句判据。`both`=**起句**用「能量 or VAD」保召回、**收句**优先听 VAD（默认）；`energy`=只看能量（旧行为，会把几个短句粘成一句）；`vad`=只信 Silero VAD（起句召回差）。见下方「为什么改判据」 |
| `--vad-trust` | 0.30 | `--seg-mode both` 下，VAD 近 2 秒命中率低于此值就认定它对这个音色不可靠，收句回退到能量。实测游戏混音 0.34~0.40、动漫女声 0.24。`0`=永远听 VAD，`1`=永远听能量 |
| `--energy-mult` | 3.0 | 能量门限 = 噪声底 × 这个倍数。漏句就调小（2.0），BGM 段落出垃圾就调大（4~5） |
| `--vad-threshold` | 0.45 | Silero VAD 阈值，只影响 VAD 那一票 |
| `--noise-horizon` | 30 | 噪声底取最近多少秒。太短会被持续人声抬上去 |
| `--preroll` | 0.6 | 开句时倒回多少秒补句首（0=关）。气声/渐强的起音原本会被整段扔掉 |
| `--lang` | 自动 | 语言代码 `zh`/`ja`/`en`…。写进 **assistant 前缀**强制定语言，并把语言判不符的句子丢掉。单一语种一定要设，见「提高准确率」第 2 条 |
| `--lang-hint` | — | 只钉语言、不丢句（判不符时保留文本）。日语为主、偶尔有英语台词时用 |
| `--asr-ctx` | 4096 | ASR 的 llama-server 总上下文，会被 `--slots` 均分。6 秒句只吃约 117 token，4096 已够；显存紧可降到 2048（实测省 647 MiB，译文逐字不变） |
| `--slots` | 2 | 并行槽位数。最终句和临时句各占一个，**别降到 1**（实测只省 8 MiB，却让临时句排队 +129ms） |
| `--hold` | 6 | 没新话几秒后渐隐 |
| `--glossary` | glossary.txt | 术语表，每行「原文=译文」。只在句子命中时才生效。拉丁词条注入 system（最多 5 条）；中日文词条直接预替换原文。`--tr-style v3` 效果最全 |
| `--model` / `--mmproj` | models/…1.7B… | 换 [Qwen3-ASR-0.6B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-0.6B-GGUF) 更省显存 |
| `--tr-partial` | 1.0 | 说话过程中也翻译临时结果，最多每 1.0 秒一次（0=关）。断句后的终稿译文会覆盖它。只影响悬浮窗的译文行，`--console` 不打印 |
| `--tr-ctx` | 2048 | 翻译 server 的上下文长度（含挂在 system 里的最近两句译文）。显存紧可以降，但要留够「两句译文 + 当前句」 |
| `--no-tr-context` | 上下文默认开 | 关掉译文上下文。省约 20ms，但「血钱→血汗钱」「script→处方」这类靠上文才能定下来的词会退回字面译法 |
| `--tr-port` | 8802 | 翻译 server 的端口。和 ASR 的 `--port`（8765）分开，两个模型各跑一个 llama-server |

### 为什么一句话最长 6 秒

句子只在三种情况结束：静音够长（`--silence`）、说到 `--max-seg` 秒被强制切、或程序退出。
实测 14 分钟实录（作者本地的一次性分析脚本，未入库）：250 段、段长中位 2.6 秒、**没有超过 4.4 秒不出句的空档**，
但有 5 处「连续硬切」，最狠一段连续 4 句共 23.8 秒——那 24 秒里每 6 秒才换一行。

嫌换行慢有两个旋钮，二选一或同时用：

- `--max-seg 4`：最长等待从 6 秒压到 4 秒，代价是更多句子被切在词中间。
- `--tr-partial 1.0`（默认已开）：让译文跟着临时结果每约 1 秒刷新，断句节奏不变。

注意这两者都改不了「说话人本身不停顿」这个事实：旧版每 1.6 秒断一次并不是听出了停顿，
而是 Silero VAD 在真实混音上大量误判为静音（见下方「为什么改判据」），属于随机切。

## 它是怎么做到半秒的

```
系统音频 (WASAPI loopback) ─► 自动增益 ─► 能量门控 + VAD 切句（主线程）
        │                                          │
        │                            ┌─────────────┴──────────────┐
        │                     线程 A：最终句，按序排队        线程 B：临时句，只留最新一份
        │                            └─────────────┬──────────────┘
        │                                   llama-server（2 槽位并行）
        │                                   Qwen3-ASR-1.7B Q8_0 GGUF
        └──────────────────────────────────────────┴──► 悬浮字幕条 / 终端 / txt / srt
```

### 为什么改判据（2026-11 实测）

旧版用 Silero VAD（阈值 0.45）单独决定"这一窗要不要"，在一段 64.6 秒的真实动漫/直播混音上：

| 判据 | 送进识别的时长 | 保留的窗 | 结果 |
|---|---|---|---|
| 旧：纯 VAD 0.45 | **12.8s / 64.6s = 20%** | 17% | 17 句变 6 句，大量 0.2s 碎片（`Missile.`、`Dang!`） |
| VAD 降到 0.25 | 36.3s = 56% | 29% | 开始出现完整句，仍漏 |
| **新默认：能量 or VAD** | **57.0s = 88%** | 86% | 11 段，句子完整，垃圾 1/11 |

VAD 逐帧概率中位数只有 0.099，一半以上的帧概率低于 0.1——**41 秒（56%）的人声被判成静音，从头到尾没进过模型**。女声、气声、句尾弱读的帧概率天然更低，所以体感是"女声识别差、有时连字都不出"，其实是"凡是 VAD 没认出来的都扔了，而女声更容易没被认出来"。VAD 本身对性别没有偏好：同一段文本用男声/女声合成，VAD 命中率 82% vs 84%，ASR 输出逐字相同。

现在的做法：

- **起句：能量 OR VAD**（`--energy-mult` 门限 = 滚动噪声底 × 倍数）。噪声底取最近 30 秒各窗 RMS 的低分位，可以瞬间下降、只允许缓慢上升（避免持续讲话把本底抬到人声电平）。只要有一票说是语音就开句——漏掉开头是整句没了，代价不对称
- **收句：优先听 VAD**（可信时），这是 2026-11-01 补的第二半，见下一节
- **pre-roll**：开句时倒回 0.6 秒补句首，起音不再被削
- **碎片地板**：静音切分要求句子至少 0.4 秒，否则继续攒——0.2 秒碎片喂给模型就会吐出 `Missile.` 这种单字误判
- **结果过滤**：纯标点/纯音乐的输出（`♪♪♪`、`What? What? What?`）直接丢弃

### 为什么短句会被粘长（2026-11-01 修）

能量门控只回答「这一窗有没有声音」，不回答「这句话说完了没」。BGM、脚步声、枪声垫底时，停顿处的 RMS 依然高于「噪声底 × 3」，`--silence` 那个条件永远不成立，于是句子一路攒到 `--max-seg` 被硬切——听感就是「好几个短句合成一句，每句都是 6 秒」。

同一段 240 秒游戏混音，两种收句判据（复刻线上同一份窗循环）：

| 收句判据 | 句数 | 句长中位 | 均值 | ≥3s 的长句 |
|---|---|---|---|---|
| 能量 OR（旧） | 68 | 1.40s | 2.16s | **20** |
| VAD 优先（新） | 84 | 1.00s | 1.37s | 11 |

但 VAD 不能无条件当收句权威。同一套判据套在 64 秒动漫女声上，会切在词中间（`なんでここに私がミシェルの` / `ちゃんが生まれた。`）——那段素材 VAD 命中率只有 24%，它的「静音」判断本身就不可信。所以加了一道门槛：**近 2 秒 VAD 命中率 ≥ `--vad-trust`（默认 0.30）才听它，否则回退能量**。实测游戏混音 34~40%、动漫女声 24%，正好分在阈值两边，两类素材各走各的：前者收句变细，后者保持完整句子。

想回退旧行为：`--seg-mode energy`（纯能量，短句会粘长）。想完全恢复更早的纯 VAD：`--seg-mode vad`（起句召回差，女声会「像没听见」）。

注意：能量门控会把 BGM 段也送进识别，靠上面的结果过滤兜底。如果某类素材垃圾句偏多，先调 `--energy-mult`（3.0 → 4.0）；如果断句偏碎，先调 `--vad-trust`（0.30 → 0.40，越高越少用 VAD 收句）。

- 识别引擎是 **llama.cpp**：同一模型用 HF transformers 每秒音频要 0.22s，llama.cpp 只要 0.03s（6 秒音频 0.2s 出字），快 10 倍。
- 说话过程中每 0.4s 就对当前句做一次临时识别；句尾静音判定成立时，如果之后没有新语音，**直接把最后一次临时结果当最终句**，不再推理（实测 90% 命中）。
- 因此延迟 ≈ `--silence` + 一个判定窗口 ≈ 0.6s；GPU 实际只花 0.15s，剩下都在等"这句话说完了没"。
- 主线程只做切句判定，永远不被推理卡住；临时结果做不过来就丢旧的，不堆积。
- 热词和上一句通过 Qwen3-ASR 的 context 传入，system prompt 前缀命中 KV cache，不增加延迟。
  但 Qwen3-ASR 会**顺着 system 往下写**：上一句放在 system 里时，新句可能以它开头，并随轮次累积
  （实测 4/8 段，第 7 段是前 3 句的拼接）。所以识别结果会先切掉与上一句重合的前缀再上屏，
  见 `_strip_echo()`；`--no-context` 可以完全关掉上下文（代价是人名一致性略降）。
- llama-server 由脚本拉起，并用 Windows Job Object 绑定：程序退出或崩溃，server 一起结束；端口上已有同模型 server 就复用，被占了自动换端口。

## 提高准确率

1. **热词最有效**。`hotwords.txt` 里放你常看内容的人名、队名、产品名、术语，逗号或换行分隔，改完重启。
2. **单一语言的内容一定要加 `--lang`**。热词表里只要混着英文词（LPL、CUDA、PyTorch…），模型就会时不时改说英语——同一段 64.6 秒日语素材，不加 `--lang` 有 4/11 段输出成罗马字英语（`Mizuki-chan, what's wrong?`），加 `--lang ja` 后 11/11 全对且一句没丢。
   - 原理：`--lang` 不只是事后过滤，它把 `language Japanese<asr_text>` 写进 **assistant 前缀**，模型才会真的按这个语言解码；只靠过滤的话，那句英语会被整句扔掉。
   - 不希望因为语言判错丢句（日语为主、偶尔有英语台词）就用 `--lang-hint ja`：同样钉语言，但判不符时保留文本。
   - 纯中文内容同理，用 `--lang zh`。
3. 音源本身混着音乐、音效、多人抢话时错字会增多，这是 ASR 的普遍限制。
4. **hotwords 只影响听写，不影响翻译**。它作为 ASR 的 system 提示做解码偏置，让耳朵听准；但句子被正确转写之后，翻译模型不懂某个俚语（例如 `rack` 是床、`doped up` 是被灌药）时，hotwords 帮不上忙——那是翻译层的知识边界，只能靠换模型或术语表解决。

## 翻译模型怎么选

> **为什么不直接用 ASR 模型翻译？** Qwen3-ASR 只被训练成转写（+语种识别），没有对齐过翻译任务。
> 实测（system 指令、user 指令、两轮 few-shot 三种提示构造都试过）：system 里写「翻译成中文、只输出译文」、
> 把指令放在 user 里、给两轮「音频→中文」的 few-shot 示范——三种办法它**都继续照抄日文**；
> 英语音频更是无论怎么提示都只输出英文。唯一像是"翻译"的现象是 system 里挂着中文上一句时吐出中文，
> 但那是把上一句**原样复读**（`美咲、美雪她怎么还不回来？` 与上下文逐字相同），不是译文；
> 偶尔出现的英文意译是语种漂移的副产物，不可控也不完整（`今天也赢点` / `什么什么的？`）。
> 同一句交给 Hy-MT2-1.8B 是 `嗯。是啊，今天天气也真好。很高兴。`。所以翻译必须独立模型。

两个启动脚本，按你的直播语种挑：

| 场景 | 脚本 | 常驻模型 | 整卡显存（含桌面基线） |
|---|---|---|---|
| 日语为主、偶尔英文 | `start-local-日英混合翻译.bat` | Hy-MT2-1.8B + Index-Translate-2B | 7047 MiB（剩 911） |
| 纯日语 | `start-local-日语翻译.bat` | Hy-MT2-1.8B | 5638 MiB（剩 2320） |
| 纯英语 | `start-local-英语翻译.bat` | Index-Translate-2B | 5859 MiB（剩 2100） |

拆分（增量）：ASR 2693 / Hy-MT2-1.8B 1188 / Index-Translate-2B 1409 MiB。
把英语模型的 ctx 从 2048 降到 512 只省 7 MiB——KV 不是开销大头，别白折腾。

- **日语**：Hy-MT2-1.8B 明显更好（15 句 13 对 vs Index 10 对）。`彼は顔が広い`、`再来週`、`先月末` 这类语用题 Index 会翻错。
- **英语**：两者关键词命中率同为 94%，但 Index 的习语更地道（`Nothing like…`、`piece of cake`、`arm and a leg`），脏话也不净化。
- **Index 不能带上下文**：它的训练形状是「单条 user」，塞入任何附加文本都会被当正文复读。所以它的 `--tr-style` 固定 `index`，上下文功能对它无效。
- 超过 40 词的长句会明显变慢：Hy 约 290ms、Index 约 340ms、4B 约 600~700ms（短句是 110/115/250ms）。

## 术语表（--glossary）

`glossary.txt` 每行 `原文=译文`，`#` 开头是注释。只有句子**命中**某条时才生效，所以表可以写很长，不影响别的句子。

按词条类型自动分两路（这个分法是实测出来的，不是设计出来的）：

| 词条类型 | 处理方式 | 为什么 |
|---|---|---|
| 拉丁词条（`rack`、`blood money`、人名） | 注入 system，最多 5 条 | 5 种写法各 5 遍：10/10 命中、0 污染 |
| 中日文词条（`再来週`、`先月末`） | 送模型前直接替换原文 | 注入 system 会让 1.8B 输出韩语（「下下周**의**会议」）；换格式、加“只用简体中文”、标“中日对照”全都治不好（3/3 仍漂移），预替换 3/3 干净 |

实测效果（Hy-MT2-1.8B + v3，每句 5 遍）：

| 句子 | 无术语表 | 有术语表 |
|---|---|---|
| `…get out of the fucking rack in the morning` | 从那种状态里出来 ✗ | 离开那该死的**床** ✓ 5/5 |
| `再来週の打ち合わせは中止になった` | 下周 ✗ | **下下周** ✓ 5/5 |
| `先月末に契約が切れた` | 下月底 ✗ | **上个月底** ✓ 5/5 |
| 英语 34 / 日语 15 全量 | 31 / 13 | **31 / 15**（不退化，日语 +2） |
| 延迟中位 | 98ms | 95ms（命中才改 system 前缀） |

两个已知限制：

- **多词短语在 `index` / `hymt2` 下会被跳过**。这两种风格没有 system 可注入，只能预替换，而把 `break a leg` 直接替换成「祝好运」会破坏原文语法。单词型（`rack`、人名）没问题。
- **`hymt2` 下部分日语词条不生效**。预替换成「下下周の…」后，1.8B 的官方模板会把它又简化回「下周」（3/3）。用 `--tr-style v3` 才有完整效果。

术语表效果与副作用、跨风格稳定性、注入格式对照这几组基准在 `test/` 下（`bench_glossary.py`、`bench_gloss_stab.py`、`bench_fmt.py`），从项目根跑 `python test/bench_glossary.py` 就能复跑。

## 测试与基准脚本

`test/` 下是当时定位问题和选型用的脚本，具体分三类：

- 显存 / 上下文基准：`test_vram_*`、`test_ctx_*`、`test_slot_cost`
- 翻译模型与术语表对照：`bench_38`、`bench_4b_detail`、`bench_index`、`bench_gloss_stab`、`bench_glossary`、`bench_fmt`、`bench_en`、`bench_final`、`bench_g9`、`bench_translate3`、`bench_user20`
- 合成测试语音：`gen_speech`（pyttsx3 → SAPI5，给上下文/显存测试复用）

从项目根运行，例如 `python test/bench_index.py`。脚本按自身位置推算项目根，没有写死绝对路径。
它们需要 `models/`、`llama/` 里的模型和本地素材，**不是 CI 测试**；新加的脚本请直接放 `test/`
（根目录的 `bench_*.py` / `test_*.py` 已被 `.gitignore` 忽略）。

## 备用后端

`--backend hf` 用 transformers 直接跑 `Qwen/Qwen3-ASR-1.7B`（需要 `pip install torch qwen-asr`），不依赖 llama.cpp，但慢 10 倍，仅作对照。

## 致谢

- [Qwen3-ASR](https://github.com/QwenLM/Qwen3-ASR) — 模型
- [llama.cpp](https://github.com/ggml-org/llama.cpp) — 推理，GGUF 来自 [ggml-org/Qwen3-ASR-1.7B-GGUF](https://huggingface.co/ggml-org/Qwen3-ASR-1.7B-GGUF)
- [Silero VAD](https://github.com/snakers4/silero-vad)（经 faster-whisper 打包）、[PyAudioWPatch](https://github.com/s0d3s/PyAudioWPatch)

---

## English

**Realtime captions for whatever your PC is playing, rendered as a subtitle bar at the bottom of the screen. Runs entirely on your local NVIDIA GPU — no cloud, no cost.**
Built for Chinese first (Qwen3-ASR beats whisper-large-v3 by a wide margin on colloquial Mandarin); English and mixed zh/en work too.

- **~0.5 s latency** from end of utterance to on-screen text (measured on an RTX 3090)
- **No virtual audio cable**: captures the system output via WASAPI loopback
- **Subtitle-style overlay**: newest line large and white, previous line small and gray, fades out when nobody is speaking; drag to move, scroll to resize
- **Hotwords** (`hotwords.txt`) for names, teams and jargon
- ~3 GB VRAM, ~13 % GPU load

**This is a fork of [TerryKiddy/live-caption](https://github.com/TerryKiddy/live-caption).** Beyond upstream: optional **local translation** (a second gguf, dual-line overlay, `--translate`, glossary, translations for partial results too); **language pinning** via an assistant-prefix prefill (`--lang` strict, `--lang-hint` soft); **asymmetric segmentation** — energy-OR for onset, VAD-preferred offset that falls back to energy when VAD's hit rate is low (`--vad-trust`); uv-first `setup.bat`. Every upstream flag still works.

**Setup (Windows):** run `setup.bat` (venv + deps + ~2.8 GB model download), unzip a llama.cpp Windows CUDA release **newer than 2026-04-12** (Qwen3-ASR support, PR #19441) into `llama\`, then run `start.bat` (auto language) or `start.bat --lang en`.

**How it hits half a second:** llama.cpp decodes 6 s of audio in 0.2 s; a partial transcription is refreshed every 0.4 s while speech is ongoing, and when the end-of-sentence silence is detected the last partial is reused as the final result if no new speech arrived (90 % hit rate). Latency is therefore ≈ `--silence` (0.5 s) + one VAD window. VAD and segmentation run on the main thread; finals and partials run on two worker threads against a 2-slot llama-server.

Options: `--lang zh|ja|en`, `--lang-hint ja`, `--list` / `--device N`, `--console`, `--srt file`, `--font-size`, `--bottom`, `--alpha`, `--silence`, `--max-seg`, `--debug dir`, `--seg-mode both|energy|vad`, `--vad-trust 0.30`, `--energy-mult N`, `--preroll sec`. See the tables above (parameter names are the same). Note: the segmenter is asymmetric by default — speech *onset* is `energy OR Silero VAD` (the old VAD-only gate silently dropped 56 % of the speech on real mixed audio), while the *offset* follows VAD but only when its recent hit rate clears `--vad-trust`, otherwise it falls back to energy. Using energy for the offset glued short utterances into 6-second `--max-seg` blocks.

MIT License.
