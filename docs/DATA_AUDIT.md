# GLCLAP-Hotword 数据发布审计

本页记录 `runs/glclap_hotword_retriever` 实际训练输入与 ModelScope 首轮 staging 的差异。这里的“覆盖”要求音频与
GLCLAP-Hotword 读入的逻辑文件一致；从完整会议重新切出的逐句音频不算精确覆盖。

## 权威训练规模

- 原始 `train.jsonl`：2,554,724 行；其中 1,164,639 行带实体；
- GLCLAP-Hotword 启动时因部分音频目录缺失过滤 425 行；
- GLCLAP-Hotword 实际有效训练集：**1,164,214 行**；
- 验证 manifest：18,562 行，其中 10,346 行带实体；另有 7 行因 MagicData 音频目录缺失被过滤，
  因而有效集为 **10,339 行**，训练时再按固定 seed 7 抽取 8,000 行。

## 首轮 staging 覆盖结果

| GLCLAP-Hotword 自产/视频来源 | train 行数 | unique 音频 | 首轮 staging 精确覆盖 |
|---|---:|---:|---:|
| ContextASR | 41,288 | 41,288 | 0 |
| domain TTS 完整 meeting | 15,769 | 15,769 | 0 |
| YouTube（hungyi/valley/mark/idiode） | 40,939 | 40,939 | 40,939 |
| Bilibili（ZH-B/ZH-B1/EN-B/EN-B1） | 19,142 | 19,142 | 19,142 |
| 李沐学 AI | 1,969 | 1,969 | 0 |
| 极客湾 | 2,206 | 2,206 | 0 |

验证集另外缺少 161 个 domain TTS 完整 meeting、81 条李沐和 86 条极客湾音频；YouTube 1,268 条与
Bilibili 282 条已被首轮 staging 覆盖。

首轮 TTS staging 有 221,136 个逐句切片，但 GLCLAP-Hotword 的 TTS manifest 指向完整 `meeting.wav` 或独立的
ContextASR wav，因此这些切片不能直接用于“精确复现 GLCLAP-Hotword”。它们可以作为扩展数据集保留，但必须与
`releases/glclap_hotword_v1` 分开描述。

## 发布要求

1. `releases/glclap_hotword_v1/*_effective.jsonl` 固定实际训练/验证输入，不能用通用数据全集代替；
2. Common Voice 26.0 English 与 MagicData SLR68 仅发布索引，由使用者从原站下载；
3. 其余来源为自采公开音频，仅限学习与研究用途，精确音频放到
   `releases/glclap_hotword_v1/audio/<audio_id>`；
4. 无法再分发的公开视频只提供视频 ID、时间戳、哈希与重建工具，不把音频伪装成可公开资产；
5. 发布前运行 `tools/release/audit_release_audio.py` 和 `scripts/verify_data.py`，要求缺失数为零。

## GLCLAP-Hotword 发布状态

最终私有发布由 22 个独立 tar 组成，共 121,474 个唯一音频、150,702,827,520 字节：

| 来源 | 分卷 | 文件 | tar 字节 |
|---|---:|---:|---:|
| ContextASR | 12 | 41,288 | 96,375,408,640 |
| domain TTS | 5 | 15,930 | 39,750,010,880 |
| YouTube | 2 | 40,939 | 11,972,300,800 |
| Bilibili | 1 | 19,142 | 1,370,767,360 |
| 李沐 | 1 | 1,969 | 593,039,360 |
| 极客湾 | 1 | 2,206 | 641,300,480 |

远端发布已按路径、大小和 SHA256 逐项核验；索引涉及的非 CV/Magic 唯一音频与
`audio_assets.jsonl` 一一对应，缺失和多余均为 0。最终以 `archives/archives.json` 为权威分卷清单。
