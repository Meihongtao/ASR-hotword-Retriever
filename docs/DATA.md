# GLCLAP-Hotword 数据与实体构造

完整训练以私有 ModelScope 数据集的 `releases/glclap_hotword_v1/` 为数据源。`train_effective.jsonl` 和
`valid_effective.jsonl` 是从真实训练 manifest 重建的权威索引；通用 HotwordSpeech staging 不能替代它们。

## 发布目录

```text
releases/glclap_hotword_v1/
├── release.json                 行数、来源与抽取 provenance
├── checksums.sha256             索引与训练数据产物校验值
├── train_effective.jsonl        1,164,214 行
├── valid_effective.jsonl        10,339 行
├── *_excluded.jsonl             因缺失音频目录被排除的审计记录
├── audio_assets.jsonl           非 CV/Magic 音频的稳定 release_path
├── artifacts/
│   ├── pool_{zh,en}.txt
│   ├── pool_{zh,en}_q3.pt
│   ├── hardneg.{zh,en}.json
│   └── confusable.{zh,en}.tsv
└── archives/                    通过许可审计的音频 tar 分卷及 archives.json
```

ModelScope 只保存数据。轻量 adapter 随 GitHub 仓库放在 `weights/`，不会上传到数据集仓库。

索引行格式：

```json
{
  "id": "commonvoice:en/clips/example.mp3",
  "audio_id": "en/clips/example.mp3",
  "audio": null,
  "text": "example transcript",
  "language": "English",
  "entities": ["example entity"],
  "source": "commonvoice"
}
```

下载后由 `scripts/prepare_data.py` 根据来源拼接本机路径并验证每个文件存在。

## 数据来源和实际计数

| 来源 | train | valid | 说明 |
|---|---:|---:|---|
| Common Voice 26.0 English | 850,394 | 8,185 | 六个 TSV 合并、按 path 去重、仅发布索引 |
| MagicData SLR68 train | 192,507 | 276 | `train/TRANS.txt`、CC BY-NC-ND 4.0、仅发布索引 |
| ContextASR-Bench | 41,288 | 0 | 原始 wav 分卷；保留上游 MIT 数据卡与来源信息 |
| domain TTS full meeting | 15,769 | 161 | 完整 `meeting.wav`，不是后续逐句 staging |
| YouTube sources | 40,939 | 1,268 | hungyi / valley101 / mark / idiode |
| Bilibili sources | 19,142 | 282 | ZH-B / ZH-B1 / EN-B / EN-B1 |
| 李沐 / 极客湾 sources | 4,175 | 167 | 视频语音切片 |

原始获取页：[Common Voice](https://commonvoice.mozilla.org/en/datasets)、
[MagicData SLR68](https://www.openslr.org/68/)、
[ContextASR-Bench](https://huggingface.co/datasets/MrSupW/ContextASR-Bench/tree/main)。

原始 train 为 2,554,724 行，1,164,639 行含非空实体；其中 425 行因源音频目录缺失被训练 loader
过滤，最终有效 1,164,214。原始 valid 为 18,562 行，10,346 行含实体，7 行同样被过滤，最终 10,339。

## 实体抽取

Common Voice 与 MagicData 的正式索引均由 `Qwen/Qwen3.5-4B` 产生。固定参数：

- ModelScope revision `fcb1a040bb418b0b8add6f6f6c475386abc2cb97`；
- `temperature=0`、`max_tokens=300`、thinking disabled；
- prompt 为 `tools/entity_extract/hotword_prompt.txt`，SHA256
  `ababcea09f10ab1a7e2ad2443ae17de359cd74eaa2208043ce17f14089286b23`；
- Common Voice 依次读取 train/dev/test/validated/invalidated/other 并按 path 去重；
- API 失败重试 3 次后中止，不把失败静默写成空实体；恢复时扫描完整输出并使用相对路径键。

完整启动方法见 `tools/entity_extract/README.md`。TTS/会议数据中已有 `hotwords_hit` 时直接使用该标注。

## 词池和硬负样本

- 中文词池 174,335；英文词池 551,894；
- `pool_*_q3.pt` 是冻结 token embedding mean pooling 后的原始向量，训练时再过 text adapter；
- `confusable.*.tsv` 与 `hardneg.*.json` 保存离线易混邻居，GLCLAP-Hotword 每词读取 top-10。

原训练机器曾使用私有 packed-blob 缓存减少网络文件系统随机 I/O；它不属于模型语义，公开主线直接读取
已重定位的音频文件，因此无需生成或发布该缓存。

## 再分发规则

Common Voice 与 MagicData 永远只发索引；Common Voice 仍需遵守 Mozilla 不在其他平台镜像数据的
现行要求，MagicData 仅限非商业使用。数据所有者已确认其余音频可在本次私有数据集中再分发，实际
分卷范围以 `archives.json` 为准。细节见 `docs/LICENSES.md` 和 `docs/DATA_AUDIT.md`。
