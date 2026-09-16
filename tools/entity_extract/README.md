# 实体热词抽取复现

本目录保存 GLCLAP-Hotword 数据标注使用的实体抽取提示词和可移植运行脚本。GLCLAP-Hotword 的 Common Voice 与 MagicData
全量结果均由 **Qwen/Qwen3.5-4B** 生成；旧的 Qwen3-1.7B 只用于前期对比实验，不是发布索引的来源。

## 固定设置

| 项目 | GLCLAP-Hotword 设置 |
|---|---|
| 模型 | `Qwen/Qwen3.5-4B` |
| ModelScope revision | `fcb1a040bb418b0b8add6f6f6c475386abc2cb97` |
| 服务框架 | vLLM 0.17.0，OpenAI-compatible Chat Completions API |
| 提示词 | `hotword_prompt.txt` |
| 提示词 SHA256 | `ababcea09f10ab1a7e2ad2443ae17de359cd74eaa2208043ce17f14089286b23` |
| temperature | 0 |
| max_tokens | 300 |
| thinking | 关闭 |
| Common Voice 输入 | 26.0 English；按 train/dev/test/validated/invalidated/other 顺序合并并按 path 去重 |
| MagicData 输入 | OpenSLR SLR68 `train/TRANS.txt` |

## 启动模型服务

实体抽取依赖单独的 vLLM 环境，建议不要和 GLCLAP 训练环境混装：

```bash
python -m venv .venv-entity
source .venv-entity/bin/activate
pip install 'vllm==0.17.0' 'modelscope==1.27.1'

python -m modelscope.cli.cli download \
  --model Qwen/Qwen3.5-4B \
  --revision fcb1a040bb418b0b8add6f6f6c475386abc2cb97 \
  --local_dir models/Qwen3.5-4B

CUDA_VISIBLE_DEVICES=0 vllm serve models/Qwen3.5-4B \
  --served-model-name qwen3.5-4b \
  --host 127.0.0.1 --port 8001 \
  --gpu-memory-utilization 0.85 \
  --max-model-len 4096
```

## 先运行小样本

```bash
python tools/entity_extract/run_all_extraction.py \
  --commonvoice-root /data/common-voice-26.0/en \
  --magicdata-root /data/magicdata \
  --cv-api http://127.0.0.1:8001/v1/chat/completions \
  --md-api http://127.0.0.1:8001/v1/chat/completions \
  --model-revision fcb1a040bb418b0b8add6f6f6c475386abc2cb97 \
  --limit 100 \
  --outdir output/entity-smoke
```

检查 `provenance.json` 中的模型、数据版本、提示词哈希和生成参数后，再去掉 `--limit`。输出为：

```text
output/
├── common_voice_en_full.jsonl
├── magicdata_mandarin_full.jsonl
└── provenance.json
```

JSONL 行格式：

```json
{"audio":"/local/path/example.wav","text":"请播放张宇的歌","language":"Chinese","entities":["张宇"]}
```

## 断点恢复与失败语义

- 每 200 行按原输入顺序提交并写盘；已完成记录会在重启时完整扫描，不使用有限行数的近似恢复；
- 恢复键使用相对音频路径，而不是 basename，避免不同目录重名；
- 单次请求最多重试 3 次；连续失败会停止当前任务，不会把 API 错误静默写成“无实体”；
- `temperature=0` 降低随机性，但 GPU kernel、服务框架或模型 revision 改变仍可能导致少量差异，因此必须保存
  `provenance.json` 和最终 JSONL 校验值。

## 提示词与质量选择

提示词只允许从同目录 `hotword_prompt.txt` 读取，不存在时直接报错。它限定人名、具体地名、机构/品牌、
专业术语、产品型号和技术缩写，并过滤常见词、裸数字与时间表达。

前期同样本对比中，Qwen3.5-4B 比 Qwen3-1.7B 更能拒绝中文整句、时间和常见词误抽，因此 GLCLAP-Hotword 最终
选择 4B；公开仓库只保留生成正式索引的 4B 主线。

## 发布前检查

```bash
sha256sum tools/entity_extract/hotword_prompt.txt
python -m py_compile tools/entity_extract/run_all_extraction.py
```

不要提交输出 JSONL、服务日志、模型文件或任何 API token。
