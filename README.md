# GLCLAP-Hotword：AmphionASR 热词召回模块复现

本仓库复现 AmphionASR 的 hotword retriever：输入一段语音和候选热词，输出每个词的
`max-over-time cosine` 分数与峰值时间。发布基线是本项目真实训练运行
`GLCLAP-Hotword`，不是根据论文超参数重新估算的结果。

上游参考为 [AmphionASR 官方项目](https://github.com/AmphionTeam/AmphionASR)及其
[公开手稿](https://github.com/AmphionTeam/AmphionASR/blob/main/main.pdf)。本仓库是独立复现与工程化扩展，
不是 AmphionTeam 官方实现或官方权重；可复现范围是热词召回模块，不包含完整 ASR 解码器或后续强化学习阶段。

- 数据（私有）：[ModelScope HotwordSpeech](https://www.modelscope.cn/datasets/Meiht0702/HotwordSpeech/)
- 推理权重：[weights/glclap_hotword_adapter.safetensors](weights/glclap_hotword_adapter.safetensors)
- 固定实验配置：[configs/glclap_hotword.yaml](configs/glclap_hotword.yaml)
- 最佳权重：20.0 MiB adapter，SHA256 `c20f5b9869b27bc13e092fbbdd9ebab453970011172dccf2c47486b06a4ec22f`
- 最佳验证结果：R@1 0.5167、R@5 0.8500、R@10 **0.9417**、R@20 0.9583
  （epoch 3 / optimizer step 18,190 / 该次有效真实体 n=120 / 每条 K=1000）

## 最快运行已训练权重

要求 Python 3.10、可用 CUDA GPU、`ffmpeg`（浏览器录音转码时需要）。以下命令均在仓库根目录运行：

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch==2.10.0 torchaudio==2.10.0 \
  --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt

bash scripts/download_models.sh
python -m glclap.inference \
  --audio /path/to/example.wav \
  --hotword Amphion --hotword OpenAI
```

其中 `qwen-asr==0.0.6` 是加载冻结音频塔所需的官方实现；旧版
`funasr==1.2.6` 不包含 Qwen3-ASR，不能作为替代依赖。

`download_models.sh` 只下载 `Qwen/Qwen3-ASR-1.7B` 基座；20 MiB adapter 已随 GitHub 仓库提供。
脚本会逐文件校验基座训练版本
（含两个权重分片）的固定 SHA256；因此上游 `master` 后续变化会直接失败，不会静默换模型。
可用 `GLCLAP_MODEL_DIR` 覆盖默认位置。

Web 界面：

```bash
python web_demo/app.py --host 127.0.0.1 --port 8899 --preload
```

浏览器访问 `http://127.0.0.1:8899`。阈值依赖音频长度，候选词的相对排序通常比固定阈值更可靠；
详见 [docs/WEB_DEMO.md](docs/WEB_DEMO.md)。

## 从零复现训练

训练数据集保持私有。获得数据集访问权限后，先在本机执行 `modelscope login --token YOUR_TOKEN`；
令牌只保存在本机环境中，不要写入仓库或命令脚本。

### 1. 下载发布产物和音频

```bash
# 默认仅下载索引、词池、硬负样本和预计算嵌入；随后自动验 SHA256
bash scripts/download_data.sh

# 下载随发布提供的音频分卷（仅限学习与研究用途）
bash scripts/download_data.sh --with-audio
```

Common Voice 26.0 English 和 MagicData SLR68 仅发布索引，不重新分发其音频。请分别从原始发布方
取得并解压。必须使用相同版本；换版本可能造成 `audio_id` 无法匹配。

### 2. 生成本机 manifest

```bash
python scripts/prepare_data.py \
  --release-root data/modelscope/releases/glclap_hotword_v1 \
  --commonvoice-root /data/common-voice-26.0-english \
  --magicdata-root /data/magicdata \
  --output-dir data/glclap_hotword

python scripts/verify_data.py data/glclap_hotword
```

下载分卷并另行取得 Common Voice、MagicData 后，校验应得到训练 1,164,214 行且音频缺失为 0。
脚本把发布索引中的稳定
`audio_id` 重定位为本机绝对路径，仓库和 ModelScope 文件中不会保存作者机器路径。

### 3. 启动训练

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_glclap_hotword.sh
```

该脚本完整固定 GLCLAP-Hotword 参数，包括 `entity` batch sampler、每语言 4095 个池负样本、离线硬负 top-10、
valid seed-7 子采样 8,000、每 9,095 optimizer steps 检索评测，以及 2 × 4 × 16 = 128 的有效 batch。
单卡可设 `GLCLAP_NPROC_PER_NODE=1`，但有效 batch 会变为 64；若要保持 128，应同时把
`--grad-accum` 改为 32，因此不再是原命令的逐步轨迹。

未来训练 checkpoint 只保存 13 个 adapter/温度张量和续训状态，不再重复保存冻结基座。已有旧版
4.5 GiB checkpoint 仍可作为 `--init-checkpoint` 读取；加载器只映射其中的 adapter。

### 4. 独立评测

```bash
python -m glclap.eval_amphion_retrieval \
  --checkpoint weights/glclap_hotword_adapter.safetensors \
  --manifest data/glclap_hotword/valid.jsonl \
  --qwen3asr-checkpoint models/Qwen3-ASR-1.7B \
  --list-size 1000 --seed 7 --output runs/eval_glclap_hotword.json
```

训练中记录的 R@K 来自固定 8,000 行验证子集上的周期性随机候选评测；`val_n` 会随抽到的实体变化，
所以单次 R@K 有抽样波动。发布指标是原始日志的观测值，不承诺每次独立抽样逐位相同。

## 模型结构

```text
16 kHz audio -> frozen Qwen3-ASR audio tower -> frame features [T, 2048]
                                             -> audio MLP -> normalize [T, 512]

hotword text -> frozen Qwen3-ASR token embedding -> masked mean [2048]
                                               -> text MLP -> normalize [512]

score(audio, hotword) = max_t dot(audio_frame[t], hotword)
loss = global bidirectional InfoNCE + local bidirectional InfoNCE
```

两个 MLP 均为 `LayerNorm(2048) → Linear(1024) → GELU → Linear(512)`，加一个可学习温度，
共 5,254,145 个可训练参数。代码只保留音频塔和 token embedding，不把未使用的 LLM decoder 常驻显存；
这不会改变 GLCLAP-Hotword 的计算结果。

与论文共同点是冻结编码组件、双 MLP、512 维共享空间、global/local 双向对比和每语言 N=4095 负样本。
GLCLAP-Hotword 的工程化增量包括：实体冲突感知 batch、离线词形/拼音硬负样本 top-10。
因此请将它描述为 AmphionASR retriever 的复现与扩展，而不是官方权重。

## GLCLAP-Hotword 数据组成

| 来源 | 有效 train | 有效 valid | 发布方式 |
|---|---:|---:|---|
| Common Voice 26.0 English | 850,394 | 8,185 | 仅索引，用户从原站获取音频 |
| MagicData SLR68 | 192,507 | 276 | 仅索引，用户从原站获取音频 |
| ContextASR-Bench | 41,288 | 0 | 按上游 MIT 数据卡发布原始 wav 分卷 |
| 自产 domain TTS 完整会议 | 15,769 | 161 | 音频分卷（研究用途） |
| YouTube 视频切片 | 40,939 | 1,268 | 音频分卷（研究用途） |
| Bilibili 视频切片 | 19,142 | 282 | 音频分卷（研究用途） |
| 李沐 / 极客湾视频切片 | 4,175 | 167 | 音频分卷（研究用途） |
| **总计** | **1,164,214** | **10,339** | 训练时 valid 再固定抽 8,000 |

原始 train 有 2,554,724 行，其中 1,164,639 行带实体；425 行因源机器上音频目录缺失被排除。
私有数据集的 `releases/glclap_hotword_v1/archives/` 发布除 Common Voice、MagicData 之外的全部训练音频；后两者
按原始发布方要求仅提供索引，由用户自行下载。
完整逐来源审计见 [docs/DATA_AUDIT.md](docs/DATA_AUDIT.md)。实体抽取使用固定 revision 的
`Qwen/Qwen3.5-4B`、temperature 0、thinking disabled 和原始 prompt；见
[tools/entity_extract/README.md](tools/entity_extract/README.md)。

## 仓库结构与验收

```text
glclap/                   GLCLAP-Hotword 模型、数据读取、训练、评测、命令行推理
configs/glclap_hotword.yaml  从真实日志还原的权威配置
scripts/                  下载、恢复音频、重定位、校验、一键训练
tools/entity_extract/     固定模型/revision/prompt 的实体抽取
tools/release/            发布索引、音频分卷、adapter 导出与 ModelScope 上传
web_demo/                 Flask 演示
tests/                    不依赖大模型的发布与加载回归测试
```

```bash
python -m pytest -q
python -m compileall -q glclap scripts tools/release tools/entity_extract
CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_train.sh  # 数据准备完成后，两步 GPU 集成测试
```

更完整说明：[数据](docs/DATA.md) · [训练](docs/TRAINING.md) · [评测](docs/EVAL.md) ·
[结果](docs/RESULTS.md) · [复现清单](docs/REPRODUCIBILITY.md) · [许可](docs/LICENSES.md)。

代码使用 MIT License。数据和基础模型各自保留原始许可；代码许可证不覆盖音频、字幕、第三方模型或
第三方数据。不要把任何 ModelScope token、TTS 密钥或本机路径提交到仓库。

## 引用与致谢

本项目的设计、实现和实验受到以下工作与开源项目的启发，感谢相关作者和维护者：

- [GLCLAP / Contextual Biasing for LLM-Based ASR with Hotword Retrieval and Reinforcement](https://arxiv.org/abs/2512.21828)：提供 Global-Local Contrastive Language-Audio Pre-training 的研究基础。
- [AmphionASR](https://github.com/AmphionTeam/AmphionASR)：本项目复现的热词召回模块及其公开手稿来源。
- [Qwen3-ASR-1.7B](https://modelscope.cn/models/Qwen/Qwen3-ASR-1.7B)：提供冻结的音频塔、文本编码组件及官方推理实现。
- [Qwen3.5-4B](https://modelscope.cn/models/Qwen/Qwen3.5-4B)：以固定 revision 和提示词进行热词实体抽取。
- [yt-dlp](https://github.com/yt-dlp/yt-dlp)：用于公开视频数据的下载与音频提取流程。

若使用本仓库整理的数据、训练配置或 adapter，请在实验中注明版本
`glclap_hotword_v1`、adapter SHA256，以及实际使用的候选池大小与随机种子。

本项目是独立复现与工程化扩展，不代表上述论文、项目或工具的官方实现、官方权重或官方数据发布。
请按照各上游项目、模型、数据集和媒体来源的许可证及使用条款进行引用和使用；相关许可边界见
[docs/LICENSES.md](docs/LICENSES.md)。
