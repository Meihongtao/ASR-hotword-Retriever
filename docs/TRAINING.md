# GLCLAP-Hotword 训练方法

唯一主线入口是 `scripts/train_glclap_hotword.sh`，底层为 `glclap.train_amphion_ddp`。配置文件
`configs/glclap_hotword.yaml` 和启动脚本都来自 `runs/glclap_hotword_retriever` 的真实日志、参数及 checkpoint，
不是论文默认值的复制。

## 冻结与可训练部分

- 冻结：Qwen3-ASR-1.7B audio tower；
- 冻结：同一基座的 LLM token embedding table，仅做 token mean pooling；
- 可训练：audio/text 两个 `LayerNorm → Linear(2048,1024) → GELU → Linear(1024,512)` adapter；
- 可训练：以 `log(1/0.07)` 初始化的 temperature/logit scale；
- 可训练参数总数：5,254,145。

运行时代码只保留实际参与前向的 audio tower 与 token embedding，不保留未调用的 decoder layers。

## 目标函数与负样本

`a_t` 为归一化音频帧向量，`k_h` 为归一化热词向量：

```text
local_score(x, h) = max_t dot(a_t, k_h)
global_audio(x)   = normalize(masked_mean(a_t))
loss              = bidirectional_InfoNCE(global) + bidirectional_InfoNCE(local)
```

local audio→text 方向还拼接每语言 4095 个词池负样本。负样本先使用离线 hard-negative 表的 top-10，
再随机补足；候选词若是当前真实实体的子串会被过滤，以减少“音频实际也包含该词”的假负例。

`EntityBatchSampler` 在每个 rank 内贪心构造实体不相交的 batch。GLCLAP-Hotword 的跨 rank 分区以每行首实体为键；
多实体行理论上仍可能通过次要实体形成跨 rank 冲突。这是原始 GLCLAP-Hotword 行为，保留它是为了复现实验轨迹。

## 精确配置

| 项 | GLCLAP-Hotword |
|---|---|
| train 有效行 | 1,164,214 |
| valid 有效行 / 实际子采样 | 10,339 / 8,000（seed 7） |
| GPU | 2 × RTX 4090 24GB |
| 每卡 micro batch | 4 |
| gradient accumulation | 16 |
| 有效 batch | 128 |
| epoch / estimated optimizer steps | 10 / 90,950 |
| AdamW | lr 3e-4，weight decay 0.01 |
| scheduler | warmup 4,547 steps，再 cosine 到峰值 1% |
| precision / clip | fp16 / 1.0 |
| hard negative / pool negative | top-10 / 每语言 4095 |
| 检索验证 | 每 9,095 optimizer steps，K=1000 |
| checkpoint | 每 2,000 optimizer steps，滚动保留 5 个 |
| early stopping | val R@10，patience 5 |

## 启动与恢复

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_glclap_hotword.sh
```

向底层入口追加参数：

```bash
CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_glclap_hotword.sh \
  --resume runs/glclap_hotword_retriever/last.pt
```

`--resume` 恢复 adapter、optimizer、scheduler、epoch 和 step；`--init-checkpoint` 只初始化 adapter。
新 checkpoint 使用 `glclap-train-state-v2`，冻结基座从 `master` 下载后按
`checksums/qwen3_asr_1.7b.sha256` 逐文件校验。旧版完整 `.pt` 也能作为
初始化权重读取，但不建议继续分发其中重复的冻结参数和 optimizer 数据。

单 GPU 可运行，但若不把 accumulation 从 16 调为 32，有效 batch 会从 128 变为 64；这属于新实验。
