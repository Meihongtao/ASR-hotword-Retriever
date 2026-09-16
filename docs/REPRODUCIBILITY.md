# GLCLAP-Hotword 复现清单

这份清单用于区分“程序能启动”和“复现了 GLCLAP-Hotword”。只有所有检查都通过，才应报告为完整复现。

## 固定标识

| 对象 | 标识 |
|---|---|
| 实验 | `glclap_hotword_retriever` |
| Qwen3-ASR 基座 | `Qwen/Qwen3-ASR-1.7B` |
| 基座内容 commit | `d69410f1c275f2b0fa60cbb9960edfcdb0ae0aec` |
| 基座文件校验 | `checksums/qwen3_asr_1.7b.sha256`（含两个权重分片） |
| 实体抽取模型 | `Qwen/Qwen3.5-4B` |
| 抽取模型 revision | `fcb1a040bb418b0b8add6f6f6c475386abc2cb97` |
| prompt SHA256 | `ababcea09f10ab1a7e2ad2443ae17de359cd74eaa2208043ce17f14089286b23` |
| 原始 `best.pt` SHA256 | `af547a4d92298b1cb3176ae2a15ac89656d4f81a26617ce2530b0899b88e00cc` |
| 发布 adapter SHA256 | `c20f5b9869b27bc13e092fbbdd9ebab453970011172dccf2c47486b06a4ec22f` |

adapter 从原始 `best.pt` 逐张量导出，包含 13 个张量、5,254,145 个参数；导出时已验证张量完全相等。

## 数据验收

1. `scripts/download_data.sh --with-audio` 下载并校验 ModelScope 发布文件及随发布提供的音频；
2. 单独取得 Common Voice 26.0 English 与 MagicData SLR68 音频；
3. `scripts/prepare_data.py` 重定位两个有效索引；
4. `scripts/verify_data.py data/glclap_hotword` 必须报告 train 1,164,214、valid 10,339、missing 0；
5. 检查 `release.json` 的逐来源行数和 `checksums.sha256`。

`audio_assets.private.jsonl` 只用于发布者在原机器上定位源文件，包含本机绝对路径，绝不能上传。

## 训练验收

- Python 3.10；直接运行依赖的固定版本见 `requirements-lock.txt`；
- 参考硬件为 2 × NVIDIA RTX 4090 24GB；
- 运行 `CUDA_VISIBLE_DEVICES=0,1 bash scripts/train_glclap_hotword.sh`；
- 启动日志应显示 train 1,164,214、valid 8,000、pool zh 174,335、pool en 551,894、
  warmup 4,547、estimated optimizer steps 90,950；
- 第一次检索评测发生在 optimizer step 9,095；
- checkpoint 的 `model` 只应含 `audio_adapter.*`、`text_adapter.*`、`logit_scale`。

完整训练前可运行 `CUDA_VISIBLE_DEVICES=0 bash scripts/smoke_train.sh`；它从已重定位数据生成 64/16 行
临时子集，单卡执行两个 optimizer steps，并验证保存/加载主路径。

参考最佳点为 step 18,190 的 R@10 0.9417。评测会重新抽取候选集合，随机状态和 GPU kernel 也会带来
差异；应同时报告 seed、候选数、查询真实体数量和完整 R@K，而不是只比较一个数字。

## 推理验收

1. `scripts/download_models.sh` 下载并校验基座，同时校验 GitHub 自带 adapter 的 SHA256；
2. 使用同一音频/候选词分别运行 CLI 与 Web core；
3. 两者的排序分数都必须是 `max_t cosine(a_t, k_h)`；
4. 峰值时间只用于定位解释，不参与排序；
5. 测试命令 `python -m pytest -q` 必须通过。

## 仍可能影响逐位一致性的因素

- CUDA/cuDNN kernel 和驱动版本；
- vLLM 或 Qwen3.5 revision 改变导致重新抽取的实体不同；
- 公共语料换版本、解压目录结构不同或文件损坏；
- 单卡训练未同步增大 gradient accumulation，导致有效 batch 改变；
- 把逐句 TTS 切片误当作 GLCLAP-Hotword 使用的完整 `meeting.wav`。
