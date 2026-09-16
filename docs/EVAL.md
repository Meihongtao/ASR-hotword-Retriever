# 评测协议

主线入口：`python -m glclap.eval_amphion_retrieval`。

对每条查询音频，评测器从 manifest 的去重实体词表中建立候选集：该音频的真实实体加随机同语言
干扰词，最多 K=1000。每个候选按以下唯一分数排序：

```text
score(h) = max_t cosine(audio_frame_t, hotword_embedding_h)
```

报告真实实体的 recall@1/5/10/20/50、mean rank，并按空格分词长度与 CJK 字符分组。

```bash
python -m glclap.eval_amphion_retrieval \
  --checkpoint weights/glclap_hotword_adapter.safetensors \
  --manifest data/glclap_hotword/valid.jsonl \
  --qwen3asr-checkpoint models/Qwen3-ASR-1.7B \
  --list-size 1000 --seed 7 --output runs/eval_glclap_hotword.json
```

使用 `--unseen-train data/glclap_hotword/train.jsonl` 会从评测真实实体和候选词表中排除训练见过的实体，用于测量
unseen-entity 泛化。使用逗号可以传多个训练 manifest。

训练中 GLCLAP-Hotword 先以 seed 7 从 10,339 个有效 valid 行固定抽取 8,000 行，再每 9,095 optimizer steps 运行
K=1000 检索验证。原始日志每次实际进入 rank 统计的真实体 n=112–135，因此 R@K 有明显抽样方差。
比较结果时必须同时报告 checkpoint、seed、K、查询数/真实体数及是否启用 unseen filter。
