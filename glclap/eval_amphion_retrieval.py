"""Hotword retrieval evaluation for the AmphionASR retriever.

Protocol (AmphionASR §5.3 style, small version):
  - candidate hotword pool = unique entities from the eval manifest
  - for each query audio, build a K-list: its true entities + random distractors
  - local score = max-over-time <a_t, k_h>; measure recall@1/@5 and mean rank.
"""
# =============================================================================
# 中文说明（独立评测）
#
# 协议：候选池 = 真词 + 随机干扰词（默认 K=1000），用 max-over-time 余弦排序，
# 统计 recall@1/5/10/20/50 与平均排名，并按 n-gram 数、是否中文分组细看。
#
# --unseen-train 可传入训练 manifest，把训练中出现过的实体从真词和词表里都剔除，
# 从而衡量“真正没见过的新实体”泛化能力。
# =============================================================================
from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path

import torch
import numpy as np

from .data import _load_audio_16k
from .data_amphion import AmphionCollator
from .amphion_retriever import AmphionRetriever
from .checkpoint import load_adapters


def load_manifest(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


# 评测主流程：加载模型 →（可选）过滤训练见过的实体 → 批量编码词表 →
# 逐条 query 打分并统计排名。
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--qwen3asr-checkpoint", required=True)
    p.add_argument("--list-size", type=int, default=1000,
                   help="candidate pool size per query (true + distractors)")
    p.add_argument("--num-queries", type=int, default=0, help="0 = all rows")
    p.add_argument("--max-audio-seconds", type=float, default=14.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--output", default="runs/amphion_eval.json")
    p.add_argument("--unseen-train", default=None,
                   help="training manifest(s) (comma-separated): entities seen there are "
                        "EXCLUDED from eval (true entities + vocab), measuring only "
                        "true unseen-entity generalization")
    args = p.parse_args()

    # build set of entities seen in training (for unseen-only eval)
    train_entities: set[str] = set()
    if args.unseen_train:
        for m in args.unseen_train.split(","):
            for r in load_manifest(m.strip()):
                for e in (r.get("entities") or []):
                    e = e.strip().lower()
                    if e:
                        train_entities.add(e)
        print(f"[unseen-filter] train entities loaded: {len(train_entities)}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collator = AmphionCollator(args.qwen3asr_checkpoint, max_audio_seconds=args.max_audio_seconds)
    print("[1/4] loading model")
    model = AmphionRetriever(args.qwen3asr_checkpoint).to(device).eval()
    load_adapters(model, args.checkpoint)
    model.eval()

    print("[2/4] loading manifest + building vocab")
    rows = [r for r in load_manifest(args.manifest) if r.get("entities")]
    if args.num_queries and args.num_queries < len(rows):
        rng = random.Random(args.seed)
        rng.shuffle(rows)
        rows = rows[: args.num_queries]
    if train_entities:
        before = len(rows)
        kept = []
        for r in rows:
            ents = [e.strip().lower() for e in r.get("entities") or []]
            if any(e not in train_entities for e in ents):
                kept.append(r)
        rows = kept
        print(f"[unseen-filter] rows {before} -> {len(rows)} (dropped queries whose entities all seen)", flush=True)
    vocab = []
    seen = set()
    for r in rows:
        for e in r["entities"]:
            e = e.strip().lower()
            if not e:
                continue
            if train_entities and e in train_entities:
                continue  # unseen-only mode: exclude train-seen entities from vocab
            if e not in seen:
                seen.add(e)
                vocab.append(e)
    idx = {e: i for i, e in enumerate(vocab)}
    print(f"    rows={len(rows)} vocab={len(vocab)} (train-seen excluded: {len(train_entities)})")

    # 词表编码：分块（512 个一批）过文本塔 + 适配器并归一化，得到 [V, 512] 的候选词嵌入。
    # 分块是为了控制显存峰值。
    print("[3/4] encoding hotword texts via LLM embed table")
    tok = collator.tokenizer
    text_emb = []  # [V, 512]
    with torch.no_grad():
        for s in range(0, len(vocab), 512):
            chunk = vocab[s:s + 512]
            ids = collator._tok(chunk)
            k = model.text_adapter(model._text_embeddings(ids.to(device)).float())
            k = torch.nn.functional.normalize(k, dim=-1)
            text_emb.append(k)
    text_emb = torch.cat(text_emb, dim=0)  # [V, 512]

    print("[4/4] retrieving")
    ranks = []          # all-true-entity ranks
    gran_ranks = {"all": [], "1gram": [], "2gram": [], "3gram": [], "3plus": [], "cjk": []}
    n_true_all = 0
    for i in range(0, len(rows), 4):
        batch_rows = rows[i:i + 4]
        wavs = [_load_audio_16k(r["audio"]) for r in batch_rows]
        cb = collator([{"audio": w, "text": r["text"], "local_text": r["text"],
                        "entities": r["entities"]} for w, r in zip(wavs, batch_rows)])
        input_features = cb["input_features"].to(device)
        feature_lens = cb["feature_lens"].to(device)
        with torch.no_grad():
            a_t, audio_pad = model._encode_audio(input_features, feature_lens, device)
            a_t = torch.nn.functional.normalize(model.audio_adapter(a_t), dim=-1)
        for b in range(len(batch_rows)):
            r = batch_rows[b]
            true = []
            for e in r["entities"]:
                e = e.strip().lower()
                if e in idx and (not train_entities or e not in train_entities):
                    true.append(e)
            if not true:
                continue
            n_true_all += len(true)
            this = set(true)
            dist = [e for e in vocab if e not in this]
            rng = random.Random(args.seed + i + b)
            rng.shuffle(dist)
            k = min(args.list_size - len(true), len(dist))
            terms = list(true) + dist[:k]
            rng.shuffle(terms)
            ids = collator._tok(terms).to(device)
            with torch.no_grad():
                k_h = torch.nn.functional.normalize(model.text_adapter(model._text_embeddings(ids).float()), dim=-1)
            # [H, T] 逐帧相似度 → 沿时间取 max 得到该 query 对每个候选词的分数。
            # padding 帧置 -inf，确保 max 不会取到补齐位置。
            sim = torch.einsum("td,hd->ht", a_t[b], k_h)  # [H, T]
            if audio_pad is not None:
                sim = sim.masked_fill(audio_pad[b].unsqueeze(0).expand(sim.size(0), -1), float("-inf"))
            score = sim.amax(dim=-1)  # [H]
            # 排名 = 比真词分数严格更高的候选词个数 + 1（越小越好）。
            for t in true:
                j = terms.index(t)
                rk = int((score > score[j]).sum()) + 1
                ranks.append(rk)
                gkey = f"{len(t.split())}gram" if len(t.split()) <= 3 else "3plus"
                gran_ranks["all"].append(rk)
                gran_ranks[gkey].append(rk)
                if re.search(r"[\u4e00-\u9fff]", t):
                    gran_ranks["cjk"].append(rk)

    def recall(arr, kk):
        return float(np.mean([1 if x <= kk else 0 for x in arr])) if arr else 0.0

    report = {
        "config": {k: str(v) for k, v in vars(args).items()},
        "n_queries_true": len(ranks),
        "n_true_entities": n_true_all,
        "chance_top1": 1.0 / max(len(vocab), 1),
        "recall@1": round(recall(ranks, 1), 4),
        "recall@5": round(recall(ranks, 5), 4),
        "recall@10": round(recall(ranks, 10), 4),
        "recall@20": round(recall(ranks, 20), 4),
        "recall@50": round(recall(ranks, 50), 4),
        "mean_rank": round(float(np.mean(ranks)), 2) if ranks else None,
        "by_ngram": {},
    }
    for key, arr in gran_ranks.items():
        if arr:
            report["by_ngram"][key] = {
                "n": len(arr),
                "recall@1": round(recall(arr, 1), 4),
                "recall@5": round(recall(arr, 5), 4),
                "recall@10": round(recall(arr, 10), 4),
                "recall@20": round(recall(arr, 20), 4),
                "recall@50": round(recall(arr, 50), 4),
            }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(report, indent=1, ensure_ascii=False))
    print("saved:", args.output)


if __name__ == "__main__":
    main()
