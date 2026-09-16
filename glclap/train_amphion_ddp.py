"""AmphionASR retriever DDP training (dual-GPU, 1.7B).

Contrastive loss sees the FULL effective batch: the model all-gathers its local
audio/text features (amphion_retriever.ddp_gather) before computing logits.
pool_neg (AmphionASR SS4.4) samples same-language hotword-pool negatives per step.

Launch:
  CUDA_VISIBLE_DEVICES=4,5 torchrun --nproc_per_node=2 -m glclap.train_amphion_ddp ...
"""
# =============================================================================
# 中文说明（训练）
#
#   有效 batch = batch-size × grad-accum × world_size。
#   对比损失在模型内部先 all-gather 各卡特征（见 amphion_retriever.ddp_gather），
#   所以负样本数量是按“全局 batch”算的，而不是单卡 batch。
#
#   训练旋钮：
#     --sampler entity  实体冲突感知 batch（同一 batch 内实体互不重复，避免假负样本）
#     --pool-neg N      每步从同语言热词池采 N 个负样本（AmphionASR §4.4）
#     --hardneg-zh/en   离线挖掘的词形/拼音难负样本表 top-K
#     --grad-accum      梯度累积，等价放大有效 batch
#
#   评测两类：
#     evaluate()            普通 valid loss + in-batch top1
#     evaluate_retrieval()  检索式 R@1/5/10/20（真词 + 干扰词候选池，与论文协议一致）
# =============================================================================
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data import EntityBatchSampler, EntityPairs
from .data_amphion import AmphionCollator
from .amphion_retriever import AmphionRetriever
from .checkpoint import is_adapter_key, load_adapter_state

try:
    from tensorboardX import SummaryWriter
    HAVE_TB = True
    TB_BACKEND = "tensorboardX"
except Exception:
    try:
        from torch.utils.tensorboard import SummaryWriter
        HAVE_TB = True
        TB_BACKEND = "torch"
    except Exception:
        HAVE_TB = False
        TB_BACKEND = None


# 固定随机种子：python / numpy / torch / cuda 全部覆盖，保证可复现。
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# 把 batch 里的张量搬到目标设备，非张量字段（如 entities 列表）原样保留。
def move_batch(batch: dict, device: torch.device) -> dict:
    return {k: v.to(device) if torch.is_tensor(v) else v for k, v in batch.items()}


# 只保存可训练的 adapter 与温度张量（13 个），冻结的基座权重由固定版本模型提供，
# 因此 checkpoint 只有约 20 MiB，而不是旧的 4.5 GiB。
def checkpoint_model_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Persist only trainable adapters; frozen towers come from the pinned base model."""
    base = model.module if hasattr(model, "module") else model
    return {
        key: value.detach().cpu()
        for key, value in base.state_dict().items()
        if is_adapter_key(key)
    }


# checkpoint 载荷：模型 adapter + 优化器 + 调度器 + 指标 + 续训状态。
def checkpoint_payload(model, optimizer, sched, metrics, **extra) -> dict:
    return {
        "format": "glclap-train-state-v2",
        "model": checkpoint_model_state(model),
        "optimizer": optimizer.state_dict(),
        "sched": sched.state_dict(),
        "metrics": metrics,
        **extra,
    }


@torch.no_grad()
# 常规验证：在 valid 集上前向计算 loss，并统计 in-batch 检索 top1 命中率。
# DDP 下先把各卡的 loss/命中数做 all_reduce 再统一求平均，保证指标口径一致。
def evaluate(model, loader, device, amp_dtype):
    model.eval()
    sums = {"loss": 0.0, "global_loss": 0.0, "local_loss": 0.0}
    count = 0
    hits_g = hits_l = 0
    for batch in loader:
        batch = move_batch(batch, device)
        with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
            out = model(batch)
        n = batch["input_features"].size(0)
        for k in sums:
            sums[k] += float(out[k]) * n
        target = torch.arange(n, device=device)
        hits_g += int(out["global_logits"].argmax(dim=1).eq(target).sum())
        hits_l += int(out["local_logits"].argmax(dim=1).eq(target).sum())
        count += n
    if dist.is_initialized() and dist.get_world_size() > 1:
        sums_t = torch.tensor(list(sums.values()) + [hits_g, hits_l, count],
                              dtype=torch.float64, device=device)
        dist.all_reduce(sums_t)
        for i, k in enumerate(list(sums.keys())):
            sums[k] = float(sums_t[i])
        hits_g, hits_l = int(sums_t[len(sums)]), int(sums_t[len(sums) + 1])
        count = int(sums_t[len(sums) + 2])
    denom = max(count, 1)
    return {**{k: v / denom for k, v in sums.items()},
            "top1_global": hits_g / denom, "top1_local": hits_l / denom}


def _save_best_roll(ckpt: dict, out_dir, keep: int = 5):
    """Save best.pt + best_1.pt..best_{keep-1}.pt rolling window of bests.

    New best -> best.pt; previous bests shift to best_1..best_{keep-2};
    the oldest best_{keep-1} is removed. Returns the list of best files.
    """
    if keep <= 1:
        (out_dir / "best.pt").write_bytes(b"")
        return ["best.pt"]
    import shutil
    # shift existing best_{keep-2}..best_1 -> best_{keep-1}..best_2
    for i in range(keep - 2, 0, -1):
        srcf = out_dir / f"best_{i}.pt"
        dstf = out_dir / f"best_{i+1}.pt"
        if srcf.exists():
            dstf.unlink(missing_ok=True)
            shutil.move(str(srcf), str(dstf))
    # shift best.pt -> best_1.pt
    b0 = out_dir / "best.pt"
    if b0.exists() and b0.stat().st_size > 0:
        (out_dir / "best_1.pt").unlink(missing_ok=True)
        shutil.move(str(b0), str(out_dir / "best_1.pt"))
    torch.save(ckpt, out_dir / "best.pt")
    files = []
    for i in range(keep):
        f = out_dir / (f"best_{i}.pt" if i else "best.pt")
        if f.exists() and f.stat().st_size > 0:
            files.append(f.name)
    return files


@torch.no_grad()
# 检索式验证（与论文协议对齐）：每个 query 构造候选池 = 同语言真词 + 池干扰词，
# 用 max-over-time 余弦排序，统计 R@1/5/10/20。
#
# 两个关键约定：
#   1) 同语言池：中文 query 用中文池、英文 query 用英文池，避免跨语言干扰导致指标失真；
#   2) 真词在所有 embedding 上缓存复用（同语言 query 共享一份 k_pool），避免重复编码。
def evaluate_retrieval(model, valid_ds, collator, device, amp_dtype,
                       n_queries=500, n_pool=1000, seed=7):
    """Retrieval-style val: local max-over-time score vs a candidate pool
    (true entities + distractors from the hotword pool), report R@1/5/10/20."""
    import torch.nn.functional as F
    mm = model.module if hasattr(model, "module") else model  # unwrap DDP
    model.eval()
    pool_zh_words = mm.pool_zh_words or []
    pool_en_words = mm.pool_en_words or []
    # use RAW manifest rows (not Dataset.__getitem__ which decodes audio to tensor)
    base = valid_ds
    while hasattr(base, "dataset"):      # unwrap Subset
        base = base.dataset
    rows = list(base.rows)
    rng = random.Random(seed)
    # true entities across val (for pool) + sample queries
    true_all = set()
    for r in rows:
        for e in r.get("entities") or []:
            true_all.add(e.strip().lower())
    rng.shuffle(rows)
    # only rows with >=1 entity are valid retrieval queries
    rows = [r for r in rows if [e for e in r.get("entities") or [] if e.strip()]]
    queries = rows[:n_queries]
    # Paper-aligned protocol: SAME-LANGUAGE candidate pool per query
    # (10k zh pool for zh queries, 10k en pool for en queries).
    def _is_zh(s: str) -> bool:
        return any("\u4e00" <= c <= "\u9fff" for c in s)
    zh_true = {e for e in true_all if _is_zh(e)}
    en_true = {e for e in true_all if not _is_zh(e)}
    zh_dist = [str(w) for w in pool_zh_words if str(w).lower() not in zh_true]
    en_dist = [str(w) for w in pool_en_words if str(w).lower() not in en_true]
    rng.shuffle(zh_dist)
    rng.shuffle(en_dist)
    # per-query pools are built on the fly below (same-language distractors)
    ranks = []
    k_cache: dict = {}
    for r in queries:
        ents = [e.strip().lower() for e in r.get("entities") or [] if e.strip()]
        if not ents:
            continue
        lang = "zh" if _is_zh(ents[0]) else "en"
        true_lang = zh_true if lang == "zh" else en_true
        dist_lang = zh_dist if lang == "zh" else en_dist
        pool = list(true_lang) + dist_lang[: max(0, n_pool - len(true_lang))]
        pool = pool[:n_pool]
        pool_set = set(pool)
        if lang not in k_cache:
            pids = collator._tok(pool).to(device)
            k_cache[lang] = F.normalize(
                mm.text_adapter(mm._text_embeddings(pids).float()), dim=-1)  # [P,512]
        k_pool = k_cache[lang]
        from .data import _load_audio_16k
        wav = _load_audio_16k(r["audio"])
        cb = collator([{"audio": wav, "text": r["text"], "local_text": r["text"],
                        "entities": ents}])
        a_t, pad = mm._encode_audio(cb["input_features"].to(device),
                                    cb["feature_lens"].to(device), device)
        a_t = F.normalize(mm.audio_adapter(a_t), dim=-1)
        sim = torch.einsum("td,hd->ht", a_t[0], k_pool)
        if pad is not None:
            sim = sim.masked_fill(pad[0].unsqueeze(0).expand(sim.size(0), -1), float("-inf"))
        score = sim.amax(dim=-1)  # [P]
        for e in ents:
            if e in pool_set:
                j = pool.index(e)
                ranks.append(int((score > score[j]).sum()) + 1)
    n = max(len(ranks), 1)
    def rk(k):
        return sum(1 for x in ranks if x <= k) / n if ranks else 0.0
    return {"val_r1": round(rk(1), 4), "val_r5": round(rk(5), 4),
            "val_r10": round(rk(10), 4), "val_r20": round(rk(20), 4),
            "val_n": len(ranks)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train-manifest", required=True)
    p.add_argument("--valid-manifest", required=True)
    p.add_argument("--qwen3asr-checkpoint", required=True)
    p.add_argument("--output-dir", default="runs/amphion_ddp")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=0)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--valid-batch-size", type=int, default=2)
    p.add_argument("--max-audio-seconds", type=float, default=14.0)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--prefetch-factor", type=int, default=2,
                   help="DataLoader prefetch_factor (samples prefetched per worker)")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--adapter-hidden", type=int, default=1024)
    p.add_argument("--sampler", choices=("entity", "random"), default="random")
    p.add_argument("--pool-zh-raw", default=None)
    p.add_argument("--pool-en-raw", default=None)
    p.add_argument("--pool-zh-words", default=None)
    p.add_argument("--pool-en-words", default=None)
    p.add_argument("--pool-neg", type=int, default=0)
    p.add_argument("--hardneg-zh", default=None,
                   help="offline hard-negative table {word: [[nb,score],...]} (zh)")
    p.add_argument("--hardneg-en", default=None)
    p.add_argument("--hardneg-topk", type=int, default=50)
    p.add_argument("--warmup-ratio", type=float, default=0.05,
                   help="fraction of total optimizer steps used for linear warmup "
                        "(then cosine decay to 1%% of peak)")
    p.add_argument("--tensorboard-dir", default=None,
                   help="dir for TensorBoard event files (default: <output-dir>/tensorboard)")
    p.add_argument("--patience", type=int, default=0,
                   help="early stopping: stop if val loss does not improve for N "
                        "consecutive evals (0 = disabled); best checkpoint saved")
    p.add_argument("--eval-every-steps", type=int, default=0,
                   help="evaluate on valid set every N optimizer steps (0 = at epoch end)")
    p.add_argument("--save-every-steps", type=int, default=0,
                   help="save checkpoint every N optimizer steps (0 = at epoch end)")
    p.add_argument("--keep-checkpoints", type=int, default=5,
                   help="max step-checkpoints to keep on disk (oldest removed)")
    p.add_argument("--keep-best", type=int, default=5,
                   help="number of best checkpoints to keep (best.pt + best_1..best_{N-1}.pt rolling)")
    p.add_argument("--tb-every-steps", type=int, default=100,
                   help="log train scalars to TensorBoard every N optimizer steps")
    p.add_argument("--valid-subsample", type=int, default=0,
                   help="evaluate on a random N-row subsample of valid (0 = full valid)")
    p.add_argument("--eval-pool", type=int, default=1000,
                   help="candidate pool size for retrieval-style val R@1/5/10")
    p.add_argument("--cache-clear-steps", type=int, default=0,
                   help="call torch.cuda.empty_cache() every N optimizer steps (0 = off)")
    p.add_argument("--stop-at", default=None,
                   help="wall-clock ISO time (YYYY-MM-DD HH:MM) to auto-save+exit")
    p.add_argument("--init-checkpoint", default=None,
                   help="load model weights from a prior checkpoint (e.g. v2 last.pt) "
                        "before fine-tuning")
    p.add_argument("--resume", default=None,
                   help="full resume: restore model + optimizer + epoch + step from a "
                        "last.pt checkpoint (seamless continue; run with --epochs N "
                        "meaning total epochs including resumed ones)")
    args = p.parse_args()

    # ---- 分布式初始化 ----
    # 用 torchrun 启动时 WORLD_SIZE>1，走 nccl 初始化；单进程直接跑时 dist 保持未初始化，
    # 后面所有 DDP 相关分支都用 world>1 / dist.is_initialized() 判断。
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    torch.cuda.set_device(local_rank)
    if world > 1 and os.environ.get("RANK") is not None:
        dist.init_process_group(backend="nccl", timeout=timedelta(hours=8))
    # world==1 single process (no torchrun): dist stays uninitialized
    seed_everything(args.seed + local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0
    amp_dtype = torch.float16  # paper: fp16
    collator = AmphionCollator(args.qwen3asr_checkpoint,
                               max_audio_seconds=args.max_audio_seconds)
    train_ds = EntityPairs(args.train_manifest, seed=args.seed,
                           max_audio_seconds=args.max_audio_seconds)
    valid_ds = EntityPairs(args.valid_manifest, seed=args.seed,
                           max_audio_seconds=args.max_audio_seconds)
    if args.valid_subsample and args.valid_subsample < len(valid_ds):
        valid_total = len(valid_ds)
        idx = list(range(len(valid_ds)))
        random.Random(args.seed).shuffle(idx)
        idx = idx[: args.valid_subsample]
        from torch.utils.data import Subset
        valid_ds = Subset(valid_ds, idx)
        if is_main:
            print(f"[valid] subsampled to {len(idx)} rows (from {valid_total})", flush=True)
    # 采样器二选一：
    #   entity —— 实体冲突感知 batch（同 batch 内实体不重复，减少假负样本）
    #   random —— 普通随机打乱
    train_batch_sampler = None
    if args.sampler == "entity":
        train_batch_sampler = EntityBatchSampler(train_ds, args.batch_size,
                                                 seed=args.seed, rank=local_rank, world=world)
    loader_worker_args = (
        {"prefetch_factor": args.prefetch_factor} if args.num_workers > 0 else {}
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size if train_batch_sampler is None else 1,
        batch_sampler=train_batch_sampler,
        shuffle=train_batch_sampler is None,
        num_workers=args.num_workers, pin_memory=True, collate_fn=collator,
        **loader_worker_args)
    valid_loader = DataLoader(valid_ds, batch_size=args.valid_batch_size, shuffle=False,
                              num_workers=args.num_workers, pin_memory=True, collate_fn=collator,
                              **loader_worker_args)

    if is_main:
        print(f"[data] train={len(train_ds)} valid={len(valid_ds)} "
              f"valid_subsample={args.valid_subsample or 'full'}", flush=True)

    model = AmphionRetriever(args.qwen3asr_checkpoint,
                             freeze_audio=True, freeze_text=True,
                             adapter_hidden=args.adapter_hidden).to(device)

    # 载入热词池原始嵌入与词表（池负样本用），并挂上离线难负表。
    if args.pool_neg > 0 and args.pool_zh_raw and args.pool_en_raw:
        zh = torch.load(args.pool_zh_raw, map_location="cpu")["raw"]
        en = torch.load(args.pool_en_raw, map_location="cpu")["raw"]
        zwords = [l.strip() for l in open(args.pool_zh_words, encoding="utf-8") if l.strip()]
        ewords = [l.strip() for l in open(args.pool_en_words, encoding="utf-8") if l.strip()]
        model.set_pools(zh, zwords, en, ewords, args.pool_neg)
        if is_main:
            print(f"[pool-neg] zh={len(zwords)} en={len(ewords)} n={args.pool_neg}", flush=True)
    if args.hardneg_zh or args.hardneg_en:
        model.set_hardneg(args.hardneg_zh, args.hardneg_en, topk=args.hardneg_topk)

    if args.init_checkpoint:
        state = load_adapter_state(args.init_checkpoint)
        missing, unexpected = model.load_state_dict(state, strict=False)
        bad_missing = [key for key in missing if is_adapter_key(key)]
        if bad_missing or unexpected:
            raise RuntimeError(
                f"invalid init checkpoint: missing={bad_missing}, unexpected={unexpected}"
            )
        if is_main:
            print(f"[init] loaded from {args.init_checkpoint}: "
                  f"missing={len(missing)} unexpected={len(unexpected)}", flush=True)

    # DDP 包装前先做一次本地前向：让所有 adapter 参数真正被使用/初始化，
    # 否则 DDP 的 find_unused_parameters 会报未使用参数。此次前向关闭 gather。
    # materialize before DDP (plain local forward, gather off)
    model.ddp_gather = False
    first = move_batch(next(iter(train_loader)), device)
    with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
        model(first)
    model.ddp_gather = world > 1
    if world > 1:
        for p in model.parameters():
            dist.broadcast(p.data, src=0)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[local_rank], find_unused_parameters=True)
    # world==1: keep bare model (no DDP wrapper) — simpler and matches
    # single-GPU runs; evaluate_retrieval/evaluate handle both via hasattr.

    out_dir = Path(args.output_dir)
    if is_main:
        out_dir.mkdir(parents=True, exist_ok=True)
    # 只把 requires_grad=True 的参数交给优化器（即两个适配器 + logit_scale）。
    optimizer = torch.optim.AdamW([x for x in model.parameters() if x.requires_grad],
                                  lr=args.lr, weight_decay=0.01)
    trainable = sum(x.numel() for x in model.parameters() if x.requires_grad)
    total = sum(x.numel() for x in model.parameters())
    if is_main:
        print(f"[model] trainable={trainable} total={total} "
              f"({100*trainable/max(total,1):.3f}%)", flush=True)

    # ---- TensorBoard ----
    tb_writer = None
    if is_main and HAVE_TB:
        tb_dir = args.tensorboard_dir or str(out_dir / "tensorboard")
        try:
            tb_writer = SummaryWriter(logdir=tb_dir)
            print(f"[tensorboard] backend={TB_BACKEND} logging to {tb_dir}", flush=True)
        except Exception as e:
            tb_writer = None
            print(f"[tensorboard] init failed: {e}", flush=True)

    # ---- 学习率调度：线性 warmup → 余弦衰减到初始值的 1% ----
    # 总优化步数按“epochs × 每 epoch 优化步数”估算；若中途 resume 会以恢复后的状态继续。
    # ---- LR schedule: linear warmup -> cosine decay to 1% ----
    # total opt steps estimated from epochs x steps/epoch; refined after resume.
    micro_per_epoch = max(1, len(train_ds) // max(1, args.batch_size * world))
    opt_per_epoch = max(1, micro_per_epoch // max(1, args.grad_accum))
    total_opt_est = max(1, (args.epochs - 0) * opt_per_epoch)
    warm_steps = max(1, int(total_opt_est * args.warmup_ratio))

    def make_lr_lambda(warm, tot):
        def lr_lambda(s):
            if s < warm:
                return max(0.0, s / warm)
            p = (s - warm) / max(1, tot - warm)
            return max(0.01, 0.5 * (1 + math.cos(math.pi * min(1.0, p))))
        return lr_lambda

    sched = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=make_lr_lambda(warm_steps, total_opt_est))
    if is_main:
        print(f"[sched] warm={warm_steps} total_est={total_opt_est} "
              f"lr={args.lr} -> cosine decay", flush=True)

    step = 0
    start_epoch = 0
    best_loss = float("inf")
    best_r10 = -1.0
    best_epoch = 0
    bad_evals = 0
    accum = max(1, args.grad_accum)
    max_steps = args.max_steps if args.max_steps and args.max_steps > 0 else None
    deadline = None
    if args.stop_at:
        deadline = datetime.strptime(args.stop_at, "%Y-%m-%d %H:%M")
        if is_main:
            print(f"[stop-at] training will auto-save+exit at {deadline}", flush=True)
    history = []

    # NOTE: resume must happen AFTER DDP wrap + broadcast (broadcast would
    # otherwise overwrite the loaded weights with rank-0 init). Load via
    # model.module (NOT model): DDP.load_state_dict expects "module."-prefixed
    # keys and silently drops our unprefixed checkpoint -> weights never load.
    # ---- 续训恢复 ----
    # 必须在 DDP 包装 + broadcast 之后执行：否则各卡初始化参数会把已加载的权重覆盖掉。
    # 另外要通过 model.module 加载——DDP 的 load_state_dict 期望带 "module." 前缀，
    # 直接传给外层会静默丢弃全部键，导致权重其实没被加载。
    if args.resume:
        ck = torch.load(args.resume, map_location="cpu", weights_only=False)
        if isinstance(ck, dict) and "model" in ck:
            # world==1 keeps the bare model (no DDP wrapper); world>1 wraps it.
            missing, unexpected = (model.module if world > 1 else model).load_state_dict(
                ck["model"], strict=False
            )
            bad_missing = [key for key in missing if is_adapter_key(key)]
            if bad_missing or unexpected:
                raise RuntimeError(
                    f"invalid resume checkpoint: missing={bad_missing}, unexpected={unexpected}"
                )
            if "optimizer" in ck:
                optimizer.load_state_dict(ck["optimizer"])
            if "sched" in ck:
                try:
                    sched.load_state_dict(ck["sched"])
                except Exception:
                    pass
            start_epoch = int(ck.get("metrics", {}).get("epoch", 0))
            step = int(ck.get("metrics", {}).get("step", 0))
            if "best_loss" in ck:
                best_loss = ck["best_loss"]
                best_r10 = ck.get("best_r10", ck.get("best_r1", -1.0))
                bad_evals = ck.get("bad_evals", 0)
            if is_main:
                print(f"[resume] restored model+optimizer+sched from {args.resume}: "
                      f"epoch={start_epoch} step={step} best_r10={best_r10:.4f}", flush=True)

    done = False
    step_ckpts = []  # filenames of saved step checkpoints (oldest first)
    eval_count = 0
    for epoch in range(start_epoch, args.epochs):
        if train_batch_sampler is not None:
            train_batch_sampler.set_epoch(epoch)
        model.train()
        progress = tqdm(train_loader, desc=f"epoch {epoch+1}", disable=not is_main)
        micro = 0
        # ---- 训练主循环 ----
        # 每个 micro-batch 前向一次；累积满 accum 步后才做一次 optimizer.step。
        # 注意 loss 先除以 accum 再 backward，使梯度与“大 batch 一次前向”等价。
        for micro, batch in enumerate(progress, start=1):
            batch = move_batch(batch, device)
            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=True):
                result = model(batch)
            result["loss"].div(accum).backward()
            # ---- 优化步：梯度裁剪 → 更新参数 → 更新学习率 → 记录日志 ----
            # 梯度裁剪阈值 1.0，防止个别异常 batch 把适配器打飞。
            if micro % accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                sched.step()
                step += 1
                progress.set_postfix(
                    loss=f"{result['loss'].item():.3f}",
                    g_loss=f"{result['global_loss'].item():.3f}",
                    l_loss=f"{result['local_loss'].item():.3f}")
                if tb_writer is not None and step % args.tb_every_steps == 0:
                    tb_writer.add_scalar("train/loss", result["loss"].item(), step)
                    tb_writer.add_scalar("train/global", result["global_loss"].item(), step)
                    tb_writer.add_scalar("train/local", result["local_loss"].item(), step)
                    tb_writer.add_scalar("train/lr", optimizer.param_groups[0]["lr"], step)
                if args.cache_clear_steps > 0 and step % args.cache_clear_steps == 0:
                    torch.cuda.empty_cache()
                # ---- step-driven checkpoint save (keep last N) ----
                if args.save_every_steps > 0 and step % args.save_every_steps == 0 and is_main:
                    fn = out_dir / f"step_{step}.pt"
                    torch.save(
                        checkpoint_payload(
                            model, optimizer, sched, {"epoch": epoch + 1, "step": step}
                        ),
                        fn,
                    )
                    step_ckpts.append(fn.name)
                    print(f"[save] {fn.name} (kept={len(step_ckpts)})", flush=True)
                    while len(step_ckpts) > args.keep_checkpoints:
                        old = step_ckpts.pop(0)
                        (out_dir / old).unlink(missing_ok=True)
                        print(f"[save] removed {old}", flush=True)
                # ---- step-driven evaluation ----
                # ---- 周期性评测 ----
                # 评测前先关闭 DDP gather（各卡独立算指标），并空缓存避免显存碎片；
                # 常规 loss 与检索式 R@K 一起评，后者失败不影响训练继续。
                if args.eval_every_steps > 0 and step % args.eval_every_steps == 0:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                    if dist.is_initialized():
                        dist.barrier()
                    if world > 1:
                        model.module.ddp_gather = False
                    metrics = evaluate(model, valid_loader, device, amp_dtype)
                    # retrieval-style val R@1/5/10 (against hotword pool)
                    try:
                        r = evaluate_retrieval(
                            model, valid_ds, collator, device, amp_dtype,
                            n_queries=min(500, len(valid_ds)),
                            n_pool=args.eval_pool or 1000,
                            seed=args.seed + eval_count)
                        metrics.update(r)
                    except Exception as e:
                        import traceback
                        print(f"[retrieval-eval] failed: {e}", flush=True)
                        traceback.print_exc()
                    if world > 1:
                        model.module.ddp_gather = True
                    metrics["epoch"] = epoch + 1
                    metrics["step"] = step
                    history.append(metrics)
                    eval_count += 1
                    if is_main:
                        cur_loss = metrics["loss"]
                        cur_r10 = metrics.get("val_r10", 0.0)
                        # ---- 早停：以检索 val R@10 为准 ----
                        # 连续 patience 次没有提升就停止；每次刷新最好成绩都会滚动保存 best.pt。
                        if args.patience > 0:
                            if cur_r10 > best_r10 + 1e-4:
                                best_r10 = cur_r10
                                bad_evals = 0
                                bfiles = _save_best_roll(
                                    checkpoint_payload(
                                        model, optimizer, sched, metrics,
                                        best_r10=best_r10, bad_evals=bad_evals,
                                    ),
                                    out_dir, keep=args.keep_best)
                                print(f"[early-stop] NEW best val_r10={cur_r10:.4f} "
                                      f"(step {step}) -> saved {bfiles}", flush=True)
                            else:
                                bad_evals += 1
                                print(f"[early-stop] val_r10={cur_r10:.4f} not better than "
                                      f"{best_r10:.4f} ({bad_evals}/{args.patience})", flush=True)
                        torch.save(
                            checkpoint_payload(
                                model, optimizer, sched, metrics,
                                best_r10=best_r10, bad_evals=bad_evals,
                            ),
                            out_dir / "last.pt",
                        )
                        (out_dir / "metrics.json").write_text(json.dumps(history, indent=2),
                                                              encoding="utf-8")
                        print(json.dumps(metrics, ensure_ascii=False), flush=True)
                        if tb_writer is not None:
                            for k, v in metrics.items():
                                if isinstance(v, (int, float)):
                                    tb_writer.add_scalar(f"eval/{k}", v, step)
                    if is_main and args.patience > 0 and bad_evals >= args.patience:
                        print(f"[early-stop] patience {args.patience} reached at step {step}; stopping",
                              flush=True)
                        done = True
                if max_steps is not None and step >= max_steps:
                    done = True
                if deadline is not None and datetime.now() >= deadline:
                    print(f"\n[stop-at] reached {deadline}, saving+exiting (step={step})", flush=True)
                    done = True
                if done:
                    break
        if done:
            break
        # ---- epoch-end evaluate/save when eval-every-steps is OFF ----
        if args.eval_every_steps <= 0:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            if dist.is_initialized():
                dist.barrier()
            if world > 1:
                model.module.ddp_gather = False
            metrics = evaluate(model, valid_loader, device, amp_dtype)
            if world > 1:
                model.module.ddp_gather = True
            metrics["epoch"] = epoch + 1
            metrics["step"] = step
            history.append(metrics)
            if is_main:
                # ---- early stopping on retrieval val R@10 (patience evals, 0 = off) ----
                cur_r10 = metrics.get("val_r10", 0.0)
                if args.patience > 0:
                    if cur_r10 > best_r10 + 1e-4:
                        best_r10 = cur_r10
                        best_epoch = epoch + 1
                        bad_evals = 0
                        _save_best_roll(
                            checkpoint_payload(
                                model, optimizer, sched, metrics,
                                best_r10=best_r10, bad_evals=bad_evals,
                            ),
                            out_dir, keep=args.keep_best)
                        print(f"[early-stop] NEW best val_r10={cur_r10:.4f} "
                              f"(epoch {epoch+1}) -> saved best (rolled)", flush=True)
                    else:
                        bad_evals += 1
                        print(f"[early-stop] val_r10={cur_r10:.4f} not better than "
                              f"{best_r10:.4f} ({bad_evals}/{args.patience})", flush=True)
                torch.save(
                    checkpoint_payload(
                        model, optimizer, sched, metrics,
                        best_r10=best_r10, bad_evals=bad_evals,
                    ),
                    out_dir / "last.pt",
                )
                (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
                print(json.dumps(metrics, ensure_ascii=False), flush=True)
                if tb_writer is not None:
                    for k, v in metrics.items():
                        if isinstance(v, (int, float)):
                            tb_writer.add_scalar(f"eval/{k}", v, step)
            if is_main and args.patience > 0 and bad_evals >= args.patience:
                print(f"[early-stop] patience {args.patience} reached at epoch {epoch+1}; stopping",
                      flush=True)
                break
        if deadline is not None and datetime.now() >= deadline:
            break

    # final evaluate+save when stopped early via max_steps
    if done:
        if dist.is_initialized():
            dist.barrier()
        if world > 1:
            model.module.ddp_gather = False
        metrics = evaluate(model, valid_loader, device, amp_dtype)
        if world > 1:
            model.module.ddp_gather = True
        metrics["epoch"] = epoch + 1
        metrics["step"] = step
        history.append(metrics)
        if is_main:
            torch.save(
                checkpoint_payload(
                    model, optimizer, sched, metrics,
                    best_loss=best_loss, bad_evals=bad_evals,
                ),
                out_dir / "last.pt",
            )
            (out_dir / "metrics.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
            print(json.dumps(metrics, ensure_ascii=False), flush=True)
            if tb_writer is not None:
                for k, v in metrics.items():
                    if isinstance(v, (int, float)):
                        tb_writer.add_scalar(f"eval/{k}", v, step)
    if tb_writer is not None:
        tb_writer.close()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
