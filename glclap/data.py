"""Portable GLCLAP-Hotword audio loading, entity dataset, and conflict-aware sampler."""
# =============================================================================
# 中文说明（数据与采样）
#
#   EntityPairs      —— 一行音频 + 一个（或一组）标注实体；训练时随机选其中一个作为局部目标。
#   EntityBatchSampler —— 实体冲突感知 batch 采样器：尽可能让同一 batch 内的实体互不重复，
#                        因为“同 batch 内其它样本的实体”会被当作负样本，若重复就成了假负样本。
# =============================================================================
from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path

import soundfile as sf
import torch
from torch.utils.data import Dataset, Sampler


# 读音频并统一成 16 kHz 单声道 float32；可按 max_samples 截断（长会议音频只取前若干秒）。
def _load_audio_16k(path: str | Path, max_samples: int | None = None) -> torch.Tensor:
    """Read an audio file as mono float32, resample to 16 kHz, and truncate."""
    import torchaudio.functional as audio_functional

    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    tensor = torch.from_numpy(audio)
    if sample_rate != 16000:
        tensor = audio_functional.resample(
            tensor.unsqueeze(0), sample_rate, 16000
        ).squeeze(0)
    if max_samples is not None:
        tensor = tensor[:max_samples]
    return tensor


# 数据集：每一行给出一段音频、完整转写，以及若干标注实体。
# 训练时从该行实体里随机挑一个作为“局部目标”（hotword），实体列表则用于负样本排除。
class EntityPairs(Dataset):
    """Effective GLCLAP-Hotword rows; choose one annotated entity as each local target."""

    def __init__(
        self, manifest: str | Path, seed: int = 7, max_audio_seconds: float = 20.0
    ):
        with Path(manifest).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        self.seed = seed
        self.max_samples = int(max_audio_seconds * 16000)

        # 目录级存在性检查：网络文件系统（NFS）上逐文件 stat 会极慢甚至卡死，
        # 因此这里只检查所在目录是否存在，具体每个文件由 release 校验脚本负责。
        # Preserve GLCLAP-Hotword's directory-level existence filter. It avoids millions
        # of metadata requests on network filesystems; release verification is
        # responsible for checking every individual file before training.
        ok_dirs: set[str] = set()
        bad_dirs: set[str] = set()
        kept = []
        for row in rows:
            if not row.get("entities"):
                continue
            directory = os.path.dirname(row["audio"])
            if directory in bad_dirs:
                continue
            if directory not in ok_dirs:
                (ok_dirs if os.path.isdir(directory) else bad_dirs).add(directory)
                if directory in bad_dirs:
                    print(
                        f"[EntityPairs] audio dir missing, dropping its rows: {directory}",
                        flush=True,
                    )
                    continue
            kept.append(row)
        self.rows = kept

    def __len__(self) -> int:
        return len(self.rows)

    # 取一行：解码音频，并用 (seed + index) 确定的随机源挑一个实体作为局部目标，
    # 保证同一行每次取到的实体一致（可复现），同时防止模型只记住某个固定实体。
    def __getitem__(self, index: int) -> dict:
        row = self.rows[index]
        audio = _load_audio_16k(row["audio"], max_samples=self.max_samples)
        entities = [
            entity
            for entity in row["entities"]
            if isinstance(entity, str) and entity.strip()
        ]
        local = entities[random.Random(self.seed + index).randrange(len(entities))]
        return {
            "audio": audio,
            "text": row["text"],
            "local_text": local,
            "language": row.get("language", ""),
            "entities": self._norm_entities(row),
        }

    @staticmethod
    def _norm_entities(row: dict) -> list[str]:
        return [
            re.sub(r"\s+", " ", entity.strip().lower())
            for entity in row.get("entities") or []
            if isinstance(entity, str) and entity.strip()
        ]


# 实体冲突感知 batch 采样器。
#
# 目标：同一 batch 内尽量不要出现相同实体，避免把“同一个词”当成负样本。
# 做法：按实体冲突度（degree）降序，贪心塞进最近若干个还没有冲突的 batch。
#
# DDP 下按“每行第一个实体”做 rank 分区；多实体行可能通过次要实体跨 rank 冲突，
# 这是原始 GLCLAP-Hotword 的行为，为保持实验轨迹一致而保留，不做静默修正。
class EntityBatchSampler(Sampler):
    """Greedily pack rows without an entity collision inside a rank batch.

    The DDP first-entity partition matches the historical GLCLAP-Hotword run. For a
    multi-entity row it does not prove that secondary entities are disjoint
    across ranks; this limitation is documented rather than silently changing
    the published experiment.
    """

    # 贪心回溯窗口：只看最近 64 个 batch 是否有空位，避免 O(样本数) 的全局扫描。
    MAX_SCAN = 64

    def __init__(
        self,
        dataset: EntityPairs,
        batch_size: int,
        seed: int = 7,
        rank: int = 0,
        world: int = 1,
    ):
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.rank = int(rank)
        self.world = int(world)
        self.epoch = 0
        self.row_entities = [EntityPairs._norm_entities(row) for row in dataset.rows]

        entity_rows: dict[str, list[int]] = {}
        for index, entities in enumerate(self.row_entities):
            for entity in set(entities):
                entity_rows.setdefault(entity, []).append(index)
        self.degree = [0] * len(self.row_entities)
        for index, entities in enumerate(self.row_entities):
            conflicts: set[int] = set()
            for entity in entities:
                conflicts.update(entity_rows[entity])
            self.degree[index] = len(conflicts)

        # ---- DDP rank 分区 ----
        # 用「seed + 实体排序序号」把实体稳定地分配给某个 rank，保证各卡负责互不相交的实体集合，
        # 从而跨卡也不会出现同一实体被互相当成负样本。
        if self.world > 1:
            rank_of_entity = {
                entity: (self.seed + offset) % self.world
                for offset, entity in enumerate(sorted(entity_rows))
            }
            self.indices = [
                index
                for index in range(len(self.row_entities))
                if self.row_entities[index]
                and rank_of_entity[self.row_entities[index][0]] == self.rank
            ]
            self._distributed = True
        else:
            self.indices = list(range(len(self.row_entities)))
            self._distributed = False

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    # 贪心打包：按顺序把每行放进最近一个“实体集合无交集且未满”的 batch，
    # 放不下就新开一个 batch。frozenset 求交判断即为冲突检测。
    def _pack(self, order: list[int]) -> list[list[int]]:
        batches: list[list[int]] = []
        batch_entities: list[set[str]] = []
        for index in order:
            entities = frozenset(self.row_entities[index])
            placed = False
            lower = max(0, len(batches) - self.MAX_SCAN)
            for batch_index in range(len(batches) - 1, lower - 1, -1):
                if (
                    len(batches[batch_index]) < self.batch_size
                    and batch_entities[batch_index].isdisjoint(entities)
                ):
                    batches[batch_index].append(index)
                    batch_entities[batch_index] |= entities
                    placed = True
                    break
            if not placed:
                batches.append([index])
                batch_entities.append(set(entities))
        return batches

    # 每个 epoch 用 (seed, epoch, rank) 派生新的随机顺序，保证不同 epoch 的打乱不同、
    # 但同一配置可复现；冲突度高的行优先处理更容易被安置。
    def __iter__(self):
        rng = random.Random(self.seed + self.epoch * 1009 + self.rank)
        order = list(self.indices)
        rng.shuffle(order)
        order.sort(key=lambda index: -self.degree[index])
        batches = self._pack(order)

        # ---- DDP 步数对齐 ----
        # 各 rank 的 batch 数可能不同，会卡在集合通信上；因此用 all_reduce(MAX) 取最大值，
        # 让 batch 较少的 rank 循环重复自己的 batch 把轮数补齐（只影响这些卡的有效样本数）。
        if self._distributed:
            import torch.distributed as dist

            count = torch.tensor([len(batches)], dtype=torch.long, device="cuda")
            dist.all_reduce(count, op=dist.ReduceOp.MAX)
            maximum = int(count.item())
            if not batches:
                if not self.indices:
                    raise RuntimeError(f"rank {self.rank} received no GLCLAP-Hotword samples")
                batches = [[self.indices[0]]]
            offset = 0
            while len(batches) < maximum:
                batches.append(batches[offset % len(batches)])
                offset += 1

        rng.shuffle(batches)
        yield from batches

    def __len__(self) -> int:
        return max(1, -(-len(self.indices) // self.batch_size))
