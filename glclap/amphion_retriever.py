"""AmphionASR hotword-retriever reproduction (paper §2.4 / §4.4).

Architecture (mirrors the paper):
  - Frozen Qwen3ASR audio tower AuT(x) -> frame embeddings [T, 2048]
  - Frozen Qwen3ASR LLM token-embedding table E_tok(h) as the TEXT encoder
    (hotwords encoded by mean-pooling the token embeddings) -> [D_text=2048]
  - Two small trainable MLP adapters g_audio, g_text map both frozen
    representations into a shared 512-dim space.
  - Retrieval score: s_h = max_t <a_t, k_h>,  a_t = g_audio(AuT(x)_t),
    k_h = g_text(E_tok(h))  (frame-level, max-over-time; paper Eq.1).
  - Loss: GLCLAP global+local bidirectional contrastive; the local branch can
    use same-language hotword-pool negatives (N per language per step, shared
    across the batch) -- the AmphionASR deviation from GLCLAP's in-batch
    negatives. Temperature is learnable (init 0.07 per paper).
"""
# =============================================================================
# 中文说明（模型与损失）
#
#   音频 x --冻结音频塔--> 帧特征 [T, 2048] --音频适配器--> a_t [T, 512]
#   热词 h --冻结 token embedding--均值池化--> [2048] --文本适配器--> k_h [512]
#
#   检索打分：s(a, h) = max_t <a_t, k_h>   ← max-over-time 余弦，训练/评估/Demo 三处必须一致
#   损失函数：全局双向 InfoNCE + 局部双向 InfoNCE（GLCLAP 的 global/local 结构）
#
#   与 GLCLAP 论文的差异（AmphionASR §4.4）：局部分支额外引入“同语言热词池负样本”，
#   每一步从池里采 N 个负样本，使对比损失不再只依赖 batch 内负样本。
# =============================================================================
from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from . import env

# 共享嵌入维度：论文规定音频与文本映射到同一个 512 维空间。
SHARED_DIM = 512  # paper: shared 512-dim space


# -----------------------------------------------------------------------------
# MLP 适配器：把冻结编码器的表示投影到共享 512 维空间。
# 结构 LayerNorm(in) -> Linear(in, hidden) -> GELU -> Linear(hidden, 512)，
# 训练时只更新这两个适配器与温度参数，音频塔/文本塔保持冻结。
# -----------------------------------------------------------------------------
class MLPAdapter(nn.Module):
    """Small MLP adapter (paper §2.4 / reference adapters):
    LayerNorm(in) -> Linear(in, hidden) -> GELU -> Linear(hidden, SHARED_DIM)."""

    def __init__(self, in_dim: int, hidden_dim: int = 1024, out_dim: int = SHARED_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# 时间维 padding 掩码：把补齐帧的分数置为 -inf，确保后续 max/softmax 永远不会选中它们。
def _mask_padded_logits(logits: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return logits
    return logits.masked_fill(padding_mask.unsqueeze(1).bool(), float("-inf"))


# 带掩码的均值池化：只在有效帧上求平均（全局分支把整段音频压成一个向量）。
def masked_mean(x: torch.Tensor, padding_mask: torch.Tensor | None) -> torch.Tensor:
    if padding_mask is None:
        return x.mean(dim=1)
    valid = (~padding_mask).to(x.dtype).unsqueeze(-1)
    return (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)


# =============================================================================
# 检索主模型：冻结 Qwen3-ASR 音频塔 + 冻结 LLM token embedding 当文本编码器，
# 仅训练两个 MLP 适配器与一个可学习温度。
# =============================================================================
class AmphionRetriever(nn.Module):
    """AmphionASR-style hotword retriever with Qwen3ASR dual frozen encoders.

    Dimension-adaptive: audio tower output dim and LLM hidden dim are read from
    the checkpoint config (1.7B: 2048/2048; 0.6B: 1024/1024).
    """

    def __init__(
        self,
        qwen3asr_checkpoint: str,
        freeze_audio: bool = True,
        freeze_text: bool = True,
        adapter_hidden: int = 1024,
        pad_token_id: int = 151643,
    ):
        super().__init__()
        self.audio, self.text_embedding = self._load_frozen_towers(qwen3asr_checkpoint)
        self._audio_frozen = bool(freeze_audio)
        self._text_frozen = bool(freeze_text)
        self.pad_token_id = int(pad_token_id)

        # dimension-adaptive (0.6B: 1024, 1.7B: 2048)
        audio_dim = int(self.audio.config.output_dim)
        text_dim = int(self.text_embedding.embedding_dim)
        self.audio_adapter = MLPAdapter(audio_dim, adapter_hidden, SHARED_DIM)
        self.text_adapter = MLPAdapter(text_dim, adapter_hidden, SHARED_DIM)
        # paper: learnable TEMPERATURE init 0.07 -> logit_scale = log(1/0.07) = 2.6593
        # (CLIP convention; old code stored 0.07 here, i.e. temp 0.93 - 13x too high)
        # 可学习温度：论文初始化温度 0.07；按 CLIP 约定存 log(1/T)=log(1/0.07)=2.6593。
        # 历史 bug：旧代码这里直接存了 0.07（相当于温度 0.93），温度高了 13 倍。
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / 0.07)))

        # DDP gather flag (set by train_amphion_ddp.py): all-gather local
        # audio/text features so the contrastive loss sees the full batch.
        self.ddp_gather = False

        # AmphionASR SS4.4: optional precomputed hotword-pool raw embeddings
        # (frozen E_tok mean-pooled) -> g_text adapter applied each step.
        self.pool_zh: torch.Tensor | None = None
        self.pool_en: torch.Tensor | None = None
        self.pool_zh_words: list[str] = []
        self.pool_en_words: list[str] = []
        self.pool_neg: int = 0

        if freeze_audio:
            for p in self.audio.parameters():
                p.requires_grad = False
        if freeze_text:
            for p in self.text_embedding.parameters():
                p.requires_grad = False

    @staticmethod
    # 只保留音频塔和 token embedding 两张表，丢弃检索用不到的 LLM decoder 层：
    # 打分结果完全不变，但训练/推理常驻显存显著下降。
    def _load_frozen_towers(checkpoint: str) -> tuple[nn.Module, nn.Module]:
        """Load once and retain only the two frozen components used by GLCLAP.

        The original experiment kept the complete ``thinker`` object even
        though retrieval only calls its token embedding table. Dropping the
        unused decoder layers does not change any score, while substantially
        reducing resident memory for training and inference.
        """
        import sys
        for cand in env.funasr_candidates():
            if Path(cand).exists() and (Path(cand) / "funasr").exists():
                if str(cand) not in sys.path:
                    sys.path.insert(0, str(cand))
                break
        try:
            # Official Qwen3-ASR package. This is the public/recommended path
            # and is pinned in requirements.txt.
            from qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                Qwen3ASRForConditionalGeneration,
            )
        except ModuleNotFoundError as exc:
            if exc.name != "qwen_asr":
                raise
            try:
                # Compatibility with the namespaced source tree used by the
                # original GLCLAP-Hotword run and some older FunASR checkouts.
                from funasr.models.qwen_asr.core.transformers_backend.modeling_qwen3_asr import (
                    Qwen3ASRForConditionalGeneration,
                )
            except ModuleNotFoundError as fallback_exc:
                raise ModuleNotFoundError(
                    "Qwen3-ASR backend not found; install qwen-asr==0.0.6 "
                    "from requirements.txt"
                ) from fallback_exc
        model = Qwen3ASRForConditionalGeneration.from_pretrained(checkpoint, dtype=torch.float16)
        audio = model.thinker.audio_tower
        text_embedding = model.thinker.model.embed_tokens
        del model
        return audio, text_embedding

    # 载入中英热词池与池负采样配置，并预计算归一化词形与 bigram 倒排索引，
    # 避免在每个训练 step 的热路径里重复做正则、建表（否则每步都会重建几十万词的 dict）。
    def set_pools(self, zh, zh_words, en, en_words, pool_neg: int) -> None:
        dev = next(self.parameters()).device
        self.pool_zh = zh.to(dev) if zh is not None else None
        self.pool_en = en.to(dev) if en is not None else None
        self.pool_zh_words = list(zh_words)
        self.pool_en_words = list(en_words)
        self.pool_neg = int(pool_neg)
        # per-entity candidate cache for hard negatives
        self._zh_meta, self._zh_index = self._build_pool_meta(self.pool_zh_words)
        self._en_meta, self._en_index = self._build_pool_meta(self.pool_en_words)
        import random
        self._rand_zh = list(range(len(self.pool_zh_words)))
        self._rand_en = list(range(len(self.pool_en_words)))
        random.Random(0).shuffle(self._rand_zh)
        random.Random(0).shuffle(self._rand_en)
        self._neg_cache: dict = {}
        # precompute normalized pool words once (avoids regex in the hot loop;
        # _sample_shared_neg scanned + re-normed thousands of words per step)
        self._norm_zh = [self._norm(w) for w in self.pool_zh_words]
        self._norm_en = [self._norm(w) for w in self.pool_en_words]
        # offline hard-negative tables (mined confusables output)
        # word -> [neighbor_word, score] sorted desc, top-K kept
        self._hardneg_zh: dict[str, list] = {}
        self._hardneg_en: dict[str, list] = {}
        # word->index maps built once (avoids rebuilding 500k dict every step)
        self._w2i_zh = {w: i for i, w in enumerate(self.pool_zh_words)} if len(self.pool_zh_words) > 100000 else None
        self._w2i_en = {w: i for i, w in enumerate(self.pool_en_words)} if len(self.pool_en_words) > 100000 else None

    def set_hardneg(self, zh_path=None, en_path=None, topk: int = 50) -> None:
        """Load offline hard-neg tables {word: [[neighbor, score], ...]}.
        Streams line-by-line and keeps only topk per word to bound memory."""
        for lang, path, tbl in (("zh", zh_path, self._hardneg_zh),
                                ("en", en_path, self._hardneg_en)):
            if not path:
                continue
            n = 0
            with open(path, encoding="utf-8") as f:
                for ln in f:
                    if not ln.strip():
                        continue
                    try:
                        rec = json.loads(ln)
                    except Exception:
                        continue
                    for w, nbrs in rec.items():
                        tbl[w] = nbrs[:topk]
                        n += 1
            print(f"[hardneg:{lang}] loaded {n} words topk={topk}", flush=True)

    @staticmethod
    # 文本归一化：转小写 + 去掉全部空白。"High NA" 与 "NA" 归一后为 "highna" / "na"，
    # 从而正确判定“一个词是否真的包含另一个词”。
    def _norm(s: str) -> str:
        """lowercase + strip all whitespace (so 'High NA' vs 'NA' collide as
        'highna' vs 'na' — the audio for one truly contains the other)."""
        import re
        return re.sub(r"\s+", "", s).lower()

    @staticmethod
    # 字符二元组集合，用于衡量两个词的词形相近程度。
    def _bigrams(s: str) -> set:
        return {s[i:i + 2] for i in range(len(s) - 1)}

    def _build_pool_meta(self, words):
        """Precompute (norm, char_set, bigram_set) per pool word + bigram
        inverted index {bigram: [pool_idx,...]} for O(shared-bigram) scan."""
        meta = []
        index: dict = {}
        for i, w in enumerate(words):
            n = self._norm(w)
            meta.append((n, set(n), self._bigrams(n)))
            for g in self._bigrams(n):
                index.setdefault(g, []).append(i)
        return meta, index

    # 混淆度 hardness = 0.7×bigram Jaccard + 0.3×字符重叠，越大表示两个词越容易混淆。
    def _hardness(self, a_norm: str, a_chars: set, a_bigrams: set, b_meta) -> float:
        b_norm, b_chars, b_bigrams = b_meta
        # bigram-Jaccard + char-overlap -> higher = more confusable (harder)
        inter = len(a_bigrams & b_bigrams)
        union = len(a_bigrams | b_bigrams)
        bj = inter / union if union else 0.0
        ci = len(a_chars & b_chars) / max(len(a_chars | b_chars), 1)
        return 0.7 * bj + 0.3 * ci

    # 为单个实体挑选候选负样本（按硬度降序，带缓存、bigram 倒排索引加速）。
    # 核心约束是【单向子串安全】：
    #   - b 是 a 的子串 → 音频里说 a 时必然包含 b 的读音，b 是假负样本，必须剔除；
    #   - a 是 b 的子串 → 允许。b 更长、并未真的出现，却是最理想的难负样本。
    def _entity_neg_candidates(self, lang: str, entity: str, excl: set):
        """Return indices of pool words that are valid negatives wrt `entity`:
          - b must NOT be a substring of a (b ⊄ a, after norm). If b were a
            substring of a, the audio speaking 'a' would literally contain the
            sound of 'b', so b is a false negative and must be excluded.
          - a being a substring of b (a ⊂ b) is ALLOWED: that b is confusable
            but not present in the audio -> an ideal hard negative.
        Sorted by hardness desc, cached per entity, bigram-indexed for speed."""
        key = (lang, entity)
        if key in self._neg_cache:
            return self._neg_cache[key]
        meta = self._zh_meta if lang == "zh" else self._en_meta
        index = self._zh_index if lang == "zh" else self._en_index
        a_norm = self._norm(entity)
        a_chars, a_bigrams = set(a_norm), self._bigrams(a_norm)
        # candidate set = union of pool words sharing >=1 bigram with entity
        cand_idx: set = set()
        for g in a_bigrams:
            for i in index.get(g, ()):
                cand_idx.add(i)
        if not cand_idx:
            # no bigram overlap: fall back to a random sample of the pool
            rnd = self._rand_zh if lang == "zh" else self._rand_en
            cand_idx = set(rnd[: min(2000, len(rnd))])
        scored = []
        for i in cand_idx:
            b_norm, b_chars, b_bigrams = meta[i]
            if not b_norm or b_norm == a_norm:
                continue
            if b_norm in excl:
                continue
            # ---- one-directional substring safety: b must not be inside a ----
            if b_norm in a_norm:
                continue
            h = self._hardness(a_norm, a_chars, a_bigrams, (b_norm, b_chars, b_bigrams))
            if h > 0:
                scored.append((h, i))
        scored.sort(reverse=True)
        idx = [i for _, i in scored]
        self._neg_cache[key] = idx
        if len(self._neg_cache) > 20000:
            # bound memory: drop oldest half
            keys = list(self._neg_cache)
            for k in keys[: len(keys) // 2]:
                del self._neg_cache[k]
        return idx

    # 单个实体的难负样本采样，三级回退：
    #   ① 离线难负表（预先挖掘的易混词）→ ② bigram 索引在线候选 → ③ 随机词补齐。
    # 三级都必须满足单向子串安全；返回 [n, 2048] 的原始词嵌入。
    def _sample_hard_neg(self, lang: str, entity: str, excl: set, n: int, dev):
        """Sample n substring-safe negatives for ONE entity: offline hard-neg
        table first (offline-mined confusables), then bigram-indexed online
        candidates, then random fill. One-directional substring safety (b must
        not be inside a) is ALWAYS enforced. Returns [n, 2048] embeddings."""
        import random
        rng = random.Random()
        pool = self.pool_zh if lang == "zh" else self.pool_en
        meta = self._zh_meta if lang == "zh" else self._en_meta
        words = self.pool_zh_words if lang == "zh" else self.pool_en_words
        rnd = self._rand_zh if lang == "zh" else self._rand_en
        a_norm = self._norm(entity)
        tbl = self._hardneg_zh if lang == "zh" else self._hardneg_en
        w2i = self._w2i_zh if lang == "zh" else self._w2i_en
        pick = []
        picked_set = set()
        # 1) offline hard negs (confusable neighbors), substring-filtered
        for nb, _s in tbl.get(entity, [])[: n // 2]:
            bn = self._norm(nb)
            if not bn or bn == a_norm or bn in excl or bn in a_norm:
                continue
            i = w2i.get(nb) if w2i else (words.index(nb) if nb in words else -1)
            if i is None or i < 0:
                continue
            if i in picked_set:
                continue
            pick.append(i)
            picked_set.add(i)
        # 2) online bigram-indexed hard candidates (fallback / extra)
        if len(pick) < n // 2:
            cand = self._entity_neg_candidates(lang, entity, excl)
            for i in cand:
                if len(pick) >= n // 2:
                    break
                if i in picked_set:
                    continue
                pick.append(i)
                picked_set.add(i)
        # 3) random fill with one-directional substring safety
        for i in rnd:
            if len(pick) >= n:
                break
            if i in picked_set:
                continue
            b_norm = meta[i][0]
            if not b_norm or b_norm == a_norm or b_norm in excl:
                continue
            if b_norm in a_norm:
                continue
            pick.append(i)
            picked_set.add(i)
        if not pick:
            pick = [0]
        pick = pick[:n]
        return pool[torch.tensor(pick, device=dev)].float()

    # 整个 batch 共用一组负样本（AmphionASR §4.4 的做法）：
    # 先取离线难负表里的易混词，再用随机池词补齐；所有负样本必须对 batch 内
    # 【每一个】实体保持子串安全，否则会出现假负样本污染损失。
    def _sample_shared_neg(self, lang: str, excl: set, n: int, dev):
        """Sample ONE shared negative set for the whole batch (Amphion §4.4):
        offline hardneg table entries first (lexical/pinyin confusables), then
        random pool words — all substring-safe against EVERY batch entity
        (b must not be inside any a). Returns [n, 2048] raw embeddings."""
        import random
        rng = random.Random()
        pool = self.pool_zh if lang == "zh" else self.pool_en
        words = self.pool_zh_words if lang == "zh" else self.pool_en_words
        rnd = self._rand_zh if lang == "zh" else self._rand_en
        tbl = self._hardneg_zh if lang == "zh" else self._hardneg_en
        w2i = self._w2i_zh if lang == "zh" else self._w2i_en
        excl_norm = {self._norm(e) for e in excl}
        pick = []
        picked_set = set()
        # 1) offline hard negatives: confusable neighbors of any batch entity
        for ent in excl:
            for nb, _s in tbl.get(ent, [])[: n]:
                bn = self._norm(nb)
                if not bn or bn in excl_norm or any(bn in a for a in excl_norm):
                    continue
                i = w2i.get(nb) if w2i else (words.index(nb) if nb in words else -1)
                if i is None or i < 0 or i in picked_set:
                    continue
                pick.append(i)
                picked_set.add(i)
        # 2) random fill, substring-safe vs ALL batch entities
        norms = self._norm_zh if lang == "zh" else self._norm_en
        for i in rnd:
            if len(pick) >= n:
                break
            if i in picked_set:
                continue
            bnn = norms[i]
            if not bnn or bnn in excl_norm or any(bnn in a for a in excl_norm):
                continue
            pick.append(i)
            picked_set.add(i)
        if not pick:
            pick = [0]
        pick = pick[:n]
        return pool[torch.tensor(pick, device=dev)].float()

    # 逐实体难负样本打分：每一行按自己的实体单独采样负样本，计算 max-over-time 相似度 [B, n]。
    def _per_entity_neg_scores(self, a_t, audio_padding, entities, n, dev):
        """Per-entity hard negatives: for each row i (entity e_i) sample its own
        substring-safe negatives and compute max-over-time scores [B, n]."""
        import random
        rng = random.Random()
        scale = self.logit_scale.exp().clamp(max=100)
        excl = set(e for ent in (entities or []) for e in ent) if entities else set()
        scores = torch.zeros(a_t.size(0), n, device=dev)
        for i, ent in enumerate(entities or []):
            if not ent:
                continue
            e = ent[0] if isinstance(ent, (list, tuple)) else ent
            has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in e)
            lang = "zh" if has_cjk else "en"
            pool = self.pool_zh if lang == "zh" else self.pool_en
            words = self.pool_zh_words if lang == "zh" else self.pool_en_words
            if pool is None:
                continue
            negs = self._sample_hard_neg(lang, e, excl, n, dev)  # [n, 2048]
            k_neg = F.normalize(self.text_adapter(negs), dim=-1)  # [n, 512]
            neg_full = scale * torch.einsum("td,hd->ht", a_t[i], k_neg)  # [n, T]
            if audio_padding is not None:
                neg_full = neg_full.masked_fill(audio_padding[i].unsqueeze(0), float("-inf"))
            scores[i] = neg_full.amax(dim=-1)
        return scores

    @staticmethod
    # 音频塔输出帧数公式（厂商 _get_feat_extract_output_lengths 的标量版）：
    # 把 mel 帧数换算成输出帧数，帧→时间戳的反演依赖这个公式。
    def _out_len(mel_len: int) -> int:
        """Vendor _get_feat_extract_output_lengths (scalar)."""
        leave = mel_len % 100
        f = (leave - 1) // 2 + 1
        return ((f - 1) // 2 + 1 - 1) // 2 + 1 + (mel_len // 100) * 13

    # 逐条前向编码音频。为什么不能批量拼接：
    #   1) 厂商 forward 只接受 [128, T] 的单条输入；
    #   2) 其分块窗口注意力（n_window=50 / n_window_infer=800）使“按时间拼接再编码”
    #      与逐条编码【不等价】，实测帧嵌入差异约 25%。
    # 因此这里牺牲吞吐换取数值正确：逐条编码后再 padding 回 [B, T', 2048]。
    def _encode_audio(self, input_features, feature_lens, device):
        """PER-UTTERANCE forward: encode each utterance SEPARATELY. The vendor
        forward only supports [128,T] single input, and its chunked window
        attention makes time-concat packing NOT equivalent to per-utterance
        encoding (measured ~25% frame-embedding deviation). Slower but exact.
        input_features [B,128,T], feature_lens [B] -> padded [B,T',2048] + pad mask.
        """
        self.audio.to(device).eval()
        B = input_features.size(0)
        feats = []
        out_lens = []
        for i in range(B):
            t = int(feature_lens[i])
            m = input_features[i][:, :t].to(device, torch.float16)  # [128, T_i]
            with torch.no_grad():
                out = self.audio(input_features=m,
                                 feature_lens=torch.tensor([t], device=device))
            feats.append(out.last_hidden_state.float())          # [T'_i, 2048]
            out_lens.append(int(out.last_hidden_state.size(0)))
        Tmax = max(out_lens)
        padded = torch.zeros(B, Tmax, feats[0].size(1), device=device)
        for i, p in enumerate(feats):
            padded[i, : p.size(0)] = p
        lengths = torch.tensor(out_lens, dtype=torch.long, device=device)
        padding = torch.arange(Tmax, device=device)[None, :] >= lengths[:, None]
        return padded, padding

    # 文本编码：取冻结 LLM 的 token embedding，按 pad 掩码做均值池化 → [B, 2048]。
    def _text_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Mean-pool LLM token embeddings -> [B, 2048] (masked by pad)."""
        emb = self.text_embedding(input_ids)                            # [B, L, 2048]
        mask = (input_ids != self.pad_token_id).to(emb.dtype)          # [B, L]
        return (emb * mask.unsqueeze(-1)).sum(dim=1) / mask.sum(dim=1, keepdim=True).clamp_min(1.0)

    # 前向总流程：音频编码 → 归一化 → 全局分支 + 局部分支 → 双向对比损失。
    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        input_features = batch["input_features"]   # [B, 128, T_mel]
        feature_lens = batch["feature_lens"]       # [B]
        local_audio, audio_padding = self._encode_audio(input_features, feature_lens,
                                                        input_features.device)
        a_t = F.normalize(self.audio_adapter(local_audio), dim=-1)          # [B, T, 512]
        k_full = F.normalize(self.text_adapter(self._text_embeddings(batch["hotword_ids"]).float()), dim=-1)
        k_global = F.normalize(self.text_adapter(self._text_embeddings(batch["text_ids"]).float()), dim=-1)

        # ---- DDP: all-gather features so the contrastive loss sees the full batch ----
        # ---- DDP 特征汇聚 ----
        # 在计算 logits 之前把各卡的音频/文本特征 all-gather 起来，使对比损失看到完整有效 batch。
        # 这样有效负样本数 = batch_size × grad_accum × world_size，而不是单卡的 batch 大小；
        # entities 也要一起汇聚，否则池负样本的排除集合不完整。
        if self.ddp_gather and self.training:
            a_t, audio_padding = self._gather_audio(a_t, audio_padding)
            k_full = self._gather_flat(k_full)
            k_global = self._gather_flat(k_global)
            if "entities" in batch:
                import torch.distributed as dist
                ws = dist.get_world_size()
                obj: list = [None] * ws
                dist.all_gather_object(obj, batch["entities"])
                batch = dict(batch)
                batch["entities"] = [e for lst in obj for e in lst]

        # ---- 全局分支 ----
        # 整段音频做均值池化得到一个向量，与全局文本（完整转写）做 in-batch 双向对比。
        # logit 先乘可学习温度 scale = exp(logit_scale)。
        global_audio = F.normalize(masked_mean(a_t, audio_padding), dim=-1) # [B', 512]
        scale = self.logit_scale.exp().clamp(max=100)
        b = global_audio.size(0)
        global_logits = scale * global_audio @ k_global.t()

        # ---- 局部分支 ----
        # 每个音频帧分别与候选词算相似度，得到 [B, C, T]，再沿时间取 max：
        # 这就是 max-over-time 打分（论文 Eq.1），也是“只要某一帧像就得分高”的来源。
        # padding 帧在此处先被置为 -inf，保证 max 不会落在补齐位置上。
        local_full = scale * torch.einsum("btd,cd->bct", a_t, k_full)
        local_full = _mask_padded_logits(local_full, audio_padding)
        local_logits = local_full.amax(dim=-1)  # [B', C] max over time

        # ---- AmphionASR SS4.4: hotword-pool negatives (audio->text direction) ----
        # ---- AmphionASR §4.4：热词池负样本（audio→text 方向）----
        # 每个 utterance 从【同语言】池中取 N 个负样本；同一步中中/英各采一组共享负样本，
        # 每一行只与自己语言的负样本打分。按行取同语言是为了避免中英混采导致的假负样本。
        if self.pool_neg > 0 and self.training and self.pool_zh is not None:
            dev = local_audio.device
            entities = batch.get("entities")
            excl = set(e for ent in (entities or []) for e in ent) if entities else set()
            # Paper: N=4095 negatives per utterance from the SAME-LANGUAGE
            # pool. One shared zh set + one shared en set per step; each row
            # scores ONLY against its own language's negatives.
            neg_zh = self._sample_shared_neg("zh", excl, self.pool_neg, dev)   # [n, 2048]
            neg_en = self._sample_shared_neg("en", excl, self.pool_neg, dev)   # [n, 2048]
            k_neg_zh = F.normalize(self.text_adapter(neg_zh), dim=-1)      # [n, 512]
            k_neg_en = F.normalize(self.text_adapter(neg_en), dim=-1)      # [n, 512]
            langs = self._row_languages(batch)                             # len B'
            is_zh = torch.tensor([l == "zh" for l in langs], device=dev,
                                 dtype=torch.bool)                         # [B']
            neg_scores = torch.zeros(b, self.pool_neg, device=dev,
                                     dtype=local_logits.dtype)
            if is_zh.any():
                fz = scale * torch.einsum("btd,hd->bht", a_t[is_zh], k_neg_zh)
                fz = _mask_padded_logits(fz, audio_padding[is_zh])
                neg_scores[is_zh] = fz.amax(dim=-1)
            if (~is_zh).any():
                fe = scale * torch.einsum("btd,hd->bht", a_t[~is_zh], k_neg_en)
                fe = _mask_padded_logits(fe, audio_padding[~is_zh])
                neg_scores[~is_zh] = fe.amax(dim=-1)
            # 正样本仍是 batch 内对角线上的文本；负样本 = in-batch 其它文本 ⊕ 池负样本。
            # 注意 text→audio 方向仍然只走 in-batch，保持与 GLCLAP 对称结构一致。
            logits_a2t = torch.cat([local_logits, neg_scores], dim=1)      # [B, B+n]
            logits_t2a = local_logits.t()                                  # text->audio in-batch
            target = torch.arange(b, device=dev)
            local_loss = (F.cross_entropy(logits_a2t, target) + F.cross_entropy(logits_t2a, target)) / 2
        else:
            target = torch.arange(b, device=global_audio.device)
            local_loss = (F.cross_entropy(local_logits, target) + F.cross_entropy(local_logits.t(), target)) / 2

        target = torch.arange(b, device=global_audio.device)
        # ---- 双向对比损失 ----
        # audio→text 与 text→audio 各算一次交叉熵再取平均，两个方向互为负样本来源。
        # 最终 loss = 全局分支 + 局部分支；返回值里把两部分 detach 出来便于监控。
        global_loss = (F.cross_entropy(global_logits, target) + F.cross_entropy(global_logits.t(), target)) / 2
        return {
            "loss": global_loss + local_loss,
            "global_loss": global_loss.detach(),
            "local_loss": local_loss.detach(),
            "global_logits": global_logits.detach(),
            "local_logits": local_logits.detach(),
        }

    # 判断每行语言：热词文本中出现 CJK 字符即视为中文，否则英文。
    # 用于选择该行使用的负样本池语言。
    def _row_languages(self, batch: dict) -> list:
        """Per-row language from the local hotword text (CJK check), matching
        the pool-negative language selection used in _sample_hard_neg."""
        langs = []
        hot = batch.get("hotword_texts")
        if hot is not None:
            for h in hot:
                s = h if isinstance(h, str) else (h[0] if isinstance(h, (list, tuple)) and h else "")
                langs.append("zh" if any("\u4e00" <= ch <= "\u9fff" for ch in s) else "en")
            return langs
        for ent in batch.get("entities") or []:
            e = ent[0] if isinstance(ent, (list, tuple)) and ent else (ent or "")
            langs.append("zh" if any("\u4e00" <= ch <= "\u9fff" for ch in e) else "en")
        return langs

    def _gather_flat(self, t: torch.Tensor) -> torch.Tensor:
        import torch.distributed as dist
        import torch.distributed.nn.functional as dnn
        world = dist.get_world_size()
        if world == 1:
            return t
        gathered = dnn.all_gather(t.contiguous())
        return torch.cat(gathered, dim=0)

    def _gather_audio(self, a_t: torch.Tensor, audio_padding: torch.Tensor):
        """all_gather [B,T,D] audio embeddings + [B,T] padding; pad to global max T."""
        import torch.distributed as dist
        import torch.distributed.nn.functional as dnn
        world = dist.get_world_size()
        if world == 1:
            return a_t, audio_padding
        t = torch.as_tensor(a_t.size(1), device=a_t.device, dtype=torch.long)
        ts = [torch.empty_like(t) for _ in range(world)]
        dist.all_gather(ts, t)
        max_t = int(max(x.item() for x in ts))
        if max_t > a_t.size(1):
            a_t = torch.cat([a_t, a_t.new_zeros(a_t.size(0), max_t - a_t.size(1), a_t.size(2))], dim=1)
            audio_padding = torch.cat(
                [audio_padding, torch.ones(audio_padding.size(0), max_t - audio_padding.size(1),
                                           dtype=audio_padding.dtype, device=audio_padding.device)], dim=1)
        gas = dnn.all_gather(a_t.contiguous())
        a_t = torch.cat(gas, dim=0)
        gps = [g.detach() for g in dnn.all_gather(audio_padding.contiguous())]
        audio_padding = torch.cat(gps, dim=0)
        return a_t, audio_padding

    @torch.no_grad()
    # 推理打分：不构建计算图，直接返回 [B, H] 的 max-over-time 余弦分数。
    # 排序只看这个分数，与训练/评估/Demo 的协议完全一致。
    def retrieve_scores(self, audio_batch, hotword_ids_batch, audio_padding=None):
        """Inference: local scores [B, H] = max over time of <a_t, k_h>."""
        self.eval()
        a_t = F.normalize(self.audio_adapter(audio_batch), dim=-1)
        k_h = F.normalize(self.text_adapter(self._text_embeddings(hotword_ids_batch)), dim=-1)
        sim = torch.einsum("btd,hd->bht", a_t, k_h)
        if audio_padding is not None:
            sim = sim.masked_fill(audio_padding.unsqueeze(1), float("-inf"))
        return sim.amax(dim=-1)
