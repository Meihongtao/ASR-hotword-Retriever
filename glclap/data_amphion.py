"""AmphionASR retriever data collator.

Produces, per sample:
  - input_features [128, T_mel], feature_lens  (Whisper-style log-mel, same as Q3)
  - text_ids     : full transcript via Qwen3ASR tokenizer (global branch)
  - hotword_ids  : the local entity/hotword via Qwen3ASR tokenizer (local branch)
  - entities     : normalized entity list (for pool-negative exclusion)
"""
# =============================================================================
# 中文说明（collator）
#
# 每个样本产出：
#   input_features [128, T_mel] + feature_lens  —— Whisper 风格 log-mel 音频特征
#   text_ids       —— 完整转写（全局分支用）
#   hotword_ids    —— 局部热词/实体（局部分支用，也是正样本）
#   entities       —— 归一化实体列表（供池负样本做排除判断）
#
# 注意：音频塔要求 fp16 输入；本文件产出的 log-mel 是 float32，
# 在 _encode_audio 里再转 fp16，避免与基座权重 dtype 冲突。
# =============================================================================
from __future__ import annotations

import numpy as np
import torch
from transformers import AutoTokenizer, WhisperFeatureExtractor


class AmphionCollator:
    def __init__(self, qwen3asr_checkpoint: str, max_audio_seconds: float = 14.0,
                 pad_token_id: int = 151643):
        self.tokenizer = AutoTokenizer.from_pretrained(qwen3asr_checkpoint, local_files_only=True)
        self.fe = WhisperFeatureExtractor.from_pretrained(qwen3asr_checkpoint)
        self.pad_token_id = pad_token_id
        self.max_samples = int(max_audio_seconds * 16000)
        self.mel_filters = torch.from_numpy(np.asarray(self.fe.mel_filters)).float()  # [201,128]

    # 分词：不加特殊 token，按 batch 内最长序列用 pad_token_id 右侧补齐。
    # 截断到 max_len=40，热词/实体一般远短于此。
    def _tok(self, values: list[str], max_len: int = 40) -> dict[str, torch.Tensor]:
        """Tokenize with Qwen3ASR tokenizer, no special tokens, pad to batch max."""
        ids = []
        for v in values:
            t = self.tokenizer(v, add_special_tokens=False).input_ids
            ids.append(t[: max_len] if len(t) > max_len else t)
        L = max(len(t) for t in ids) if ids else 1
        out = torch.full((len(ids), L), self.pad_token_id, dtype=torch.long)
        for i, t in enumerate(ids):
            out[i, : len(t)] = torch.tensor(t, dtype=torch.long)
        return out

    # Whisper 风格 log-mel：STFT 取功率谱 → mel 滤波 → log10 → 动态范围压到 8 → 归一化到约 [-1, 1]。
    # 参数（n_fft/hop_length/mel_filters）全部取自基座自带的 feature extractor，保证与训练基座一致。
    def _log_mel(self, wav: torch.Tensor) -> torch.Tensor:
        w = wav.float()
        window = torch.hann_window(self.fe.n_fft)
        stft = torch.stft(w, self.fe.n_fft, self.fe.hop_length, window=window, return_complex=True)
        magnitudes = stft[..., :-1].abs() ** 2            # [201, T_mel]
        mel = self.mel_filters.T @ magnitudes             # [128, T_mel]
        log_spec = torch.clamp(mel, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def __call__(self, batch: list[dict]) -> dict:
        audios = [x["audio"][: self.max_samples] for x in batch]
        mels = [self._log_mel(a) for a in audios]
        lens = torch.tensor([m.shape[1] for m in mels], dtype=torch.long)
        T = int(lens.max())
        B = len(mels)
        input_features = torch.zeros(B, 128, T)
        for i, m in enumerate(mels):
            input_features[i, :, : m.shape[1]] = m
        # 局部目标只取每行挑中的那一个实体（由数据集决定），
        # 但完整实体列表要保留下来，供池负采样时排除“真词”避免假负样本。
        # local entity = one entity per row (dataset picks one), but keep full
        # entity list for pool-negative exclusion.
        local_texts = [x["local_text"] for x in batch]
        texts = [x["text"] for x in batch]
        return {
            "input_features": input_features,
            "feature_lens": lens,
            "text_ids": self._tok(texts),
            "hotword_ids": self._tok(local_texts),
            "entities": [x.get("entities") or [x["local_text"].strip().lower()] for x in batch],
        }
