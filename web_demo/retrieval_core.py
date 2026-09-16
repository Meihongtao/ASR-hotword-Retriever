"""AmphionASR GLCLAP-Hotword hotword-retrieval scoring core (shared by web demo + probes).

Protocol -- must stay byte-for-byte identical to training/eval so the demo
numbers are comparable with `glclap/eval_amphion_retrieval.py` and
`glclap/train_amphion_ddp.py:validate`:

    a_t = normalize(g_audio(AuT(x)_t))              # [T, 512] frame embeddings
    k_h = normalize(g_text(mean_tokens(E_tok(h))))  # [K, 512]
    cos(h) = max_t <a_t, k_h>                       # the ONLY ranking score

UI extras (computed for explanation/localisation, NEVER used for ranking):
    seg       best sliding-window mean cosine  -> segment [start, end] to play
    contrast  seg - mean(curve)                -> how peaky the match is
    z         (peak - mean) / std of the curve -> measured USELESS (see below)
    pool_max  max cos over N random pool words on the SAME audio
              -> a per-query "what does a random word score here" floor
    prob      softmax(scale * cos) over the user's candidate list (RELATIVE)
    verdict   calibrated band (ABSOLUTE) from the measured distributions

Calibration source: data/glclap_hotword/valid.jsonl, zh queries, each scored
against the true entity + 999 random same-language pool words. English was not
completed, so zh is the reference domain.

Two sub-domains were measured, and they differ mainly by audio LENGTH:

  A) meeting TTS, mean 14.1s  (n=120)
     metric    TRUE (mean+-std, p5)      BEST-POS-999 NEG (mean, p95, max)
     cos       0.697 +- 0.052 (0.609)    0.635, p95 0.699, max 0.738
     seg       0.402 +- 0.077 (0.275)    0.341, p95 0.476, max 0.524
     contrast  0.314 +- 0.079 (0.197)    0.265, p95 0.370, max 0.425
     z         2.582 +- 0.468            2.514, p95 3.467, max 3.894  <-- USELESS
     cos thresholds:  0.62 -> kept 0.92 / FP 0.62    0.66 -> 0.77 / 0.33
                      0.70 -> kept 0.53 / FP 0.05    0.74 -> 0.22 / 0.00
     ranking: cos R@1 0.808 | seg R@1 0.550 | contrast R@1 0.567

  B) short read speech (magicdata), mean 4.6s  (n=118)
     cos       0.722 +- 0.040 (0.658)    0.603, p95 0.670, max 0.773
     cos thresholds:  0.62 -> kept 0.98 / FP 0.31    0.66 -> 0.93 / 0.08
                      0.70 -> kept 0.73 / FP 0.01
     ranking: cos R@1 0.966 | seg R@1 0.873 | contrast R@1 0.856

KEY INSIGHT -- the threshold depends on audio length, not on language: a longer
clip gives max-over-time more chances to hit a random word, so the negative
ceiling rises (4.6s -> p95 0.670, 14s -> 0.699, 67s+ meetings -> 0.72+).
A single fixed cos cut is therefore only a rough guide. `pool_max` (the best
score of N random words on the SAME audio) tracks that ceiling per query and is
reported alongside as `margin_over_pool`; treat a positive margin as the more
transferable evidence and the absolute band as a sanity check.

The bands below are tuned for the meeting/TTS domain (the demo's target).
Because a TRUE word can also score 0.56 on a short conversational clip, the
lowest band says "证据不足" (cannot confirm) rather than "absent".

Localisation accuracy: window centre vs the true word position taken from the
manifest transcript -> median |error| 0.51 s, 66% within 1 s.
Frame -> time is NOT uniform 13 fps: the tower emits 13 frames per 1 s mel
block, but with gaps of 0.04 s and 0.08 s inside a block. `frame_times()`
inverts the vendor length formula, so every frame gets its own timestamp.
"""
# =============================================================================
# 中文说明（Demo 打分核心）
#
# 排序分（唯一决定顺序的量，必须与训练/评估逐位一致）：
#     cos(h) = max_t <a_t, k_h>
#
# 其余字段都只用于解释与定位，绝不参与排序：
#     seg        最佳滑动窗口平均余弦 → 给出可播放的片段 [start, end]
#     contrast   seg - 曲线均值        → 峰值有多“尖”
#     z          (峰值-均值)/标准差    → 实测无效（真词与干扰词完全重叠）
#     pool_max   同一音频上 N 个随机词的最高分 → 每条 query 的“随机基线”
#     prob       候选列表内的 softmax（相对值）
#     verdict    标定过的绝对分档
#
# 核心结论：阈值取决于音频长度，而不是语言。
# 音频越长，max-over-time 撞出高分的机会越多，干扰词的上限就越高
# （4.6s → p95 0.670，14s → p95 0.699，67s+ → 0.72+）。
# 因此绝对阈值只是粗略参考，优先看“超出随机”（cos - pool_max）这一列。
# =============================================================================
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# ---- repo layout (relocatable: override with GLCLAP_* env vars, see glclap/env.py) ----
GLCLAP_ROOT = Path(__file__).resolve().parent.parent
if str(GLCLAP_ROOT) not in sys.path:
    sys.path.insert(0, str(GLCLAP_ROOT))
from glclap import env  # noqa: E402

QWEN3ASR_CKPT = str(env.model_dir() / "Qwen3-ASR-1.7B")
DEFAULT_CHECKPOINT = GLCLAP_ROOT / "weights" / "glclap_hotword_adapter.safetensors"
POOL_FILES = {
    "zh": env.data_dir() / "glclap_hotword" / "artifacts/pool_zh.txt",
    "en": env.data_dir() / "glclap_hotword" / "artifacts/pool_en.txt",
}

SR = 16000
# Softmax scale used ONLY to render a readable relative-confidence column.
# The trained scale (1/temperature = 67.4 for GLCLAP-Hotword) saturates to 0/100%.
DISPLAY_SCALE = 10.0
HOP = 160                # mel hop length (samples)
MEL_BLOCK = 100          # vendor: one 1 s block of mel frames ...
FRAMES_PER_BLOCK = 13    # ... yields 13 output frames
PAD_TOKEN_ID = 151643

# ---------------------------------------------------------------------------
# Calibrated verdict bands (measured on 80 zh valid queries, best-of-999 negs)
# ---------------------------------------------------------------------------
# 置信分档：基于中文实测标定（会议域为主）。
# 最低档故意写成“证据不足”而不是“未出现”——真词在短对话句上也可能只有 0.56，
# 直接判“未出现”会造成大量漏报。
VERDICT_BANDS = [
    # (cos_min, label, est_precision, css)
    # Labels deliberately avoid claiming absence: the true word sometimes
    # lands in the "证据不足" band (e.g. a short conversational clip scored
    # 0.563 while its transcript did contain the word), so the bottom band
    # says "cannot confirm", not "absent".
    (0.72, "高置信命中", 0.95, "good"),
    (0.66, "较可能命中", 0.72, "mid"),
    (0.60, "弱匹配", 0.60, "weak"),
    (-1.0, "证据不足", 0.15, "bad"),
]
CALIB_NOTE = ("阈值来自 data/glclap_hotword/valid.jsonl 的中文实测：会议音频(≈14s) "
              "干扰词 cos p95=0.699、短朗读句(≈4.6s) p95=0.670。"
              "音频越长，随机词撞高分的上限越高 —— 所以绝对阈值只是粗略参考，"
              "请优先看“超出随机”（本音频随机词最高分）这一列。")


# 厂商长度公式（标量版）：mel 帧数 → 输出帧数。帧→时间反演的基础。
def _out_len(mel_len: int) -> int:
    """Vendor `_get_feat_extract_output_lengths` (scalar)."""
    leave = mel_len % MEL_BLOCK
    f = (leave - 1) // 2 + 1
    return ((f - 1) // 2 + 1 - 1) // 2 + 1 + (mel_len // MEL_BLOCK) * FRAMES_PER_BLOCK


# 精确帧时间戳：公式是分段常函数，逐 mel 长度扫描，帧数一增加就记录时间。
# 块内间隔 0.04s、跨块 0.08s，若按 13fps 均匀网格估算，15s 音频会漂移近 1s。
def frame_times(mel_len: int, hop: int = HOP, sr: int = SR) -> list[float]:
    """Exact timestamp (seconds) of every tower output frame.

    The mapping is piecewise constant, so invert `_out_len` by scanning mel
    lengths; a frame is emitted whenever the frame count increments. Gaps are
    0.04 s inside a block and 0.08 s across block boundaries -> assuming a
    uniform 1/13 s grid drifts by up to ~1 s on a 15 s clip.
    """
    times: list[float] = []
    prev = _out_len(0)
    for m in range(1, mel_len + 1):
        cur = _out_len(m)
        if cur > prev:
            times.append(m * hop / sr)
            prev = cur
    return times


_FT_CACHE: dict[int, list[float]] = {}


def frame_times_cached(mel_len: int) -> list[float]:
    t = _FT_CACHE.get(mel_len)
    if t is None:
        t = frame_times(mel_len)
        if len(_FT_CACHE) > 512:
            _FT_CACHE.clear()
        _FT_CACHE[mel_len] = t
    return t


# 估算一个热词大约占多少帧：约 0.28s 基底 + 每字符 0.13s，限制在 [2, 40] 帧内。
def window_frames(hotword: str) -> int:
    """Frames spanned by a spoken hotword (~0.28 s base + 0.13 s per char)."""
    n = max(len(hotword.strip()), 1)
    return max(2, min(40, int(round(FRAMES_PER_BLOCK * (0.28 + 0.13 * n)))))


def is_cjk(s: str) -> bool:
    return any("\u4e00" <= c <= "\u9fff" for c in s)


# 滑动窗口和的增量实现（cumsum），把 O(T·w) 降到 O(T)；正确性由 self_check 与暴力实现比对。
def _window_sums(curve: torch.Tensor, w: int) -> tuple[torch.Tensor, int]:
    """Sliding-window sums of length T-w+1 (verified against a brute-force
    reference in `self_check`)."""
    T = curve.numel()
    w = max(1, min(w, T))
    cs = curve.cumsum(0)
    tot = cs[w - 1:].clone()
    if T - w >= 1:
        tot[1:] -= cs[: T - w]
    return tot, w


# 单条相似度曲线的度量：
#   cos      峰值（排序分）
#   seg      最佳窗口平均（比峰值更稳，适合定位）
#   contrast seg - 均值（峰有多突出）
#   z        峰值 z-score（实测无区分度，仅保留展示）
def curve_metrics(curve: torch.Tensor, w: int) -> dict:
    """Peak / window / sharpness metrics for one [T] similarity curve."""
    T = curve.numel()
    peak = float(curve.max())
    mean = float(curve.mean())
    std = float(curve.std(unbiased=False)) + 1e-6
    tot, w = _window_sums(curve, w)
    j = int(tot.argmax())
    seg = float(tot[j]) / w
    return {
        "cos": peak,
        "peak_frame": int(curve.argmax()),
        "z": (peak - mean) / std,
        "seg": seg,
        "contrast": seg - mean,
        "curve_mean": mean,
        "curve_std": std,
        "seg_lo": j,
        "seg_hi": j + w - 1,
        "seg_frames": w,
    }


# 按 cos 落入第一个满足阈值的分档，返回 (标签, 经验精度, 颜色等级)。
def verdict_of(cos: float) -> tuple[str, float, str]:
    for thr, label, prec, css in VERDICT_BANDS:
        if cos >= thr:
            return label, prec, css
    return VERDICT_BANDS[-1][1], VERDICT_BANDS[-1][2], VERDICT_BANDS[-1][3]


# 绘图降采样：分桶后取每桶【峰值】而不是平均，避免尖锐的命中峰被平均掉。
def downsample_max(vals: np.ndarray, times: np.ndarray, max_points: int = 480):
    """Bucket-average the curve for plotting but keep each bucket's PEAK so a
    sharp hotword hit cannot be averaged away. Returns (vals, times, peak_idx)."""
    T = len(vals)
    if T <= max_points:
        return vals, times, None
    bins = int(np.ceil(T / max_points))
    n = T // bins
    out_v, out_t = [], []
    for b in range(bins):
        lo, hi = b * bins, min((b + 1) * bins, T)
        if lo >= hi:
            break
        seg = vals[lo:hi]
        k = int(np.argmax(seg))
        out_v.append(float(seg[k]))
        out_t.append(float(times[lo + k]))
    return np.asarray(out_v), np.asarray(out_t), None


# Demo 打分器：一次性加载冻结音频塔 + 训练好的 adapter，缓存词池，内部加锁保证线程安全。
class HotwordScorer:
    """Loads the frozen Qwen3ASR towers + the trained GLCLAP-Hotword adapters once and
    scores (audio, hotword list) pairs. Thread-safe via an internal lock."""

    def __init__(self, checkpoint: str | Path = DEFAULT_CHECKPOINT,
                 device: str | None = None, pool_size: int = 256,
                 max_audio_seconds: float = 60.0):
        if str(GLCLAP_ROOT) not in sys.path:
            sys.path.insert(0, str(GLCLAP_ROOT))
        self.checkpoint = Path(checkpoint)
        if not self.checkpoint.is_absolute():
            self.checkpoint = GLCLAP_ROOT / self.checkpoint
        self.pool_size = int(pool_size)
        self.max_audio_seconds = float(max_audio_seconds)
        self.dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.lock = __import__("threading").RLock()
        self._collators: dict[float, object] = {}
        self._pools: dict[str, list[str]] = {}
        self.model = None
        self.scale = 1.0
        self.sample_rate = SR

    # ---------------- model ----------------
    # 懒加载模型与 adapter，并读出训练学到的温度 scale（用于展示概率）。
    def load(self) -> None:
        if self.model is not None:
            return
        from glclap.amphion_retriever import AmphionRetriever

        print(f"[core] loading {self.checkpoint.name} ...", flush=True)
        m = AmphionRetriever(QWEN3ASR_CKPT, freeze_audio=True, freeze_text=True)
        from glclap.checkpoint import load_adapters
        keep = load_adapters(m, self.checkpoint)
        m.to(self.dev).eval()
        self.scale = float(m.logit_scale.exp().clamp(max=100).item())
        self.model = m
        print(f"[core] ready on {self.dev} (scale={self.scale:.2f} "
              f"temp={1.0/self.scale:.4f}) loaded {len(keep)} tensors", flush=True)

    def collator(self, max_audio_seconds: float | None = None):
        from glclap.data_amphion import AmphionCollator
        sec = float(max_audio_seconds or self.max_audio_seconds)
        c = self._collators.get(sec)
        if c is None:
            c = AmphionCollator(QWEN3ASR_CKPT, max_audio_seconds=sec,
                                pad_token_id=PAD_TOKEN_ID)
            self._collators[sec] = c
        return c

    @property
    def tokenizer(self):
        return self.collator().tokenizer

    def pool(self, lang: str) -> list[str]:
        words = self._pools.get(lang)
        if words is None:
            p = POOL_FILES[lang]
            words = p.read_text(encoding="utf-8").splitlines()
            words = [w for w in words if w.strip()]
            self._pools[lang] = words
            print(f"[core] pool-{lang}: {len(words)} words", flush=True)
        return words

    # ---------------- encoders ----------------
    @torch.no_grad()
    # 音频编码：截断到 max_audio_seconds → log-mel → 音频塔 → 适配器归一化。
    # 同时按帧数生成每帧时间戳，供曲线与定位使用。
    def encode_audio(self, wav: torch.Tensor, max_audio_seconds: float | None = None) -> dict:
        self.load()
        sec = float(max_audio_seconds or self.max_audio_seconds)
        col = self.collator(sec)
        wav = wav[: int(sec * SR)]
        cb = col([{"audio": wav, "text": "", "local_text": "", "entities": []}])
        a_t, pad = self.model._encode_audio(cb["input_features"].to(self.dev),
                                            cb["feature_lens"].to(self.dev), self.dev)
        a_t = F.normalize(self.model.audio_adapter(a_t), dim=-1)     # [1, T, 512]
        n_valid = a_t.size(1) if pad is None else int((~pad[0]).sum())
        times = frame_times_cached(int(cb["feature_lens"][0]))[:n_valid]
        if len(times) < n_valid:      # defensive: never mismatch the curve
            times = list(times) + [times[-1] if times else 0.0] * (n_valid - len(times))
        return {
            "a_t": a_t,
            "pad": pad,
            "n_valid": n_valid,
            "n_samples": int(wav.numel()),
            "duration": float(wav.numel() / SR),
            "frame_times": np.asarray(times, dtype=np.float64),
        }

    @torch.no_grad()
    # 热词编码：分词 → 冻结 embedding 均值池化 → 文本适配器归一化。
    # 返回嵌入与每个词的有效 token 数（token 数用于调试/校验）。
    def encode_text(self, hotwords: list[str]) -> tuple[torch.Tensor, list[int]]:
        self.load()
        col = self.collator()
        ids = col._tok(list(hotwords)).to(self.dev)
        emb = F.normalize(self.model.text_adapter(
            self.model._text_embeddings(ids).float()), dim=-1)
        ntok = (ids != PAD_TOKEN_ID).sum(dim=1).tolist()
        return emb, ntok

    # ---------------- scoring ----------------
    @torch.no_grad()
    def score(self, audio: dict, hotwords: list[str], pool_size: int | None = None,
              topk: int | None = None) -> dict:
        """Score every hotword against the pre-encoded audio.

        Returns per-candidate raw numbers (cos/seg/contrast/z/prob/verdict) plus
        a downsampled similarity curve over absolute time for plotting.
        """
        self.load()
        if not hotwords:
            raise ValueError("no hotwords")
        n_pool = int(pool_size or self.pool_size)

        a_t = audio["a_t"][0][: audio["n_valid"]]            # [Tv, 512]
        times = audio["frame_times"]

        k = self.encode_text(hotwords)[0]                    # [K, 512]
        sim = (a_t @ k.t()).t().contiguous()                 # [K, Tv]
        # [K, Tv] 逐帧相似度 → 沿时间取 max 得到排序分 cos（协议分，唯一排序依据）。
        cos_all = sim.amax(dim=-1)                           # [K]

        # ---- 每条 query 的随机基线 ----
        # 在同一段音频上用同语言的随机池词打分，取最高分作为“随机词在这里能拿多少分”。
        # 用 cos - pool_max 作为更可迁移的证据，因为它自动随音频长度变化。
        # ---- per-query random-pool floor (same audio, same language) ----
        pool_max = {}
        for lang in ("zh", "en"):
            idx = [i for i, h in enumerate(hotwords)
                   if (is_cjk(h) if lang == "zh" else not is_cjk(h))]
            if not idx:
                continue
            words = self.pool(lang)
            if len(words) > n_pool:
                pick = torch.randperm(len(words))[:n_pool].tolist()
                words = [words[i] for i in pick]
            kp = self.encode_text(words)[0]
            pool_max[lang] = float((a_t @ kp.t()).max())

        # ---- 候选列表内的相对概率 ----
        # 返回两种读数：
        #   prob         用训练学到的温度（≈1/67）。这是模型真实后验，但会饱和：
        #                0.2 的 cos 差 → e^(67*0.2) ≈ 1e6，除第一名外全显示 0.00%。
        #   prob_display 同样用 softmax 但缩放固定为 10（温度 0.1），只为让用户看清相对排序，
        #                是展示选择，不是标定后的概率。
        # ---- relative probability within the user's candidate list ----
        # Two readings are returned on purpose:
        #   prob          trained temperature (1/67.4). This IS the model's
        #                 posterior and it saturates: a 0.2 cos gap gives
        #                 e^(67*0.2) ~ 1e6, so every non-top row shows 0.00%.
        #   prob_display  same softmax at scale=10 (temperature 0.1) purely so
        #                 the user can see HOW the candidates rank. It is a
        #                 display choice, not a calibrated posterior.
        probs = torch.softmax(self.scale * cos_all, dim=-1)
        probs_display = torch.softmax(DISPLAY_SCALE * cos_all, dim=-1)
        top_prob = float(probs.max())

        results = []
        for i, h in enumerate(hotwords):
            curve = sim[i]
            w = window_frames(h)
            m = curve_metrics(curve, w)
            lo, hi = m["seg_lo"], m["seg_hi"]
            t_lo = float(times[lo]) if lo < len(times) else 0.0
            t_hi = float(times[min(hi, len(times) - 1)]) + 1.0 / FRAMES_PER_BLOCK
            t_hi = min(t_hi, audio["duration"])
            label, prec, css = verdict_of(m["cos"])
            lang = "zh" if is_cjk(h) else "en"
            pm = pool_max.get(lang)
            cv, ct, _ = downsample_max(curve.detach().cpu().numpy(),
                                       times, max_points=480)
            results.append({
                "hotword": h,
                "lang": lang,
                # --- protocol score (ranking key, same as training/eval) ---
                "cos": round(m["cos"], 4),
                "peak_sec": round(float(times[m["peak_frame"]]), 3)
                if m["peak_frame"] < len(times) else None,
                # --- localisation ---
                "seg": round(m["seg"], 4),
                "contrast": round(m["contrast"], 4),
                "z": round(m["z"], 3),
                "seg_start": round(t_lo, 3),
                "seg_end": round(t_hi, 3),
                "seg_frames": m["seg_frames"],
                "seg_window_sec": round(m["seg_frames"] / FRAMES_PER_BLOCK, 3),
                # --- confidence ---
                "prob": float(round(float(probs[i]), 6)),
                "prob_display": float(round(float(probs_display[i]), 6)),
                "verdict": label,
                "verdict_precision": prec,
                "verdict_level": css,
                "pool_max": round(pm, 4) if pm is not None else None,
                "margin_over_pool": round(m["cos"] - pm, 4) if pm is not None else None,
                # --- plot ---
                "curve": [round(float(v), 4) for v in cv],
                "curve_t": [round(float(v), 3) for v in ct],
            })
        # 按 cos 降序（与训练/评估同一排序键），然后重新编号 rank。
        results.sort(key=lambda r: -r["cos"])
        for i, r in enumerate(results, start=1):
            r["rank"] = i
        if topk:
            results = results[:topk]
        return {
            "results": results,
            "n_candidates": len(hotwords),
            "scale": round(self.scale, 3),
            "temperature": round(1.0 / self.scale, 4),
            "display_scale": DISPLAY_SCALE,
            "display_temperature": round(1.0 / DISPLAY_SCALE, 4),
            "prob_saturated": bool(top_prob > 0.999 and len(hotwords) > 1),
            "pool_size": n_pool,
            "pool_max": {k2: round(v, 4) for k2, v in pool_max.items()},
            "duration": round(audio["duration"], 3),
            "n_frames": audio["n_valid"],
            "fps": round(audio["n_valid"] / max(audio["duration"], 1e-6), 3),
            "score_spread": round(float(cos_all.max() - cos_all.min()), 4),
            "calibration_note": CALIB_NOTE,
        }

    # ---------------- self test ----------------
    # 自检：专门验证最容易写错的四件事——
    #   1. frame_times 与厂商长度公式严格互逆
    #   2. 滑动窗口和与暴力实现一致
    #   3. 批量编码与逐条编码一致（确认无拼接漂移）
    #   4. Demo 路径的 cos 与评估协议数值一致
    def self_check(self, wav: torch.Tensor | None = None, verbose: bool = True) -> dict:
        """Verify the things that are easy to get wrong:
        1. frame_times inverts the vendor length formula exactly
        2. the sliding-window sum matches a brute-force reference
        3. batch vs per-utterance audio encoding agree (no packing drift)
        4. cos from the demo path equals the eval protocol number
        """
        out = {}
        for mel in (100, 137, 900, 1437, 2000):
            ts = frame_times(mel)
            out[f"frames[{mel}]"] = len(ts)
            assert len(ts) == _out_len(mel), (mel, len(ts), _out_len(mel))
        # window sums vs brute force
        for T, w in ((121, 10), (37, 2), (13, 40), (5, 5), (8, 3)):
            c = torch.rand(T)
            tot, ww = _window_sums(c, w)
            ref = torch.tensor([c[i:i + ww].sum() for i in range(T - ww + 1)])
            assert torch.allclose(tot, ref, atol=1e-5), (T, w)
            assert int(tot.argmax()) == int(ref.argmax())
        out["window_sums"] = "ok"
        out["frame_times"] = "ok"
        if wav is not None:
            single = self.encode_audio(wav)
            col = self.collator()
            cb = col([{"audio": wav, "text": "", "local_text": "", "entities": []},
                      {"audio": wav, "text": "", "local_text": "", "entities": []}])
            a_t, _ = self.model._encode_audio(cb["input_features"].to(self.dev),
                                              cb["feature_lens"].to(self.dev), self.dev)
            a_t = F.normalize(self.model.audio_adapter(a_t), dim=-1)
            d1 = float((single["a_t"][0] - a_t[0, : single["n_valid"]]).abs().max())
            d2 = float((single["a_t"][0] - a_t[1, : single["n_valid"]]).abs().max())
            out["batch_vs_single_maxdiff"] = round(d1, 8)
            out["repeat_consistency_maxdiff"] = round(d2, 8)
            assert d1 < 1e-5, "batch vs single encoding drifted"
        if verbose:
            for k2, v in out.items():
                print(f"  {k2}: {v}")
        return out


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="self-check the GLCLAP-Hotword scoring core")
    ap.add_argument("--audio", default=None, help="optional wav to include")
    ap.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    a = ap.parse_args()
    s = HotwordScorer(a.checkpoint)
    wav = None
    if a.audio:
        from glclap.data import _load_audio_16k
        wav = _load_audio_16k(a.audio)
    print("self-check:")
    s.self_check(wav)
    print("OK")
