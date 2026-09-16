"""Command-line inference for the published GLCLAP-Hotword adapter."""
# =============================================================================
# 中文说明（命令行推理）
#
#   输入：一个音频文件 + 若干候选热词。
#   输出：每个热词的 max-over-time 余弦分与峰值时间，按分数降序。
#
#   打分协议与训练/评估完全一致：cos = max_t <a_t, k_h>。
#   peak_time_seconds 由 帧→时间 反演得到，不能按 13 fps 均匀网格估算
#   （音频塔每个 1s mel 块出 13 帧，块内间隔 0.04s、跨块 0.08s，均匀网格会漂移近 1s）。
# =============================================================================
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from . import env
from .amphion_retriever import AmphionRetriever
from .checkpoint import load_adapters
from .data import _load_audio_16k
from .data_amphion import AmphionCollator


# 帧→时间戳反演：厂商长度公式是分段常函数，所以逐个 mel 长度扫描，
# 每当输出帧数 +1 就记录一个时间戳，从而得到每一帧的真实时间。
def frame_times(mel_len: int, hop: int = 160, sample_rate: int = 16000) -> list[float]:
    def output_length(length: int) -> int:
        leave = length % 100
        frames = (leave - 1) // 2 + 1
        return ((frames - 1) // 2 + 1 - 1) // 2 + 1 + (length // 100) * 13

    times, previous = [], output_length(0)
    for length in range(1, mel_len + 1):
        current = output_length(length)
        if current > previous:
            times.append(length * hop / sample_rate)
            previous = current
    return times


def default_checkpoint() -> Path:
    return env.repo_root() / "weights" / "glclap_hotword_adapter.safetensors"


# 解析候选热词：--hotword 可重复传入，--hotwords 支持逗号/换行分隔；
# 去重但保持首次出现顺序，空输入直接报错。
def parse_hotwords(values: list[str], joined: str | None) -> list[str]:
    result = [item.strip() for item in values if item.strip()]
    if joined:
        result.extend(item.strip() for item in joined.replace("\n", ",").split(","))
    result = [item for item in result if item]
    if not result:
        raise ValueError("provide at least one --hotword or --hotwords value")
    return list(dict.fromkeys(result))


@torch.inference_mode()
# 主流程：加载模型与 adapter → 编码音频 → 编码候选词 → 逐词算 max-over-time 分数并定位峰值帧。
def run(args) -> list[dict]:
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AmphionRetriever(
        args.qwen3asr_checkpoint, freeze_audio=True, freeze_text=True
    )
    load_adapters(model, args.checkpoint)
    model.to(device).eval()
    collator = AmphionCollator(
        args.qwen3asr_checkpoint, max_audio_seconds=args.max_audio_seconds
    )
    wave = _load_audio_16k(args.audio, max_samples=int(args.max_audio_seconds * 16000))
    batch = collator(
        [{"audio": wave, "text": "", "local_text": "", "entities": []}]
    )
    audio_frames, padding = model._encode_audio(
        batch["input_features"].to(device), batch["feature_lens"].to(device), device
    )
    audio_frames = F.normalize(model.audio_adapter(audio_frames), dim=-1)
    ids = collator._tok(args.hotword_values).to(device)
    text = F.normalize(model.text_adapter(model._text_embeddings(ids).float()), dim=-1)
    # [K, T] 的逐帧相似度矩阵：每一行是一个候选词在整段音频上的相似度曲线。
    # 先沿时间取 max 得到排序分，同时记下峰值帧号用于定位。
    curves = torch.einsum("btd,kd->bkt", audio_frames, text)[0]
    if padding is not None:
        curves = curves.masked_fill(padding[0].unsqueeze(0), float("-inf"))
    scores, frames = curves.max(dim=-1)

    times = frame_times(int(batch["feature_lens"][0]))
    output = []
    for word, score, frame in zip(args.hotword_values, scores.tolist(), frames.tolist()):
        output.append(
            {
                "hotword": word,
                "score": round(float(score), 6),
                "peak_time_seconds": round(float(times[frame]), 3)
                if frame < len(times)
                else None,
            }
        )
    # 按分数降序返回；注意这是候选词之间的相对排序，不是绝对“是否出现”的判断。
    output.sort(key=lambda item: item["score"], reverse=True)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Rank candidate hotwords for one audio file")
    parser.add_argument("--audio", required=True)
    parser.add_argument("--hotword", action="append", default=[])
    parser.add_argument("--hotwords", help="comma- or newline-separated candidate list")
    parser.add_argument("--checkpoint", default=str(default_checkpoint()))
    parser.add_argument(
        "--qwen3asr-checkpoint",
        default=str(env.model_dir() / "Qwen3-ASR-1.7B"),
    )
    parser.add_argument("--max-audio-seconds", type=float, default=14.0)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    args.hotword_values = parse_hotwords(args.hotword, args.hotwords)
    result = {
        "audio": str(Path(args.audio)),
        "checkpoint": str(Path(args.checkpoint)),
        "ranking": run(args),
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
