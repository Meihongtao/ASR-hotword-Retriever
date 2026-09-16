"""Web demo: upload audio + type hotwords -> see GLCLAP-Hotword retrieval.

Run on the GPU box (bind to localhost unless you add authentication):
  cd <repo_root>  # see glclap/env.py
  CUDA_VISIBLE_DEVICES=6 python3.10 web_demo/app.py --port 8899

Then on your laptop:
  ssh -L 8899:127.0.0.1:8899 4090-48
  open http://127.0.0.1:8899

Scoring lives in `web_demo/retrieval_core.py` (self-checkable, mirrors the
training/eval protocol exactly). This file is only HTTP + audio plumbing.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import threading
import traceback
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

GLCLAP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(GLCLAP_ROOT))
sys.path.insert(0, str(GLCLAP_ROOT / "web_demo"))
from glclap import env  # noqa: E402

from retrieval_core import (  # noqa: E402
    DEFAULT_CHECKPOINT, VERDICT_BANDS, CALIB_NOTE, HotwordScorer,
)

SAMPLES = env.data_dir() / "glclap_hotword" / "valid.jsonl"

MODELS = {
    "GLCLAP-Hotword (best, step 18190)": DEFAULT_CHECKPOINT,
}
DEFAULT_MODEL = "GLCLAP-Hotword (best, step 18190)"
DEFAULT_HOTWORDS = (
    "喷射混凝土 五跳 外柱 钢筋绑扎 王洋 成都 战场 费玉清 孙红 "
    "升主动脉 千层阶 层析柱 燃点试验 电动煨弯机 俯拍"
)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 512 * 1024 * 1024  # 512 MB audio cap
# Reload templates on every request and disable browser caching so editing
# index.html during a demo session takes effect without restarting the server.
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.jinja_env.auto_reload = True

_lock = threading.Lock()
_scorer: HotwordScorer | None = None


# 全局单例打分器：模型只加载一次，多个请求复用。
def get_scorer() -> HotwordScorer:
    """Single lazily-loaded scorer shared by all requests (model load is slow)."""
    global _scorer
    if _scorer is None:
        with _lock:
            if _scorer is None:
                s = HotwordScorer(DEFAULT_CHECKPOINT)
                s.load()
                _scorer = s
    return _scorer


# ---------------------------------------------------------------- audio input
# 读入浏览器上传/录音得到的音频：统一转 16 kHz 单声道，并按 max_seconds 截断。
def load_audio_any(wav_path: str, max_seconds: float):
    """Read any uploaded audio as mono 16 kHz, transcoding with ffmpeg when
    soundfile cannot decode the container (webm/opus from the browser mic)."""
    from glclap.data import _load_audio_16k

    max_samples = int(max_seconds * 16000)
    try:
        return _load_audio_16k(wav_path, max_samples=max_samples)
    except Exception:
        out = os.path.splitext(wav_path)[0] + "_ffmpeg16k.wav"
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", wav_path, "-ar", "16000", "-ac", "1", "-f", "wav", out],
            capture_output=True,
        )
        if r.returncode != 0:
            raise RuntimeError(
                "无法解码该音频（soundfile 与 ffmpeg 都失败）: "
                + r.stderr.decode(errors="ignore")[-300:]
            )
        try:
            return _load_audio_16k(out, max_samples=max_samples)
        finally:
            try:
                os.unlink(out)
            except OSError:
                pass


# 拆分用户输入的候选热词（支持换行/逗号/顿号等分隔），去重并保持顺序。
def split_hotwords(raw: str) -> list[str]:
    """Split on commas/spaces/newlines, dedupe case-insensitively, keep order.

    Multiple spaces inside a candidate are preserved (so 'Fin FET' stays one
    hotword) but a run of separators splits it.
    """
    import re

    parts = [p.strip() for p in re.split(r"[,，;；\t\n]+", raw)]
    words: list[str] = []
    for p in parts:
        p = re.sub(r"\s{2,}", " ", p).strip()
        if p:
            words.append(p)
    seen, out = set(), []
    for w in words:
        k = w.casefold()
        if k not in seen:
            seen.add(k)
            out.append(w)
    return out


# ---------------------------------------------------------------------- routes
@app.after_request
def _no_cache(resp):
    if resp.mimetype in ("text/html", "application/json"):
        resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
        resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/")
def index():
    return render_template("index.html", models=list(MODELS.keys()),
                           default_hotwords=DEFAULT_HOTWORDS,
                           default_model=DEFAULT_MODEL)


@app.route("/api/info")
def info():
    return jsonify({
        "models": list(MODELS.keys()),
        "checkpoints": MODELS,
        "default_model": DEFAULT_MODEL,
        "loaded": _scorer is not None,
        "calibration": {"bands": VERDICT_BANDS, "note": CALIB_NOTE},
    })


@app.route("/api/selfcheck")
def selfcheck():
    """Expose the core self-check so the scoring path can be verified in-browser."""
    try:
        s = get_scorer()
        return jsonify(s.self_check(verbose=False))
    except Exception as e:  # pragma: no cover
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


def _prefer_fast(rows: list[dict]) -> list[dict]:
    """Compatibility hook; released GLCLAP-Hotword audio is already locally materialized."""
    return rows


@app.route("/api/sample")
def sample():
    """Random validation utterance + its true hotwords, for a one-click try.

    Uses the materialized GLCLAP-Hotword validation manifest.
    """
    import random

    rows = []
    with SAMPLES.open(encoding="utf-8") as f:
        for ln in f:
            if not ln.strip():
                continue
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("entities"):
                rows.append(r)
    if not rows:
        return jsonify({"error": "no sample rows"}), 500
    pool = _prefer_fast(rows)
    r = random.choice(pool)
    return jsonify({
        "audio": r["audio"],
        "text": r.get("text", ""),
        "entities": r["entities"],
        "language": r.get("language", ""),
        "sample_pool": len(pool),
        "sample_total": len(rows),
    })


@app.route("/api/audio")
def audio_file():
    """Stream a server-side wav to the browser (used by the sample button)."""
    from urllib.parse import quote

    p = request.args.get("path", "")
    f = Path(p).resolve()
    allowed_roots = (env.data_dir().resolve(),)
    if not any(f.is_relative_to(root) for root in allowed_roots) or not f.is_file():
        return jsonify({"error": "bad path"}), 400
    ascii_name = f.name.encode("ascii", "ignore").decode("ascii")
    if ascii_name == f.name:
        cd = f'inline; filename="{f.name}"'
    else:
        cd = ("inline; filename=\"%s\"; filename*=UTF-8''%s"
              % (ascii_name or "audio.wav", quote(f.name)))
    return f.read_bytes(), 200, {
        "Content-Type": "audio/wav" if f.suffix.lower() == ".wav" else "audio/mpeg",
        "Content-Disposition": cd,
    }


# 核心接口：接收音频与候选热词，调用打分核心，返回排序结果与曲线数据。
@app.route("/api/recall", methods=["POST"])
def recall():
    model_name = request.form.get("model", DEFAULT_MODEL)
    topk = int(request.form.get("topk", 10))
    max_seconds = float(request.form.get("max_audio_seconds", 60.0))
    pool_size = int(request.form.get("pool_size", 256))
    hotwords = split_hotwords(request.form.get("hotwords", ""))
    if not hotwords:
        hotwords = split_hotwords(DEFAULT_HOTWORDS)
    if model_name not in MODELS:
        return jsonify({"error": f"unknown model {model_name}"}), 400

    up = request.files.get("audio")
    if up is None or up.filename == "":
        return jsonify({"error": "请先选择音频文件"}), 400

    suffix = os.path.splitext(secure_filename(up.filename))[1] or ".wav"
    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    try:
        up.save(tmp.name)
        tmp.close()
        s = get_scorer()
        wav = load_audio_any(tmp.name, max_seconds)
        if wav.numel() < 1600:
            return jsonify({"error": "音频太短（<0.1s）"}), 400
        # warn the user when their upload was cut by the truncation limit
        raw_dur = None
        try:
            import soundfile as sf
            raw_dur = float(sf.info(tmp.name).duration)
        except Exception:
            pass
        with s.lock:
            audio = s.encode_audio(wav, max_audio_seconds=max_seconds)
            out = s.score(audio, hotwords, pool_size=pool_size, topk=topk)
        out.update({
            "model": model_name,
            "checkpoint": str(s.checkpoint.name),
            "hotwords_all": hotwords,
            "uploaded_duration": round(raw_dur, 2) if raw_dur else None,
        })
        return jsonify(out)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500
    finally:
        try:
            os.unlink(tmp.name)
        except OSError:
            pass


# 启动入口：可 --preload 预热模型（避免首次请求等待加载）。
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, default=8899)
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--preload", action="store_true", help="load the model on startup")
    p.add_argument("--no-preload", action="store_true")
    # Backwards compatibility: earlier launches used these flags
    # (`--gpu-id 2 --gpu 0.38`). Keep accepting them so an existing start
    # command / supervisor script does not break after this rewrite.
    p.add_argument("--gpu-id", type=int, default=None,
                   help="deprecated: physical GPU index (same as CUDA_VISIBLE_DEVICES)")
    p.add_argument("--gpu", type=float, default=None,
                   help="deprecated: accepted and ignored (memory fraction no longer used)")
    args = p.parse_args()
    if args.gpu_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
        print(f"[web_demo] CUDA_VISIBLE_DEVICES={args.gpu_id} (from --gpu-id)", flush=True)
    if args.gpu is not None:
        print(f"[web_demo] note: --gpu {args.gpu} ignored (kept for compatibility)", flush=True)
    if not args.no_preload:
        get_scorer()
    print(f"[web_demo] listening on {args.host}:{args.port}", flush=True)
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
