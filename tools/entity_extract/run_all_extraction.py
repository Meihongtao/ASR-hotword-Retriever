#!/usr/bin/env python3
"""Reproduce the entity extraction used to build the GLCLAP-Hotword indexes.

功能:
  1. 检查本地 OpenAI-compatible vLLM 服务（GLCLAP-Hotword 使用 Qwen3.5-4B）
  2. 两个数据集完整处理:
     - common-voice-en: 合并 6 个 tsv, 按 path 去重, 用 clip_durations 索引验证存在
     - magicdata: 读取 TRANS.txt (中文)
  3. 输出统一 JSONL: {"audio", "text", "language", "entities"}
  4. 断点续跑 (--resume): 跳过已处理 path
  5. 分片写入, 崩溃安全; 自动重试; 进度报告
  6. 可选只跑某数据集/某模型, 限量测试

The exact prompt, immutable model revision, decoding options, resume behaviour,
and launch commands are documented in this directory's README.md.
"""
import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock

# ------------------------- 配置 -------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROMPT_FILE = os.path.join(SCRIPT_DIR, "hotword_prompt.txt")
OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")

CV_SPLITS = ["train", "dev", "test", "validated", "invalidated", "other"]

# 服务端点
APIS = {
    "qwen3-1.7b": "http://127.0.0.1:8000/v1/chat/completions",
    "qwen3.5-4b": "http://127.0.0.1:8001/v1/chat/completions",
}

# ------------------------- 参数 -------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Unified full extraction for both datasets")
    p.add_argument("--commonvoice-root",
                   help="extracted Common Voice 26.0 English root containing train.tsv and clips/")
    p.add_argument("--magicdata-root",
                   help="extracted MagicData SLR68 root containing train/TRANS.txt")
    p.add_argument("--prompt", default=PROMPT_FILE)
    p.add_argument("--model-id", default="Qwen/Qwen3.5-4B",
                   help="public model identifier recorded in provenance metadata")
    p.add_argument(
        "--model-revision",
        default="fcb1a040bb418b0b8add6f6f6c475386abc2cb97",
        help="immutable ModelScope revision used for the GLCLAP-Hotword entity indexes",
    )
    p.add_argument("--cv-model", choices=["qwen3-1.7b", "qwen3.5-4b", "none"], default="qwen3.5-4b",
                   help="served model name for Common Voice (GLCLAP-Hotword: qwen3.5-4b)")
    p.add_argument("--md-model", choices=["qwen3-1.7b", "qwen3.5-4b", "none"], default="qwen3.5-4b",
                   help="model for magicdata (default qwen3.5-4b)")
    p.add_argument("--cv-api", default=None, help="API endpoint for common-voice (default: auto from model)")
    p.add_argument("--md-api", default=None, help="API endpoint for magicdata (default: auto from model)")
    p.add_argument("--only", choices=["common-voice", "magicdata"], default=None,
                   help="process only one dataset")
    p.add_argument("-w", "--workers", type=int, default=32, help="HTTP request concurrency")
    p.add_argument("--limit", type=int, default=0, help="process at most N rows per dataset (0=all)")
    p.add_argument("--no-resume", action="store_true", help="disable resume (reprocess all)")
    p.add_argument("--no-check-api", action="store_true", help="skip API check")
    p.add_argument("-o", "--outdir", default=OUTPUT_DIR, help=f"output dir (default {OUTPUT_DIR})")
    p.add_argument("--max-tokens", type=int, default=300)
    return p.parse_args()

# ------------------------- 工具 -------------------------
def load_prompt(path):
    with open(path, encoding="utf-8") as handle:
        return handle.read()

def check_api(api, timeout=10):
    url = api.replace("/v1/chat/completions", "/v1/models")
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            return r.status == 200
    except Exception:
        return False

# --- 数据收集 ---
def collect_cv_rows(data_root, limit):
    """Common Voice: merge all tsv, dedupe by path."""
    durations = set()
    with open(os.path.join(data_root, "clip_durations.tsv"), encoding="utf-8") as f:
        f.readline()
        for line in f:
            p = line.strip().split("\t")
            if p:
                durations.add(p[0])
    print(f"  clip_durations index: {len(durations)}", flush=True)

    seen, rows = set(), []
    for fn in CV_SPLITS:
        path = os.path.join(data_root, f"{fn}.tsv")
        if not os.path.exists(path):
            continue
        print(f"  reading {fn}.tsv ...", flush=True)
        with open(path, encoding="utf-8") as f:
            header = f.readline().rstrip("\n").split("\t")
            pi, si = header.index("path"), header.index("sentence")
            for line in f:
                parts = line.rstrip("\n").split("\t")
                if len(parts) <= max(pi, si):
                    continue
                p, s = parts[pi], parts[si]
                if not p or not s or p in seen or p not in durations:
                    continue
                seen.add(p)
                rows.append({"path": p, "text": s, "language": "English"})
                if limit and len(rows) >= limit:
                    return rows
    return rows

def collect_md_rows(data_root, limit):
    """MagicData: read TRANS.txt."""
    rows = []
    tf = os.path.join(data_root, "train", "TRANS.txt")
    with open(tf, encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        ui, si, ti = header.index("UtteranceID"), header.index("SpeakerID"), header.index("Transcription")
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) <= max(ui, si, ti):
                continue
            utt, spk, text = parts[ui], parts[si], parts[ti]
            if not utt or not text:
                continue
            rows.append({"path": os.path.join(spk, utt), "text": text, "language": "Chinese"})
            if limit and len(rows) >= limit:
                break
    return rows

def load_done_paths(outfile, audio_base):
    done = set()
    if not os.path.exists(outfile):
        return done
    with open(outfile, encoding="utf-8") as f:
        for line in f:
            try:
                audio = str(json.loads(line)["audio"])
                done.add(os.path.relpath(audio, audio_base).replace("\\", "/"))
            except Exception:
                pass
    return done

def call_llm(api, model, prompt, text, max_tokens, retries=3):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": f"{prompt}\n\nText: \"{text}\"\nReturn only the JSON object."}],
        "max_tokens": max_tokens, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    body = json.dumps(payload).encode()
    for attempt in range(retries):
        try:
            req = urllib.request.Request(api, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=300) as resp:
                data = json.load(resp)
            content = data["choices"][0]["message"]["content"]
            m = re.search(r"\{.*\}", content, re.S)
            if not m:
                raise ValueError(f"model returned no JSON object: {content[:200]!r}")
            obj = json.loads(m.group(0))
            hw = obj.get("hotwords", [])
            if not isinstance(hw, list) or not all(isinstance(x, str) for x in hw):
                raise ValueError(f"invalid hotwords payload: {obj!r}")
            return [x.strip() for x in hw if x.strip()]
        except Exception as exc:
            if attempt < retries - 1:
                time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"entity extraction failed after {retries} attempts") from exc

# ------------------------- 处理主流程 -------------------------
def process_dataset(name, rows, audio_base, api, model, prompt, args, outfile, lock):
    """Process one dataset: filter resume, batch, write."""
    done_paths = set()
    if not args.no_resume:
        done_paths = load_done_paths(outfile, audio_base)
        before = len(rows)
        rows = [r for r in rows if r["path"].replace("\\", "/") not in done_paths]
        print(f"  resume: skipped {before - len(rows)} done, {len(rows)} to process", flush=True)

    if not rows:
        print(f"  [{name}] nothing to process", flush=True)
        return

    # audio abs path
    for r in rows:
        r["audio"] = os.path.join(audio_base, r["path"])

    total, done = len(rows), 0
    t0 = time.time()
    for bstart in range(0, total, 200):
        chunk = rows[bstart:bstart + 200]
        results = {}
        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(call_llm, api, model, prompt, r["text"], args.max_tokens): i for i, r in enumerate(chunk)}
            for fut in as_completed(futs):
                i = futs[fut]
                results[i] = fut.result()
                done += 1
                if done % 500 == 0 or done == total:
                    el = time.time() - t0
                    rate = done / el if el > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    print(f"  [{name}] {done}/{total} ({rate:.1f} req/s, ETA {eta/60:.1f} min)", flush=True)

        with lock:
            with open(outfile, "a", encoding="utf-8") as f:
                for i, r in enumerate(chunk):
                    rec = {"audio": r["audio"], "text": r["text"], "language": r["language"],
                           "entities": results.get(i, [])}
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                f.flush()

    print(f"  [{name}] DONE: {total} records -> {outfile} ({(time.time()-t0)/60:.1f} min)", flush=True)

def main():
    args = parse_args()
    selected_cv = args.only in (None, "common-voice") and args.cv_model != "none"
    selected_md = args.only in (None, "magicdata") and args.md_model != "none"
    if selected_cv and not args.commonvoice_root:
        raise SystemExit("--commonvoice-root is required when Common Voice is selected")
    if selected_md and not args.magicdata_root:
        raise SystemExit("--magicdata-root is required when MagicData is selected")
    if not selected_cv and not selected_md:
        raise SystemExit("no dataset selected: enable a model for at least one dataset")
    prompt = load_prompt(args.prompt)
    os.makedirs(args.outdir, exist_ok=True)
    lock = Lock()

    # Resolve API endpoints
    cv_api = args.cv_api or APIS.get(args.cv_model)
    md_api = args.md_api or APIS.get(args.md_model)

    # API check
    checks = []
    if selected_cv:
        checks.append((args.cv_model, cv_api))
    if selected_md:
        checks.append((args.md_model, md_api))
    if not args.no_check_api:
        for m, api in checks:
            print(f"Checking {m} at {api} ...", flush=True)
            if not check_api(api):
                print(f"  ERROR: {m} API not reachable. Start the vLLM service first.", flush=True)
                sys.exit(1)
            print(f"  {m} OK", flush=True)

    # --- Common Voice ---
    if selected_cv:
        print(f"\n{'='*60}\n[common-voice-en] collecting rows ...\n{'='*60}", flush=True)
        rows = collect_cv_rows(args.commonvoice_root, args.limit)
        print(f"  total unique rows: {len(rows)}", flush=True)
        outfile = os.path.join(args.outdir, "common_voice_en_full.jsonl")
        process_dataset("common-voice-en", rows, os.path.join(args.commonvoice_root, "clips"),
                        cv_api, args.cv_model, prompt, args, outfile, lock)

    # --- MagicData ---
    if selected_md:
        print(f"\n{'='*60}\n[magicdata_mandarin] collecting rows ...\n{'='*60}", flush=True)
        rows = collect_md_rows(args.magicdata_root, args.limit)
        print(f"  total rows: {len(rows)}", flush=True)
        outfile = os.path.join(args.outdir, "magicdata_mandarin_full.jsonl")
        process_dataset("magicdata-mandarin", rows, os.path.join(args.magicdata_root, "train"),
                        md_api, args.md_model, prompt, args, outfile, lock)

    provenance = {
        "model_id": args.model_id,
        "model_revision": args.model_revision,
        "served_models": {"commonvoice": args.cv_model, "magicdata": args.md_model},
        "temperature": 0.0,
        "max_tokens": args.max_tokens,
        "thinking": False,
        "prompt_file": os.path.basename(args.prompt),
        "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
        "commonvoice_version": "Common Voice 26.0 English",
        "magicdata_version": "MagicData Mandarin Chinese Read Speech Corpus (SLR68)",
    }
    with open(os.path.join(args.outdir, "provenance.json"), "w", encoding="utf-8") as handle:
        json.dump(provenance, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")

    print("\n=== ALL DONE ===")
    for fn in sorted(os.listdir(args.outdir)):
        if fn.endswith(".jsonl"):
            fp = os.path.join(args.outdir, fn)
            print(f"  {fn}: {sum(1 for _ in open(fp))} records")

if __name__ == "__main__":
    main()
