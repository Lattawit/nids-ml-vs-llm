"""Resumable, strictly sequential LLM runner for the revision experiments.

Each job = (model, mode, seed) over one 700-row subset. Every response is
appended to the job's JSONL as soon as it arrives (row index, true/pred label,
raw reply, matched flag, seconds for the call), so a crash or restart only
re-does unfinished rows. A sidecar ``*_meta.jsonl`` records each session: Ollama
version, model digest, options, start/end, rows done, loop wall-clock seconds.

Modes:
- ``zero``    : original zero-shot prompt + parser (``prompts_v4``).
- ``few``     : original 1-example/class prompt + parser; examples per seed from
                ``results/revision/setup/fewshot_reference_examples_*.json``.
- ``dynamic`` : few-shot template with per-row retrieved examples (E4) from
                ``results/revision/e4/e4_retrieved_examples_seed{seed}.json``.
- ``three``   : few-shot template with 3 examples/class (paper Table VI, E8) from
                ``results/revision/e8/e8_threeshot_reference_blocks_*.json``;
                run with num_ctx=8192 (prompt ~6.5k tokens), as in the original.

Settings identical to the paper: temperature=0, num_predict=60, num_ctx left at
Ollama's default (4096 on this machine). Before each job the other model is
unloaded and one warm-up call loads this one, so load time is excluded from
the per-row timings (same protocol as the Sep-19 controlled retime).

Edit ``JOBS`` below; no command-line arguments (repo convention).
"""

import json
import os
import sys
import time
from datetime import datetime, timezone

import httpx
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ollama_host import get_ollama_client, resolve_ollama_host  # noqa: E402
from prompts_v4 import (  # noqa: E402
    few_shot_prompt,
    label_names,
    parse_few_shot,
    parse_zero_shot,
    reference_block,
    reverse_labels,
    selected_features,
    zero_shot_prompt,
)

# ==================================================
# CONFIG
# ==================================================

SEEDS = [42, 1, 2, 3, 4]
MODELS = ["qwen2.5:7b", "qwen2.5:14B"]

# E1: 4 configs x 5 seeds, grouped by model to minimise model swaps
JOBS = [
    {"exp": "e1", "model": m, "mode": mode, "seed": s}
    for m in MODELS for mode in ["zero", "few"] for s in SEEDS
]

OPTIONS = {"temperature": 0, "num_predict": 60}
SUBSET_TMPL = "results/revision/setup/llm_subset_v4_seed{seed}.csv"
REFS_FILE = "results/revision/setup/fewshot_reference_examples_seeds42-1-2-3-4.json"
DYNAMIC_TMPL = "results/revision/e4/e4_retrieved_examples_seed{seed}.json"
THREE_FILE = "results/revision/e8/e8_threeshot_reference_blocks_seeds42-1-2-3-4.json"
NUM_CTX = {"three": 8192}  # modes not listed use Ollama's default context
MAX_ROWS = None  # set an int for a pilot run (first N rows of each job)

# ==================================================

client = get_ollama_client()
HOST = resolve_ollama_host()


def job_options(job):
    ctx = NUM_CTX.get(job["mode"])
    return {**OPTIONS, "num_ctx": ctx} if ctx else OPTIONS


def job_stem(job):
    slug = job["model"].replace(":", "-")
    ctx = NUM_CTX.get(job["mode"], "default")
    return (f"results/revision/{job['exp']}/llm/"
            f"{job['exp']}_{slug}_{job['mode']}_seed{job['seed']}_t0_np60_ctx{ctx}")


def ollama_info(model):
    version = httpx.get(f"{HOST}/api/version", timeout=10).json()["version"]
    tags = httpx.get(f"{HOST}/api/tags", timeout=10).json()["models"]
    digest = next((t["digest"] for t in tags if t["name"] == model), None)
    return version, digest


def load_model(model, options=OPTIONS):
    for other in MODELS:
        if other != model:
            client.generate(model=other, prompt="", keep_alive=0)  # unload
    client.chat(model=model, messages=[{"role": "user", "content": "Reply with OK."}],
                options=options)  # warm-up / load (at this job's num_ctx), not timed


def done_rows(path):
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    done.add(json.loads(line)["row"])
    return done


def run_job(job, refs):
    stem = job_stem(job)
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    out_path, meta_path = f"{stem}.jsonl", f"{stem}_meta.jsonl"

    df = pd.read_csv(SUBSET_TMPL.format(seed=job["seed"]))
    X = df[selected_features]
    y_true = df["Label"].values
    n_rows = len(X) if MAX_ROWS is None else min(MAX_ROWS, len(X))

    done = done_rows(out_path)
    todo = [i for i in range(n_rows) if i not in done]
    print(f"\n=== {stem}  ({len(done)} done, {len(todo)} to do)", flush=True)
    if not todo:
        return

    if job["mode"] == "few":
        ex = refs[str(job["seed"])]["examples"]
        fixed_block = reference_block({int(k): v for k, v in ex.items()})
    elif job["mode"] == "dynamic":
        with open(DYNAMIC_TMPL.format(seed=job["seed"])) as f:
            retrieved = json.load(f)["rows"]
    elif job["mode"] == "three":
        with open(THREE_FILE) as f:
            fixed_block = json.load(f)[str(job["seed"])]["reference_block"]
    opts = job_options(job)

    version, digest = ollama_info(job["model"])
    load_model(job["model"], opts)
    session = {"start_utc": datetime.now(timezone.utc).isoformat(), "ollama_version": version,
               "model": job["model"], "model_digest": digest, "options": opts,
               "num_ctx": opts.get("num_ctx", "default"), "rows_before": len(done)}
    t_loop = time.time()
    n_new = n_unmatched = 0

    with open(out_path, "a") as out:
        for i in todo:
            row = X.iloc[i]
            if job["mode"] == "zero":
                prompt, parse = zero_shot_prompt(row), parse_zero_shot
            elif job["mode"] in ("few", "three"):
                prompt, parse = few_shot_prompt(row, fixed_block), parse_few_shot
            else:
                ex = retrieved[str(i)]["examples"]
                prompt = few_shot_prompt(row, reference_block({int(k): v for k, v in ex.items()}))
                parse = parse_few_shot

            t0 = time.time()
            response = client.chat(model=job["model"],
                                   messages=[{"role": "user", "content": prompt}],
                                   options=opts)
            secs = time.time() - t0
            raw = response["message"]["content"]
            pred, matched = parse(raw)

            rec = {"row": i, "true": int(y_true[i]), "pred": reverse_labels[pred],
                   "matched": matched, "raw": raw, "secs": round(secs, 4),
                   "prompt_eval_count": response.get("prompt_eval_count"),
                   "eval_count": response.get("eval_count")}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            n_new += 1
            n_unmatched += not matched
            print(f"[{i + 1}/{n_rows}] True={label_names[int(y_true[i])]} Pred={pred}"
                  f"{'' if matched else ' (UNMATCHED->Benign)'} {secs:.2f}s", flush=True)

    session.update({"end_utc": datetime.now(timezone.utc).isoformat(),
                    "rows_this_session": n_new, "unmatched_this_session": n_unmatched,
                    "loop_wall_clock_s": round(time.time() - t_loop, 2)})
    with open(meta_path, "a") as f:
        f.write(json.dumps(session) + "\n")
    print(f"=== session done: {n_new} rows, {n_unmatched} unmatched, "
          f"{session['loop_wall_clock_s']} s", flush=True)


if __name__ == "__main__":
    with open(REFS_FILE) as f:
        refs = json.load(f)
    for job in JOBS:
        run_job(job, refs)
    print("\nAll jobs done.")
