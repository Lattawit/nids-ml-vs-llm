"""E8: 3-examples-per-class prompt (paper Table VI) on the 5 revision subsets.

Table VI's 3-shot rows (``results/audit_followup/step2_*``, num_ctx=8192) were
run on the seed-42 subset only. This prepares the same prompt for every subset
seed so the 3-shot result can be reported as mean +- SD like Table II.

Per seed and class, the reference examples are the seed's 1-shot example
(``fewshot_reference_examples_*.json``) plus 2 extra real pre-SMOTE training
rows drawn exactly as ``llm_classification_few_shot_v4_3shot_ablation_ctx8192.py``
draws them: ``RandomState(EXTRA_SEED).choice(class positions minus base row,
2, replace=False)``. ``EXTRA_SEED`` is 101 for seed 42 (= the original run 1,
so that prompt is reproduced exactly) and 1000 + seed for seeds 1-4.

Checks (assert):
- the knowledge text and the per-row prompt template of the original 3-shot
  script equal ``prompts_v4`` (prompts rebuilt by executing the original
  script's loop code on all 700 seed-42 rows);
- every extra example rendered from the cached float64 matrix equals the
  original rendering path (chunked ``usecols`` read of ``X_train.csv``).

Output: ``results/revision/e8/e8_threeshot_reference_blocks_seeds42-1-2-3-4.json``
(per seed: extra_seed, train positions per class, the reference block text).
The LLM runs use ``run_llm_jobs.py`` mode ``three`` (num_ctx=8192).
"""

import ast
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
import prompts_v4 as P  # noqa: E402

# ==================================================
# CONFIG
# ==================================================

# original 3-shot script: under results/audit_followup/ in the development repo,
# under research/ in the release
ORIG = next(p for p in ["results/audit_followup/llm_classification_few_shot_v4_3shot_ablation_ctx8192.py",
                        "research/llm_classification_few_shot_v4_3shot_ablation_ctx8192.py"]
            if os.path.exists(p))
REFS = "results/revision/setup/fewshot_reference_examples_seeds42-1-2-3-4.json"
CACHE = "results/revision/cache"
TRAIN_X = "results/split_data_v4/X_train.csv"
SUBSET42 = "results/revision/setup/llm_subset_v4_seed42.csv"
OUT_DIR = "results/revision/e8"
SEEDS = [42, 1, 2, 3, 4]
EXTRA_SEED = {42: 101, 1: 1001, 2: 1002, 3: 1003, 4: 1004}
EXAMPLES_PER_CLASS = 3
CHUNK_SIZE = 1_000_000
os.makedirs(OUT_DIR, exist_ok=True)

with open(REFS) as f:
    refs = json.load(f)
y_train = np.load(f"{CACHE}/y_train_real.npy")
X_real = np.load(f"{CACHE}/X_train_real_22feat_float64.npy", mmap_mode="r")


def block3(examples):
    """Reference block exactly as the original 3-shot script builds it."""
    block = ""
    for label in sorted(examples):
        for idx, example in enumerate(examples[label], start=1):
            block += f"\nReference Example {idx} ({P.label_names[label]})\n"
            block += example
            block += "\n"
    return block


out, all_extra = {}, {}
for seed in SEEDS:
    base = {int(k): int(v) for k, v in refs[str(seed)]["train_row_position"].items()}
    base_txt = {int(k): v for k, v in refs[str(seed)]["examples"].items()}
    positions, examples = {}, {}
    for label in sorted(np.unique(y_train)):
        label = int(label)
        pos = np.flatnonzero(y_train == label)
        remaining = pos[pos != base[label]]
        extra = np.random.RandomState(EXTRA_SEED[seed]).choice(
            remaining, size=EXAMPLES_PER_CLASS - 1, replace=False)
        positions[label] = [base[label]] + [int(p) for p in extra]
        examples[label] = [base_txt[label]] + [
            P.render_example(pd.Series(X_real[p], index=P.selected_features)) for p in extra]
        for p, txt in zip(extra, examples[label][1:]):
            all_extra[int(p)] = txt
    out[str(seed)] = {"extra_seed": EXTRA_SEED[seed], "train_row_position": positions,
                      "reference_block": block3(examples)}

# ---- check: extras rendered from the matrix == original rendering path ------
row0, checked = 0, 0
for chunk in pd.read_csv(TRAIN_X, usecols=P.selected_features, chunksize=CHUNK_SIZE,
                         low_memory=False):
    chunk.columns = chunk.columns.str.strip()
    for p, txt in all_extra.items():
        if row0 <= p < row0 + len(chunk):
            assert P.render_example(chunk.iloc[p - row0]) == txt, f"render differs at {p}"
            checked += 1
    row0 += len(chunk)
assert checked == len(all_extra)
print(f"{checked} extra examples: matrix rendering == original rendering path")

# ---- check: original 3-shot script's knowledge + per-row prompt ------------
tree = ast.parse(open(ORIG).read())
know = next(n for n in tree.body if isinstance(n, ast.Assign)
            and getattr(n.targets[0], "id", None) == "knowledge")
assert ast.literal_eval(know.value) == P.knowledge, "knowledge differs"
loop = next(n for n in tree.body if isinstance(n, ast.For) and "iterrows" in ast.unparse(n.iter))
stmts = []
for s in loop.body:
    stmts.append(s)
    if isinstance(s, ast.Assign) and getattr(s.targets[0], "id", None) == "prompt":
        break
code = compile(ast.Module(body=stmts, type_ignores=[]), "<orig>", "exec")
blk = out["42"]["reference_block"]
X42 = pd.read_csv(SUBSET42)[P.selected_features]
for i, row in X42.iterrows():
    ns = {"row": row, "selected_features": P.selected_features,
          "knowledge": P.knowledge, "reference_block": blk}
    exec(code, ns)
    assert ns["prompt"] == P.few_shot_prompt(row, blk), f"3-shot prompt differs at row {i}"
print(f"OK: {len(X42)} seed-42 3-shot prompts byte-identical to the original script's template")

path = f"{OUT_DIR}/e8_threeshot_reference_blocks_seeds42-1-2-3-4.json"
with open(path, "w") as f:
    json.dump(out, f, indent=1)
print("Saved", path)
