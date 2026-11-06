"""Aggregate revision results across seeds (E1, E2, E4, E5 on the 5 subsets).

Inputs:
- ML per-row predictions: ``results/revision/ml_subsets/predictions_<set>_seed<s>.csv``
- LLM JSONLs: ``results/revision/{e1,e4}/llm/*.jsonl`` (+ ``*_meta.jsonl``)
- E4 baselines: ``results/revision/e4/e4_baseline_predictions_seed<s>.csv``

Outputs (``results/revision/summary/``):
- ``per_seed_metrics.csv``   accuracy / macro-F1 / time / unmatched per system x seed
- ``summary_mean_sd.csv``    mean, SD (ddof=1), min, max over seeds
- ``per_class_f1_mean_sd.csv``
- ``mcnemar.csv``            exact McNemar per seed + pooled over all seeds' rows
Only systems with all seeds complete are summarised; incomplete ones are listed.
"""

import glob
import json
import os
import re

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from statsmodels.stats.contingency_tables import mcnemar

SEEDS = [42, 1, 2, 3, 4]
N_ROWS = 700
OUT_DIR = "results/revision/summary"
CLASSES = ["Benign", "Bot", "Brute Force", "DoS/DDoS", "Infiltration", "PortScan", "Web Attack"]
ML_NAMES = {"random_forest": "RF", "xgboost": "XGB", "mlp": "MLP"}
os.makedirs(OUT_DIR, exist_ok=True)

# system -> seed -> DataFrame(row, true, pred), plus time / unmatched
preds, times, unmatched = {}, {}, {}

# ---- ML -------------------------------------------------------------------
ml_time = pd.read_csv("results/revision/ml_subsets/overall_by_seed.csv")
for f in glob.glob("results/revision/ml_subsets/predictions_*_seed*.csv"):
    set_name, seed = re.match(r".*predictions_(.+)_seed(\d+)\.csv", f).groups()
    seed = int(seed)
    df = pd.read_csv(f)
    for m in [c for c in df.columns if c in ML_NAMES]:
        sysname = f"{ML_NAMES[m]} [{set_name}]"
        preds.setdefault(sysname, {})[seed] = df[["row", "true"]].assign(pred=df[m])
        t = ml_time.query("model_set == @set_name and model == @m and seed == @seed")
        times.setdefault(sysname, {})[seed] = float(t["median_predict_s"].iloc[0])
        unmatched.setdefault(sysname, {})[seed] = 0

# ---- LLM ------------------------------------------------------------------
for f in glob.glob("results/revision/e*/llm/*.jsonl"):
    if f.endswith("_meta.jsonl"):
        continue
    exp, model, mode, seed = re.match(r".*/(e\d)_(.+?)_(zero|few|dynamic|three)_seed(\d+)_", f).groups()
    seed = int(seed)
    recs = [json.loads(line) for line in open(f) if line.strip()]
    if len({r["row"] for r in recs}) < N_ROWS:
        print(f"incomplete ({len(recs)}/{N_ROWS}): {f}")
        continue
    sysname = f"{model} {mode}"
    df = pd.DataFrame(recs).drop_duplicates("row", keep="last").sort_values("row")
    preds.setdefault(sysname, {})[seed] = df[["row", "true", "pred"]]
    times.setdefault(sysname, {})[seed] = float(df["secs"].sum())
    unmatched.setdefault(sysname, {})[seed] = int((~df["matched"]).sum())

# ---- E4 retrieval baselines ---------------------------------------------------
for f in glob.glob("results/revision/e4/e4_baseline_predictions_seed*.csv"):
    seed = int(re.search(r"seed(\d+)", f).group(1))
    df = pd.read_csv(f)
    for b in [c for c in df.columns if c not in ("row", "true")]:
        sysname = f"baseline {b}"
        preds.setdefault(sysname, {})[seed] = df[["row", "true"]].assign(pred=df[b])
        unmatched.setdefault(sysname, {})[seed] = 0

# ---- per-seed metrics ---------------------------------------------------------
rows, cls_rows = [], []
for sysname, by_seed in preds.items():
    for seed, df in by_seed.items():
        y, p = df["true"].to_numpy(), df["pred"].to_numpy()
        rows.append({"system": sysname, "seed": seed, "accuracy": 100 * accuracy_score(y, p),
                     "macro_f1": 100 * f1_score(y, p, average="macro", zero_division=0),
                     "time_s": times.get(sysname, {}).get(seed, np.nan),
                     "unmatched": unmatched[sysname][seed]})
        f1s = f1_score(y, p, average=None, labels=range(7), zero_division=0)
        for c, v in zip(CLASSES, f1s):
            cls_rows.append({"system": sysname, "seed": seed, "class": c, "f1": 100 * v})
per_seed = pd.DataFrame(rows).sort_values(["system", "seed"])
per_seed.to_csv(f"{OUT_DIR}/per_seed_metrics.csv", index=False)

complete = [s for s, d in preds.items() if set(d) == set(SEEDS)]
for s in sorted(set(preds) - set(complete)):
    print(f"not all seeds yet: {s} {sorted(preds[s])}")

agg = (per_seed[per_seed.system.isin(complete)]
       .groupby("system")[["accuracy", "macro_f1", "time_s", "unmatched"]]
       .agg(["mean", "std", "min", "max"]))
agg.columns = [f"{a}_{b}" for a, b in agg.columns]
agg = agg.sort_values("accuracy_mean", ascending=False)
agg.to_csv(f"{OUT_DIR}/summary_mean_sd.csv")

cls = (pd.DataFrame(cls_rows).query("system in @complete")
       .groupby(["system", "class"])["f1"].agg(["mean", "std", "min", "max"]).reset_index())
cls.to_csv(f"{OUT_DIR}/per_class_f1_mean_sd.csv", index=False)

# ---- McNemar ------------------------------------------------------------------


def mcn(a, b):
    """Exact McNemar on two systems' correctness over the same rows."""
    ca, cb = a["pred"].to_numpy() == a["true"].to_numpy(), b["pred"].to_numpy() == b["true"].to_numpy()
    assert (a["row"].to_numpy() == b["row"].to_numpy()).all()
    table = [[int((ca & cb).sum()), int((ca & ~cb).sum())],
             [int((~ca & cb).sum()), int((~ca & ~cb).sum())]]
    return table[0][1], table[1][0], mcnemar(table, exact=True).pvalue


ml_sys = [s for s in complete if s.endswith("[v4_70feat]")]
llm_sys = [s for s in complete if s.startswith("qwen")]
pairs = [("XGB [v4_70feat]", "RF [v4_70feat]"), ("qwen2.5-14B few", "qwen2.5-7b few")]
if ml_sys and llm_sys:
    best_ml = max(ml_sys, key=lambda s: agg.loc[s, "accuracy_mean"])
    best_llm = max(llm_sys, key=lambda s: agg.loc[s, "accuracy_mean"])
    pairs.append((best_ml, best_llm))
if "qwen2.5-14B dynamic" in complete:
    pairs += [("qwen2.5-14B dynamic", "qwen2.5-14B few"),
              ("qwen2.5-14B dynamic", "baseline retrieval_1nn")]
if "qwen2.5-14B three" in complete:
    pairs += [("qwen2.5-14B three", "qwen2.5-14B few"),
              ("qwen2.5-14B dynamic", "qwen2.5-14B three")]
# paper claim: XGBoost vs every LLM configuration
pairs += [("XGB [v4_70feat]", s) for s in sorted(llm_sys) if ("XGB [v4_70feat]", s) not in pairs]

mc_rows = []
for a, b in pairs:
    if a not in complete or b not in complete:
        continue
    for seed in SEEDS:
        n_ab, n_ba, p = mcn(preds[a][seed], preds[b][seed])
        mc_rows.append({"a": a, "b": b, "seed": seed, "a_only_correct": n_ab,
                        "b_only_correct": n_ba, "p_value": p})
    pa = pd.concat([preds[a][s].assign(row=preds[a][s]["row"] + 1000 * i) for i, s in enumerate(SEEDS)])
    pb = pd.concat([preds[b][s].assign(row=preds[b][s]["row"] + 1000 * i) for i, s in enumerate(SEEDS)])
    n_ab, n_ba, p = mcn(pa, pb)
    mc_rows.append({"a": a, "b": b, "seed": "pooled", "a_only_correct": n_ab,
                    "b_only_correct": n_ba, "p_value": p})
mc = pd.DataFrame(mc_rows)
mc.to_csv(f"{OUT_DIR}/mcnemar.csv", index=False)

# ---- E4: how often the LLM agrees with the nearest retrieved example ---------
agree_rows = []
if "qwen2.5-14B dynamic" in preds:
    for seed, df in preds["qwen2.5-14B dynamic"].items():
        nn = pd.read_csv(f"results/revision/e4/e4_baseline_predictions_seed{seed}.csv")
        m = df.merge(nn[["row", "retrieval_1nn"]], on="row")
        agree_rows.append({"seed": seed, "n": len(m),
                           "llm_equals_nearest_pct": 100 * (m["pred"] == m["retrieval_1nn"]).mean()})
    pd.DataFrame(agree_rows).to_csv(f"{OUT_DIR}/e4_llm_vs_nearest_agreement.csv", index=False)

pd.set_option("display.width", 200)
print(agg[["accuracy_mean", "accuracy_std", "accuracy_min", "accuracy_max",
           "macro_f1_mean", "macro_f1_std", "time_s_mean", "time_s_std", "unmatched_mean"]].round(2))
if len(mc):
    print(mc.to_string(index=False))
if agree_rows:
    print(pd.DataFrame(agree_rows).to_string(index=False))
