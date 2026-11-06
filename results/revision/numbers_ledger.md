# Numbers ledger

Every number used in the paper or the response letter. Values are copied from the
source file (≤2 decimals, no further rounding). "Script" produced the source file.

## Reproduction / setup
| Claim | Value | Source file | Script |
|---|---|---|---|
| Seed-42 subset identical to original 700-row subset | byte-identical | `results/revision/setup/setup_subsets_log.txt` | `research/revision/setup_subsets.py` |
| Revision prompts identical to original | 700/700 zero + 700/700 few | `results/revision/setup/check_prompts_log.txt` | `research/revision/check_prompts_v4.py` |
| Ollama version for revision runs | 0.40.13 | `results/revision/e1/llm/*_meta.jsonl` | `research/revision/run_llm_jobs.py` |

## E3 — real vs synthetic (balanced training set, 200,000/class)
Source: `results/revision/e3/e3_real_vs_synthetic_v4_target200k_rs42.csv` (script `research/revision/e3_real_vs_synthetic.py`, input `results/split_data_v4/training_balancing_report.csv`).

| Class | Real available (train) | Real in balanced set | Synthetic | % synthetic | Undersampled |
|---|---|---|---|---|---|
| Benign | 8,863,713 | 200,000 | 0 | 0.00 | yes |
| Bot | 102,538 | 102,538 | 97,462 | 48.73 | no |
| Brute Force | 72,271 | 72,271 | 127,729 | 63.86 | no |
| DoS/DDoS | 905,997 | 200,000 | 0 | 0.00 | yes |
| Infiltration | 97,868 | 97,868 | 102,132 | 51.07 | no |
| PortScan | 63,486 | 63,486 | 136,514 | 68.26 | no |
| Web Attack | 2,103 | 2,103 | 197,897 | 98.95 | no |
| Overall | — | 738,266 | 661,734 | 47.27 | — |

## E1 — ML on 5 subsets
Source: `results/revision/ml_subsets/overall_by_seed.csv` (script `research/revision/eval_ml_on_subsets.py`). Summary rows (mean ± SD) to be added by the aggregation script.

## E1 / E2 / E4 / E5 — 5 disjoint 700-row subsets (seeds 42, 1, 2, 3, 4), mean ± SD (ddof=1)
Source: `results/revision/summary/summary_mean_sd.csv` (script `research/revision/aggregate_results.py`; inputs `results/revision/ml_subsets/predictions_*`, `results/revision/e{1,4}/llm/*.jsonl`, `results/revision/e4/e4_baseline_predictions_seed*.csv`). Per-seed values: `per_seed_metrics.csv`; per-class F1: `per_class_f1_mean_sd.csv`. LLM time = sum of per-row seconds for 700 rows.

| System | Acc mean | Acc SD | Acc min | Acc max | Macro-F1 mean | Macro-F1 SD | Time (s) mean ± SD | Unmatched (max/run) |
|---|---|---|---|---|---|---|---|---|
| XGB 70 feat (models/v4) | 92.31 | 0.6 | 91.57 | 93.0 | 92.4 | 0.58 | 0.02 ± 0.0 | 0 |
| MLP 70 feat | 89.94 | 0.41 | 89.29 | 90.29 | 90.05 | 0.39 | 0.01 ± 0.01 | 0 |
| RF 70 feat | 89.77 | 1.41 | 87.43 | 91.0 | 89.99 | 1.38 | 0.04 ± 0.0 | 0 |
| XGB 22 feat (E2) | 91.37 | 1.08 | 90.0 | 92.43 | 91.5 | 1.04 | 0.01 ± 0.01 | 0 |
| RF 22 feat (E2) | 91.0 | 1.82 | 88.43 | 92.57 | 91.12 | 1.76 | 0.04 ± 0.0 | 0 |
| MLP 22 feat (E2) | 89.06 | 0.73 | 88.14 | 90.0 | 89.15 | 0.67 | 0.01 ± 0.0 | 0 |
| XGB no-SMOTE weighted (E5) | 93.14 | 0.79 | 92.0 | 94.0 | 93.17 | 0.77 | 0.0 ± 0.0 | 0 |
| RF no-SMOTE weighted (E5) | 90.6 | 1.39 | 88.14 | 91.43 | 90.75 | 1.32 | 0.03 ± 0.0 | 0 |
| k-NN k=5, SMOTE set (E4 b, extra) | 87.83 | 0.78 | 86.57 | 88.71 | 88.15 | 0.77 | — | 0 |
| Retrieval 1-NN label (E4 a) | 75.6 | 0.66 | 74.86 | 76.43 | 72.76 | 0.99 | — | 0 |
| k-NN k=5, real rows (E4 b) | 75.4 | 0.84 | 74.14 | 76.29 | 72.24 | 1.0 | — | 0 |
| Qwen2.5-14B dynamic few-shot (E4) | 73.57 | 0.71 | 72.71 | 74.57 | 70.35 | 1.23 | 2184.64 ± 5.94 | 0 |
| Qwen2.5-14B few-shot | 61.49 | 6.66 | 54.57 | 69.29 | 57.37 | 7.0 | 580.03 ± 13.08 | 0 |
| Qwen2.5-7B few-shot | 58.71 | 5.2 | 53.43 | 65.71 | 56.05 | 5.91 | 134.65 ± 2.32 | 0 |
| Qwen2.5-14B zero-shot | 16.29 | 0.59 | 15.57 | 17.0 | 10.05 | 0.2 | 490.72 ± 1.35 | 0 |
| Qwen2.5-7B zero-shot | 8.34 | 0.91 | 7.43 | 9.43 | 4.34 | 0.26 | 124.29 ± 1.22 | 3 (4 total) |

Seed-42 drift check (Ollama 0.40.13 vs original): 14B few 65.71 (original 65.71); 7B few 62.29 (original 61.57). Source `per_seed_metrics.csv`.

E4 (c) LLM answer = class of nearest retrieved example: 94.57 / 95.86 / 95.57 / 95.29 / 95.14 % (seeds 42/1/2/3/4). Source `results/revision/summary/e4_llm_vs_nearest_agreement.csv`.

## McNemar (exact), per seed and pooled over 3,500 rows
Source: `results/revision/summary/mcnemar.csv` (`aggregate_results.py`). b = a-only correct, c = b-only correct.

| a vs b | 42 | 1 | 2 | 3 | 4 | pooled |
|---|---|---|---|---|---|---|
| XGB vs RF (70 feat) | 33/23 p=0.23 | 46/13 p=1.92e-05 | 38/20 p=0.02 | 31/17 p=0.06 | 32/18 p=0.06 | 180/91 p=6.94e-08 |
| 14B few vs 7B few | 41/17 p=2.23e-03 | 109/40 p=1.41e-08 | 28/49 p=0.02 | 42/42 p=1.0 | 57/32 p=0.01 | 277/180 p=6.59e-06 |
| XGB vs 14B dynamic (best ML vs best LLM) | 150/28 | 155/25 | 162/24 | 162/30 | 155/21 | 784/128 p=1.17e-115 |
| 14B dynamic vs 14B few | 152/90 p=8.11e-05 | 151/79 p=2.38e-06 | 164/34 p=1.28e-21 | 147/20 p=4.42e-25 | 108/76 p=0.02 | 722/299 p=5.55e-41 |
| 14B dynamic vs retrieval 1-NN | 12/25 p=0.05 | 8/20 p=0.04 | 5/26 p=1.92e-04 | 7/22 p=8.13e-03 | 11/21 p=0.11 | 43/114 p=1.32e-08 |

## Held-out test (4,331,990 rows, natural distribution)
Source: `results/revision/heldout/heldout_overall_<set>.csv` (script `research/revision/eval_ml_heldout.py`). Values ×100.

| Model set | Model | Accuracy | Weighted F1 | Macro F1 |
|---|---|---|---|---|
| v4 70 feat (= paper Table III) | RF | 86.84 | 91.99 | 77.47 |
| | XGB | 83.32 | 89.89 | 74.22 |
| | MLP | 72.34 | 82.5 | 71.12 |
| E2 22 feat | RF | 84.19 | 90.42 | 75.51 |
| | XGB | 79.12 | 87.25 | 74.69 |
| | MLP | 69.82 | 80.5 | 67.93 |
| E5 no SMOTE, weighted | RF | 85.07 | 90.94 | 78.97 |
| | XGB | 79.09 | 87.23 | 76.19 |

Training time (s): E2 RF 29.99, XGB 94.49, MLP 558.03 (`results/revision/e2/e2_train_times.csv`); E5 RF 20.6, XGB 45.11 (`results/revision/e5/e5_train_times.csv`).

## E7 — source-separated test (train CSE-CIC-IDS2018, test CIC-IDS2017)
Source: `results/revision/e7/e7_overall_noProtocol_rs42.csv`, `e7_per_class_noProtocol_rs42.csv`, `e7_deduped_counts_by_source.csv`, `e7_training_composition.csv`, `e7_source_boundary.json` (script `research/revision/e7_source_separated.py`). 5 shared classes (Benign, Bot, Brute Force, DoS/DDoS, Web Attack; Infiltration has 36 rows in 2017, PortScan 0 in 2018), Protocol dropped (69 features). Values ×100.

| Setting | Model | Accuracy | Weighted F1 | Macro F1 | n test |
|---|---|---|---|---|---|
| Train 2018 → test 2017 | RF | 86.12 | 80.7 | 25.34 | 2,430,057 |
| | XGB | 80.66 | 78.74 | 32.08 | 2,430,057 |
| | MLP | 81.8 | 82.48 | 24.45 | 2,430,057 |
| Control 2018 70:30 | RF | 99.89 | 99.91 | 83.6 | 3,533,822 |
| | XGB | 99.93 | 99.96 | 83.61 | 3,533,822 |

Per-class recall, train 2018 → test 2017 (RF / XGB / MLP): Benign 99.3 / 92.24 / 93.5; Bot 11.86 / 1.75 / 0.0; Brute Force 5.43 / 55.78 / 34.15; DoS/DDoS 3.61 / 6.97 / 7.98; Web Attack 0.23 / 1.68 / 1.68. Control per-class F1 ≥ 99.7 for all classes except Web Attack (18.63 RF / 18.23 XGB; 258 test rows).
Deduped rows by source: 2017 = 2,520,787, 2018 = 11,919,179 (boundary replay; per-class counts in `e7_deduped_counts_by_source.csv`).

## E6 — 5-fold stratified CV (R1#2), full deduped dataset (14,439,966 rows, 70 features, 7 classes)
Source: `results/revision/e6/e6_cv5_overall_rs42.csv`, `e6_cv5_summary_rs42.csv`, `e6_cv5_per_class_rs42.csv`, `e6_cv5_fold_composition_rs42.csv` (script `research/revision/e6_cv5_ml.py`). StratifiedKFold(5, shuffle, rs=42); v4 undersample + SMOTE to 200k/class inside each training fold only; test folds natural distribution (~2,887,993 rows each). Mean ± SD (ddof=1) over 5 folds, ×100.

| Model | Accuracy | Weighted F1 | Macro F1 (min–max) | Fit time per fold (s) |
|---|---|---|---|---|
| RF | 85.95 ± 0.06 | 91.46 ± 0.03 | 77.45 ± 0.10 (77.38–77.63) | 65.2–68.3 |
| XGB | 82.33 ± 0.4 | 89.27 ± 0.25 | 72.91 ± 0.96 (72.23–74.56) | 76.1–83.6 |
| MLP | 70.99 ± 1.59 | 81.39 ± 1.18 | 68.69 ± 0.88 (67.36–69.6) | 214.8–437.3 |

Compare single 70:30 split (paper Table III): RF macro F1 77.47, XGB 74.22, MLP 71.12.
Per-class F1 mean over folds (RF / XGB / MLP): Benign 91.32 / 88.82 / 80.23; Bot 99.1 / 99.28 / 93.21; Brute Force 99.8 / 99.83 / 98.07; DoS/DDoS 99.72 / 99.9 / 98.08; Infiltration 9.48 / 8.39 / 5.55; PortScan 99.36 / 99.36 / 86.65; Web Attack 43.41 / 14.78 / 19.03.

## Original (paper) values used for comparison
| Claim | Value | Source |
|---|---|---|
| 14B few-shot, original subset | acc 65.71, macro F1 64.11, 3167.07 s | `results/llm_results_v4/qwen2.5_14B_few_shot/llm_few_results.csv` |
| 7B few-shot | acc 61.57, macro F1 60.11, 144.89 s | `results/llm_results_v4/qwen2.5_7b_few_shot/llm_few_results.csv` |
| 14B zero-shot | acc 16.14, macro F1 9.64, 659.58 s | `results/llm_results_v4/qwen2.5_14B_zero_shot/llm_zero_results.csv` |
| 7B zero-shot | acc 8.29, macro F1 4.12, 141.81 s | `results/llm_results_v4/qwen2.5_7b_zero_shot/llm_zero_results.csv` |
| ML on original subset | RF 90.57 / XGB 92.0 / MLP 90.29 acc | `results/llm_evaluation_v4/model_results_v4.csv` |
| 3-shot, num_ctx 8192 (Table VI), seeds 101/202/303, original subset | acc 68.00 / 73.86 / 70.43; macro F1 66.03 / 71.71 / 67.62 | `results/audit_followup/step2_summary.txt` |

## Per-class F1, 5 subsets (paper Tables IV/V), mean (SD)
Source: `results/revision/summary/per_class_f1_mean_sd.csv` (`aggregate_results.py`).

| Class | RF | XGB | MLP | 7B zero | 7B few | 14B zero | 14B few | 14B dynamic |
|---|---|---|---|---|---|---|---|---|
| Benign | 69.81 (3.7) | 73.81 (1.8) | 64.96 (1.63) | 22.18 (2.37) | 31.2 (8.16) | 26.07 (1.45) | 29.67 (7.61) | 51.98 (0.54) |
| Bot | 99.7 (0.45) | 99.8 (0.45) | 99.8 (0.45) | 0.0 (0.0) | 86.11 (3.39) | 0.0 (0.0) | 92.04 (9.87) | 99.5 (0.62) |
| Brute Force | 100.0 (0.0) | 100.0 (0.0) | 99.7 (0.27) | 0.0 (0.0) | 95.32 (1.01) | 3.08 (1.67) | 83.38 (6.08) | 98.43 (0.64) |
| DoS/DDoS | 99.1 (0.42) | 100.0 (0.0) | 97.06 (1.1) | 0.0 (0.0) | 39.04 (9.23) | 21.21 (6.06) | 48.2 (14.69) | 96.73 (1.59) |
| Infiltration | 69.27 (5.55) | 75.41 (1.96) | 74.62 (1.07) | 1.17 (1.74) | 28.84 (8.93) | 2.37 (2.27) | 22.77 (10.78) | 14.07 (5.51) |
| PortScan | 99.9 (0.22) | 100.0 (0.0) | 99.8 (0.45) | 7.01 (1.62) | 76.39 (4.83) | 13.77 (2.88) | 75.9 (5.21) | 96.4 (1.37) |
| Web Attack | 92.12 (0.78) | 97.75 (0.47) | 94.4 (1.72) | 0.0 (0.0) | 35.48 (35.79) | 3.87 (2.82) | 49.61 (33.06) | 35.38 (4.38) |

## XGBoost confusion matrix pooled over 5 subsets (paper Fig. 3)
Source: `results/revision/summary/xgb_confusion_pooled_5seeds.csv` (`research/revision/fig3_confusion_pooled.py`). Rows = actual, 500 per class. Off-diagonal: Benign→Infiltration 125, Infiltration→Benign 120, Web Attack→Benign 20, Web Attack→Infiltration 2, Bot→Benign 1, Bot→Infiltration 1 (269 errors of 3,500).

## E8 — 3 examples/class (former Table VI) on the 5 subsets, Qwen2.5-14B, num_ctx 8192
Source: `results/revision/summary/summary_mean_sd.csv`, `per_seed_metrics.csv`, `mcnemar.csv` (`aggregate_results.py`; LLM JSONLs `results/revision/e8/llm/`, runner `run_e8_jobs.py`, prompts `e8_threeshot_examples.py`).
| Claim | Value |
|---|---|
| Accuracy mean ± SD (min–max) | 69.37 ± 0.99 (68.43–70.86) |
| Macro F1 mean ± SD | 66.55 ± 1.32 |
| Time per 700 rows (s) | 767.04 ± 13.37 |
| Per seed acc (42/1/2/3/4) | 68.43 / 70.86 / 69.0 / 69.86 / 68.71 |
| Unmatched | 0 in all 5 runs |
| Seed-42 vs original Table VI run 1 (EXTRA_SEED 101, Ollama 0.34.2 era) | 68.43 vs 68.00; 45/700 predictions differ |
| McNemar 3-shot vs 1-shot, pooled | 378/102, p = 2.61e-38 (per seed p < 0.01 on 4 of 5; seed 4: 45/49, p = 0.76) |
| McNemar dynamic vs 3-shot, pooled | 642/495, p = 1.45e-05 |

## McNemar XGB (70 feat) vs every LLM configuration, pooled over 3,500 rows
Source: `results/revision/summary/mcnemar.csv`. Largest per-seed p across all LLM configs: 2.29e-21 (vs 14B dynamic, seed 42).
| LLM | XGB-only / LLM-only correct | p |
|---|---|---|
| 14B dynamic | 784 / 128 | 1.17e-115 |
| 14B three | 885 / 82 | 6.92e-171 |
| 14B few | 1130 / 51 | 6.65e-266 |
| 7B few | 1246 / 70 | 4.25e-279 |
| 14B zero | 2772 / 111 | < 1e-300 |
| 7B zero | 3002 / 63 | < 1e-300 |

Unmatched LLM responses, all revision runs: 4 of 21,000 (30 runs × 700), all 7B zero-shot. Source `per_seed_metrics.csv`.

## Derived values used in the text
| Claim | Value | Derivation / source |
|---|---|---|
| 3-shot prompt length "about 6,600 tokens" | prompt_eval_count min 6,384, max 6,809, mean 6,568 | `results/revision/e8/llm/*.jsonl` |
| Dynamic few-shot cost "about 3.8 times" the fixed prompt | 2184.64 / 580.03 = 3.77 | Table II times above |
| "14.4 million flows" (letter) | 14,439,966 | `results/dedup_v4/dedup_report.csv` |
| "over 3,000 times longer" | 124.29 s / 0.0378 s (RF mean median-predict) = 3,288 | `summary_mean_sd.csv`, `ml_subsets/overall_by_seed.csv` |
| "about 17 points below XGBoost" (1-NN label vs XGBoost, Sec. IV-D) | 92.31 − 75.60 = 16.71 | `summary_mean_sd.csv` |
