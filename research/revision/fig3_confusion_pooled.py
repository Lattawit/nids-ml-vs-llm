"""Fig. 3 for the revised paper: XGBoost confusion matrix pooled over the 5 subsets.

Sums the per-seed confusion matrices of models/v4 XGBoost (70 features) on the
5 disjoint 700-row subsets (3,500 rows, 500 per class). Same style as the
original Fig. 3 (seaborn heatmap, Blues). Outputs:
- ``results/revision/summary/xgb_confusion_pooled_5seeds.csv`` (counts)
- ``paper/figures/fig3_xgb_confusion_pooled.pdf`` (vector, for LaTeX)
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import seaborn as sns  # noqa: E402
from sklearn.metrics import confusion_matrix  # noqa: E402

SEEDS = [42, 1, 2, 3, 4]
CLASSES = ["Benign", "Bot", "Brute Force", "DoS/DDoS", "Infiltration", "PortScan", "Web Attack"]

cm = np.zeros((7, 7), dtype=int)
for s in SEEDS:
    df = pd.read_csv(f"results/revision/ml_subsets/predictions_v4_70feat_seed{s}.csv")
    cm += confusion_matrix(df["true"], df["xgboost"], labels=range(7))
assert cm.sum() == 3500
pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_csv(
    "results/revision/summary/xgb_confusion_pooled_5seeds.csv")
print(pd.DataFrame(cm, index=CLASSES, columns=CLASSES).to_string())

# TrueType (Type 42) fonts: IEEE PDF eXpress rejects matplotlib's default Type 3 fonts
plt.rcParams.update({"font.size": 11, "pdf.fonttype": 42, "ps.fonttype": 42})
fig, ax = plt.subplots(figsize=(6.4, 5.2))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASSES, yticklabels=CLASSES,
            vmin=0, vmax=500, ax=ax, cbar_kws={"shrink": 0.85})
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)
plt.tight_layout()
fig.savefig("paper/figures/fig3_xgb_confusion_pooled.pdf", bbox_inches="tight")
print("Saved paper/figures/fig3_xgb_confusion_pooled.pdf")
