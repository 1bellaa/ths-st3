"""
plot_model_summary.py
=====================
One figure per input_type, subplots per drug.
Each subplot shows test AUC for RF and LR side by side.
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# ── Snakemake bindings ────────────────────────────────────────────────────────
metrics_files = snakemake.input.metrics_files   # list of all *_metrics.csv
out_plot      = snakemake.output.summary_plot
input_type    = snakemake.params.input_type
drugs         = snakemake.params.drugs
log_file      = snakemake.log[0]

Path(log_file).parent.mkdir(parents=True, exist_ok=True)
log = open(log_file, "w")

def msg(m):
    print(m, flush=True)
    log.write(m + "\n")

msg(f"📊 Model summary plot — input_type: {input_type}")

# ── Load all metrics for this input_type ─────────────────────────────────────
dfs = []
for f in metrics_files:
    try:
        dfs.append(pd.read_csv(f))
    except Exception as e:
        msg(f"   ⚠️  Could not read {f}: {e}")

if not dfs:
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "No metrics data", ha="center", va="center")
    plt.savefig(out_plot)
    plt.close()
    log.close()
    sys.exit(0)

all_metrics = pd.concat(dfs, ignore_index=True)

# Keep only this input_type, test split
df = all_metrics[
    (all_metrics["input_type"] == input_type) &
    (all_metrics["split"] == "test")
].copy()

msg(f"   Rows after filter: {len(df)}")

# ── Layout ────────────────────────────────────────────────────────────────────
n_drugs = len(drugs)
ncols   = min(4, n_drugs)
nrows   = int(np.ceil(n_drugs / ncols))

fig, axes = plt.subplots(nrows, ncols,
                         figsize=(4.5 * ncols, 4 * nrows),
                         sharey=True)
axes = np.array(axes).flatten()

RF_COLOR = "#27ae60"
LR_COLOR = "#2980b9"
METRICS  = ["auc_roc", "f1", "recall", "specificity"]
x        = np.arange(len(METRICS))
width    = 0.35

for i, drug in enumerate(drugs):
    ax = axes[i]
    drug_df = df[df["drug"] == drug]

    rf_row = drug_df[drug_df["model"] == "rf"]
    lr_row = drug_df[drug_df["model"] == "lr"]

    rf_vals = [rf_row[m].values[0] if not rf_row.empty and m in rf_row else 0
               for m in METRICS]
    lr_vals = [lr_row[m].values[0] if not lr_row.empty and m in lr_row else 0
               for m in METRICS]

    bars_rf = ax.bar(x - width/2, rf_vals, width,
                     label="RF",  color=RF_COLOR, alpha=0.85)
    bars_lr = ax.bar(x + width/2, lr_vals, width,
                     label="LR",  color=LR_COLOR, alpha=0.85)

    # Value labels on bars
    for bar in list(bars_rf) + list(bars_lr):
        h = bar.get_height()
        if h > 0.01:
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.01,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=7)

    ax.set_title(drug.capitalize(), fontweight="bold", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(["AUC", "F1", "Recall", "Specificity"], fontsize=9)
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("Score", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

# Hide unused subplots
for j in range(n_drugs, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"Model Performance — {input_type} — Test Set\n"
    f"AUC / F1 / Recall / Specificity",
    fontweight="bold", fontsize=13, y=1.01
)
plt.tight_layout()
Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_plot, dpi=180, bbox_inches="tight")
plt.close()

msg(f"✅ Summary plot → {out_plot}")
log.close()