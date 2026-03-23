"""
plot_roc_by_input_type.py
=========================
One figure per drug. Subplots: one per input_type.
Each subplot: RF test, LR test ROC curves on same axes.
Reads *_roc_data.csv files produced by run_ml.py.

Run standalone:
    python workflow/scripts/plot_roc_by_input_type.py \
        --ml_dir results/ml \
        --drug isoniazid \
        --out results/plots
"""

import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
import pandas as pd

# ── Mode: Snakemake or CLI ────────────────────────────────────────────────────
try:
    roc_files   = snakemake.input.roc_files
    out_plot    = snakemake.output.plot
    drug        = snakemake.params.drug
    input_types = snakemake.params.input_types
    log_file    = snakemake.log[0]
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_file, "w")
    def msg(m): print(m); log.write(m+"\n"); log.flush()
except NameError:
    p = argparse.ArgumentParser()
    p.add_argument("--ml_dir",      required=True)
    p.add_argument("--drug",        required=True)
    p.add_argument("--input_types", nargs="+",
                   default=["snp","pan","snp_pan","card","snp_card","pan_card","snp_pan_card"])
    p.add_argument("--models",      nargs="+", default=["rf","lr"])
    p.add_argument("--out",         default="results/plots")
    args = p.parse_args()
    ml_dir      = Path(args.ml_dir)
    drug        = args.drug
    input_types = args.input_types
    out_plot    = str(Path(args.out) / f"{drug}_roc_by_input_type.png")
    roc_files   = [str(ml_dir / f"{it}_{drug}_{m}_roc_data.csv")
                   for it in input_types for m in args.models]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    def msg(m): print(m, flush=True)

msg(f"📊 ROC by input type — {drug.upper()}")

INPUT_LABELS = {
    "snp":          "SNP only",
    "pan":          "Pangenome only",
    "snp_pan":      "SNP + Pan",
    "card":         "CARD only",
    "snp_card":     "SNP + CARD",
    "pan_card":     "Pan + CARD",
    "snp_pan_card": "SNP + Pan + CARD",
}
RF_COLOR = "#27ae60"
LR_COLOR = "#2980b9"

# ── Load all ROC data ─────────────────────────────────────────────────────────
dfs = []
for f in roc_files:
    try:
        df = pd.read_csv(f)
        if not df.empty and "drug" in df.columns:
            dfs.append(df)
    except Exception:
        pass

if not dfs:
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.text(0.5, 0.5, f"No ROC data for {drug}", ha="center", va="center",
            transform=ax.transAxes, color="grey")
    ax.axis("off")
    plt.savefig(out_plot, dpi=120, bbox_inches="tight")
    plt.close()
    msg("⚠️  No ROC data — empty plot saved")
    sys.exit(0)

all_roc  = pd.concat(dfs, ignore_index=True)
drug_roc = all_roc[all_roc["drug"] == drug]

# Only keep input_types that have data
present_types = [it for it in input_types
                 if not drug_roc[drug_roc["input_type"] == it].empty]
msg(f"   Input types with data: {present_types}")

if not present_types:
    fig, ax = plt.subplots()
    ax.text(0.5, 0.5, "No data", ha="center", va="center")
    ax.axis("off")
    plt.savefig(out_plot, dpi=120)
    plt.close()
    sys.exit(0)

# ── Layout ────────────────────────────────────────────────────────────────────
n     = len(present_types)
ncols = min(4, n)
nrows = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(5.5 * ncols, 4.5 * nrows),
                          sharex=True, sharey=True)
axes = np.array(axes).flatten()

for i, it in enumerate(present_types):
    ax      = axes[i]
    it_roc  = drug_roc[drug_roc["input_type"] == it]
    it_label = INPUT_LABELS.get(it, it)

    if it_roc.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, color="grey", fontsize=10)
        ax.set_title(it_label, fontweight="bold", fontsize=10)
        ax.plot([0,1],[0,1],"k:",lw=1,alpha=0.4)
        ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])
        continue

    for model, color in [("rf", RF_COLOR), ("lr", LR_COLOR)]:
        m_roc = it_roc[it_roc["model"] == model]
        for split, style in [("test", "-")]:
            sub = m_roc[m_roc["split"] == split]
            if sub.empty:
                continue
            auc   = sub["auc"].iloc[0]
            label = f"{'RF' if model=='rf' else 'LR'} {split.capitalize()} (AUC={auc:.3f})"
            ax.plot(sub["fpr"], sub["tpr"], lw=1.8,
                    linestyle=style, color=color, label=label)

    ax.plot([0,1],[0,1],"k:",lw=1,alpha=0.4)
    ax.set_title(it_label, fontweight="bold", fontsize=10)
    ax.set_xlabel("FPR", fontsize=9)
    ax.set_ylabel("TPR", fontsize=9)
    ax.set_xlim([-0.02,1.02]); ax.set_ylim([-0.02,1.02])
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="lower right")

for j in range(n, len(axes)):
    axes[j].set_visible(False)

# Shared legend
rf_patch  = mpatches.Patch(color=RF_COLOR, label="Random Forest")
lr_patch  = mpatches.Patch(color=LR_COLOR, label="Logistic Regression")
test_line = mlines.Line2D([], [], color="grey", linestyle="-",  label="Test")
fig.legend(handles=[rf_patch, lr_patch, test_line],
           loc="lower center", ncol=4, fontsize=9,
           bbox_to_anchor=(0.5, -0.03), framealpha=0.85)

fig.suptitle(f"ROC Curves by Input Type — {drug.upper()} Resistance",
             fontweight="bold", fontsize=14, y=1.01)
plt.tight_layout()
Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_plot, dpi=200, bbox_inches="tight")
plt.close()
msg(f"✅ ROC by input type → {out_plot}")
try: log.close()
except: pass