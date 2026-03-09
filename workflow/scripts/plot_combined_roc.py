"""
plot_combined_roc.py — Combined ROC curve: RF + LR on same plot
================================================================
One plot per drug per input_type, showing:
  - RF test ROC  (solid green)
  - RF val ROC   (dashed green)
  - LR test ROC  (solid blue)
  - LR val ROC   (dashed blue)
  - Random baseline (black dotted)

Train curves are omitted from combined plot to keep it readable.
"""

import sys
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

# ── Snakemake bindings ────────────────────────────────────────────────────────
rf_roc_data  = snakemake.input.rf_roc_data
lr_roc_data  = snakemake.input.lr_roc_data
out_combined = snakemake.output.combined_roc
drug         = snakemake.params.drug
input_type   = snakemake.params.input_type
log_file     = snakemake.log[0]

Path(log_file).parent.mkdir(parents=True, exist_ok=True)
log = open(log_file, "w")

def msg(m):
    print(m, flush=True)
    log.write(m + "\n")
    log.flush()

msg(f"📊 Combined ROC: {drug.upper()} [{input_type}]")

# ── Load ROC data ─────────────────────────────────────────────────────────────
rf_df = pd.read_csv(rf_roc_data)
lr_df = pd.read_csv(lr_roc_data)

def get_auc(df, split):
    rows = df[df["split"] == split]
    return rows["auc"].iloc[0] if not rows.empty else None

# ── Plot ──────────────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6.5))

RF_COLOR = "#27ae60"   # green
LR_COLOR = "#2980b9"   # blue

plot_splits = [
    ("test",  "-",  "Test"),
    ("val",   "--", "Val"),
]

for split, style, split_label in plot_splits:
    # RF
    rf_sub = rf_df[rf_df["split"] == split]
    if not rf_sub.empty:
        auc = get_auc(rf_df, split)
        ax.plot(rf_sub["fpr"], rf_sub["tpr"],
                lw=2, linestyle=style, color=RF_COLOR,
                label=f"RF {split_label} (AUC={auc:.3f})")

    # LR
    lr_sub = lr_df[lr_df["split"] == split]
    if not lr_sub.empty:
        auc = get_auc(lr_df, split)
        ax.plot(lr_sub["fpr"], lr_sub["tpr"],
                lw=2, linestyle=style, color=LR_COLOR,
                label=f"LR {split_label} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.5, label="Random (AUC=0.500)")

# Legend with model colour patches
import matplotlib.patches as mpatches
import matplotlib.lines as mlines

rf_patch   = mpatches.Patch(color=RF_COLOR, label="Random Forest")
lr_patch   = mpatches.Patch(color=LR_COLOR, label="Logistic Regression")
test_line  = mlines.Line2D([], [], color="grey", linestyle="-",  label="Test split")
val_line   = mlines.Line2D([], [], color="grey", linestyle="--", label="Val split")

handles, labels = ax.get_legend_handles_labels()
legend1 = ax.legend(handles=handles, loc="lower right", fontsize=8.5)
ax.add_artist(legend1)
ax.legend(handles=[rf_patch, lr_patch, test_line, val_line],
          loc="upper left", fontsize=8.5, framealpha=0.8)

ax.set_title(
    f"RF vs LR — {drug.capitalize()} Resistance [{input_type}]",
    fontweight="bold", fontsize=13,
)
ax.set_xlabel("False Positive Rate", fontsize=11)
ax.set_ylabel("True Positive Rate", fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])

plt.tight_layout()
Path(out_combined).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_combined, dpi=200, bbox_inches="tight")
plt.close()

msg(f"✅ Combined ROC → {out_combined}")
log.close()