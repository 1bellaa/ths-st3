"""
plot_feature_venn.py
====================
Reads top-10 feature CSVs (from run_ml.py) for all drugs for a given
model + input_type. Produces:

  1. Membership matrix plot — which features appear in which drugs
  2. CSV of all features with drug memberships + summed importance

Run standalone:
    python plot_feature_venn.py \
        --ml_dir results/ml \
        --input_type snp \
        --model rf \
        --drugs isoniazid rifampicin ethambutol streptomycin \
        --out results/plots
"""

import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

# ── Mode: Snakemake or CLI ────────────────────────────────────────────────────
try:
    feature_files = snakemake.input.feature_files
    out_plot      = snakemake.output.venn_plot
    out_csv       = snakemake.output.venn_csv
    model_type    = snakemake.params.model
    input_type    = snakemake.params.input_type
    drugs         = snakemake.params.drugs
    log_file      = snakemake.log[0]
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_file, "w")
    def msg(m): print(m); log.write(m+"\n"); log.flush()
except NameError:
    p = argparse.ArgumentParser()
    p.add_argument("--ml_dir",     required=True)
    p.add_argument("--input_type", required=True)
    p.add_argument("--model",      default="rf")
    p.add_argument("--drugs",      nargs="+",
                   default=["isoniazid","rifampicin","ethambutol","streptomycin"])
    p.add_argument("--out",        default="results/plots")
    args = p.parse_args()
    ml_dir     = Path(args.ml_dir)
    input_type = args.input_type
    model_type = args.model
    drugs      = args.drugs
    feature_files = [str(ml_dir / f"{input_type}_{d}_{model_type}_features.csv")
                     for d in drugs]
    out_plot   = str(Path(args.out) / f"{input_type}_{model_type}_feature_venn.png")
    out_csv    = str(Path(args.out) / f"{input_type}_{model_type}_feature_venn.csv")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    def msg(m): print(m, flush=True)

msg(f"📊 Feature overlap — {model_type.upper()} [{input_type}]")

# ── Load feature CSVs ─────────────────────────────────────────────────────────
drug_features = {}   # drug → set of feature names
drug_imp      = {}   # drug → {feature: abs_importance}

for f, drug in zip(feature_files, drugs):
    try:
        df = pd.read_csv(f)
        if df.empty or "feature" not in df.columns:
            msg(f"   ⚠️  {drug}: empty or missing 'feature' column")
            continue
        drug_features[drug] = set(df["feature"].tolist())
        drug_imp[drug]      = dict(zip(df["feature"], df["importance"].abs()))
        msg(f"   {drug}: {len(drug_features[drug])} features")
    except Exception as e:
        msg(f"   ⚠️  {drug}: {e}")

if not drug_features:
    for out in [out_plot]:
        fig, ax = plt.subplots(figsize=(5,3))
        ax.text(0.5,0.5,"No feature data",ha="center",va="center",
                transform=ax.transAxes,color="grey")
        ax.axis("off")
        plt.savefig(out,dpi=120,bbox_inches="tight")
        plt.close()
    pd.DataFrame().to_csv(out_csv, index=False)
    sys.exit(0)

present_drugs = sorted(drug_features.keys())
all_features  = set().union(*drug_features.values())

# ── Build membership table ────────────────────────────────────────────────────
records = []
for feat in sorted(all_features):
    membership    = {d: (feat in drug_features[d]) for d in present_drugs}
    n_present     = sum(membership.values())
    imp_sum       = sum(drug_imp[d].get(feat, 0) for d in present_drugs)
    records.append({
        "feature":        feat,
        "n_drugs":        n_present,
        "drugs":          ", ".join(d for d in present_drugs if membership[d]),
        "importance_sum": round(imp_sum, 6),
        **{f"in_{d}": membership[d] for d in present_drugs},
    })

feat_df = pd.DataFrame(records).sort_values(
    ["n_drugs","importance_sum"], ascending=[False,False]
).reset_index(drop=True)
feat_df.to_csv(out_csv, index=False)
msg(f"   Unique features: {len(feat_df)}")
msg(f"   In ALL {len(present_drugs)} drugs: {(feat_df['n_drugs']==len(present_drugs)).sum()}")

# ── Plot: membership matrix + importance bar ──────────────────────────────────
DRUG_COLORS = ["#e74c3c","#e67e22","#2ecc71","#3498db",
               "#9b59b6","#1abc9c","#e91e63","#ff9800"]
color_map = {d: DRUG_COLORS[i % len(DRUG_COLORS)] for i,d in enumerate(present_drugs)}

features_ordered = feat_df["feature"].tolist()
n_feats   = len(features_ordered)
n_drugs   = len(present_drugs)
fig_h     = max(8, n_feats * 0.28 + 3)

fig = plt.figure(figsize=(14, fig_h))
gs  = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.04)
ax_m = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])

y_pos = np.arange(n_feats)

for xi, drug_name in enumerate(present_drugs):
    for yi, feat in enumerate(features_ordered):
        in_drug = feat_df.loc[yi, f"in_{drug_name}"]
        if in_drug:
            ax_m.scatter(xi, yi, s=90, color=color_map[drug_name],
                         zorder=3, edgecolors="white", linewidths=0.5)
        else:
            ax_m.scatter(xi, yi, s=35, color="#ecf0f1",
                         zorder=2, edgecolors="#bdc3c7", linewidths=0.5)

# Connect dots across drugs for multi-drug features
for yi, feat in enumerate(features_ordered):
    xs = [xi for xi, d in enumerate(present_drugs)
          if feat_df.loc[yi, f"in_{d}"]]
    if len(xs) > 1:
        ax_m.hlines(yi, min(xs), max(xs), colors="#7f8c8d", lw=1.5, zorder=1)

ax_m.set_xlim(-0.5, n_drugs - 0.5)
ax_m.set_ylim(-0.5, n_feats - 0.5)
ax_m.set_xticks(range(n_drugs))
ax_m.set_xticklabels([d.upper()[:4] for d in present_drugs], fontsize=9, fontweight="bold")
ax_m.set_yticks(y_pos)
ax_m.set_yticklabels(features_ordered, fontsize=7)
ax_m.set_xlabel("Drug", fontsize=10)
ax_m.set_title("Feature × Drug Membership", fontweight="bold", fontsize=11)
ax_m.grid(True, axis="x", alpha=0.15)
ax_m.invert_yaxis()

# Importance bar
imp_vals   = feat_df["importance_sum"].values
n_drug_vals = feat_df["n_drugs"].values
bar_colors = [DRUG_COLORS[min(v-1, len(DRUG_COLORS)-1)] for v in n_drug_vals]
ax_b.barh(y_pos, imp_vals, color=bar_colors, alpha=0.85, edgecolor="white")
ax_b.set_xlabel("Σ |Importance|", fontsize=9)
ax_b.set_yticks(y_pos)
ax_b.set_yticklabels([])
ax_b.set_title("Summed Importance", fontweight="bold", fontsize=11)
ax_b.grid(True, axis="x", alpha=0.3)
ax_b.invert_yaxis()

legend_patches = [
    mpatches.Patch(color=DRUG_COLORS[i], alpha=0.85,
                   label=f"In {i+1} drug{'s' if i>0 else ''}")
    for i in range(min(n_drugs, 4))
]
ax_b.legend(handles=legend_patches, fontsize=8, loc="lower right", framealpha=0.85)

fig.suptitle(
    f"Top-10 Feature Overlap Across Drugs\n{model_type.upper()} — {input_type}",
    fontweight="bold", fontsize=13, y=1.01,
)
plt.savefig(out_plot, dpi=180, bbox_inches="tight")
plt.close()
msg(f"✅ Feature venn → {out_plot}")
try: log.close()
except: pass