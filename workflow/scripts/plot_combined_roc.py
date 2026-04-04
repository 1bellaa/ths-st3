"""
plot_combined_roc.py — Combined ROC curves
===================================================================================
Usage:
    python plot_combined_roc.py --ml-dir <path/to/ml/dir> [--out-dir <output/dir>]

Expects CSVs named:  {input_type}_{drug}_rf_roc_data.csv
                     {input_type}_{drug}_lr_roc_data.csv

Produces per drug:
  - combined_roc_{drug}.png           RF + LR, all input types
  - combined_roc_{drug}_rf.png        RF only, all input types
  - combined_roc_{drug}_lr.png        LR only, all input types

Compilations:
  - combined_roc.png                  all drugs × all input types (RF + LR)
  - combined_roc_rf.png               all drugs × all input types (RF only)
  - combined_roc_lr.png               all drugs × all input types (LR only)
  - subplot_compilation.png           grid of all per-drug RF+LR plots
  - subplot_compilation_rf.png        grid of all per-drug RF-only plots
  - subplot_compilation_lr.png        grid of all per-drug LR-only plots

Legend:
  - Color  = input_type  (snp=blue, pan=yellow, card=red, snp_card=violet,
                           pan_card=orange, snp_pan=green, snp_pan_card=brown)
  - Style  = model       (RF=solid, LR=dashed)  [only in RF+LR plots]
  - AUC table box in lower-right corner of each plot
"""

import argparse
import sys
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Plot combined ROC for all drugs x input types")
parser.add_argument("--ml-dir",  required=True, help="Directory containing the *_roc_data.csv files")
parser.add_argument("--out-dir", default=".",   help="Output directory for PNG files (default: current dir)")
args = parser.parse_args()

ml_dir  = Path(args.ml_dir)
out_dir = Path(args.out_dir)

if not ml_dir.exists():
    print(f"ERROR: --ml-dir '{ml_dir}' does not exist.", file=sys.stderr)
    sys.exit(1)

out_dir.mkdir(parents=True, exist_ok=True)

# ── Known drugs & input types ─────────────────────────────────────────────────
DRUGS = ["isoniazid", "rifampicin", "streptomycin", "ethambutol"]

INPUT_TYPE_COLORS = {
    "snp":          "#2980b9",   
    "pan":          "#f1c40f",   
    "card":         "#e74c3c",   
    "snp_card":     "#8e44ad",   
    "pan_card":     "#e67e22",   
    "snp_pan":      "#27ae60",   
    "snp_pan_card": "#7f4f24",   
}

MODEL_STYLES = {
    "rf": "-",   
    "lr": "--",   
}

MODEL_LABELS = {
    "rf": "Random Forest",
    "lr": "Logistic Regression",
}

DRUG_STYLES = {
    "isoniazid":   "-",    
    "rifampicin":  "--",  
    "streptomycin": ":",  
    "ethambutol":  "-.", 
}

# ── Filename parsing ──────────────────────────────────────────────────────────
def parse_key(path, model):
    stem = path.stem
    base = stem.replace(f"_{model}_roc_data", "")
    for drug in DRUGS:
        if base.endswith(f"_{drug}"):
            input_type = base[: -(len(drug) + 1)]
            return input_type, drug
    return None, None

# ── Auto-discover all matching CSVs ──────────────────────────────────────────
roc_lookup = {}   # (input_type, drug) -> {"rf": path, "lr": path}
found = 0
for model in ("rf", "lr"):
    for path in sorted(ml_dir.glob(f"*_{model}_roc_data.csv")):
        it, drug = parse_key(path, model)
        if it is None:
            print(f"  WARNING: Could not parse input_type/drug from: {path.name}, skipping.")
            continue
        if it not in INPUT_TYPE_COLORS:
            print(f"  WARNING: Unknown input_type '{it}' in: {path.name}, skipping.")
            continue
        roc_lookup.setdefault((it, drug), {})[model] = path
        found += 1

if found == 0:
    print(f"ERROR: No matching CSVs found in '{ml_dir}'.", file=sys.stderr)
    sys.exit(1)

all_drugs = sorted({drug for (_, drug) in roc_lookup})
print(f"Found {found} ROC CSV(s) | {len(roc_lookup)} combinations | drugs: {all_drugs}\n")

# ── Core plot function ────────────────────────────────────────────────────────
def draw_roc(ax, subset, model_filter=None, drug_linestyle=False):
    """
    Draw ROC curves onto ax.
    subset           : dict (input_type, drug) -> {"rf": path, "lr": path}
    model_filter     : "rf", "lr", or None (both)
    drug_linestyle   : if True, linestyle encodes drug instead of model
    Returns          : list of (label, auc) for AUC table, set of seen input_types, set of seen drugs
    """
    auc_entries      = []
    seen_input_types = set()
    seen_drugs       = set()

    for (input_type, drug), model_paths in sorted(subset.items()):
        color = INPUT_TYPE_COLORS.get(input_type, "black")
        models_to_plot = (
            {k: v for k, v in model_paths.items() if k == model_filter}
            if model_filter else model_paths
        )
        for model, path in sorted(models_to_plot.items()):
            df        = pd.read_csv(path, sep=None, engine="python")
            test_rows = df[df["split"] == "test"]
            if test_rows.empty:
                print(f"  WARNING: No test split in {path.name}, skipping.")
                continue
            auc_val   = df["auc"].max()
            linestyle = DRUG_STYLES.get(drug, "-") if drug_linestyle else MODEL_STYLES[model]
            ax.plot(
                test_rows["fpr"], test_rows["tpr"],
                lw=1.6,
                linestyle=linestyle,
                color=color,
                alpha=0.75,
            )
            seen_input_types.add(input_type)
            seen_drugs.add(drug)
            label = f"{input_type} {drug[:4]}" if drug_linestyle else f"{input_type} {model.upper()}"
            auc_entries.append((label, auc_val))
            print(f"  + {input_type} | {drug} | {model.upper()} AUC={auc_val:.3f}")

    ax.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.6)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    return auc_entries, seen_input_types, seen_drugs


def add_legends(ax, seen_input_types, model_filter=None, drug_linestyle=False, seen_drugs=None):
    """Add input-type colour legend and line style legend to ax."""
    color_patches = [
        mpatches.Patch(color=INPUT_TYPE_COLORS[it], label=it)
        for it in INPUT_TYPE_COLORS
        if it in seen_input_types
    ]
    if drug_linestyle:
        style_lines = [
            mlines.Line2D([], [], color="black", linestyle=DRUG_STYLES[d], lw=2, label=d.capitalize())
            for d in DRUGS if seen_drugs and d in seen_drugs
        ]
        style_title = "Drug"
    elif model_filter:
        style_lines = [
            mlines.Line2D([], [], color="black",
                          linestyle=MODEL_STYLES[model_filter],
                          lw=2, label=MODEL_LABELS[model_filter])
        ]
        style_title = "Model"
    else:
        style_lines = [
            mlines.Line2D([], [], color="black", linestyle=ls, lw=2, label=MODEL_LABELS[m])
            for m, ls in MODEL_STYLES.items()
        ]
        style_title = "Model"
    random_line = mlines.Line2D([], [], color="black", linestyle=":", lw=1.5,
                                label="Random (AUC=0.500)")

    leg1 = ax.legend(handles=color_patches, title="Input type",
                     loc="lower right", fontsize=7, title_fontsize=7.5, framealpha=0.85)
    ax.add_artist(leg1)
    ax.legend(handles=style_lines + [random_line], title=style_title,
              loc="center right", fontsize=7, title_fontsize=7.5, framealpha=0.85)


def add_auc_table(ax, auc_entries):
    """Add a small AUC value table in the lower-right corner of the plot."""
    if not auc_entries:
        return
    table_text = "\n".join(f"{lbl:<22} {auc:.3f}" for lbl, auc in sorted(auc_entries))
    ax.text(
        0.72, 0.02, table_text,
        transform=ax.transAxes,
        fontsize=6.5,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="grey"),
    )


# ── Standalone figure helper ──────────────────────────────────────────────────
def make_roc_plot(subset, title, out_path, model_filter=None, drug_linestyle=False):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    auc_entries, seen_input_types, seen_drugs = draw_roc(ax, subset, model_filter, drug_linestyle)
    add_legends(ax, seen_input_types, model_filter, drug_linestyle, seen_drugs)
    add_auc_table(ax, auc_entries)
    ax.set_title(title, fontweight="bold", fontsize=14)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}\n")


# ── Subplot compilation helper ────────────────────────────────────────────────
def make_subplot_compilation(drug_subsets, suptitle, out_path, model_filter=None, drug_linestyle=False):
    """
    drug_subsets : list of (drug, subset_dict)
    """
    n     = len(drug_subsets)
    ncols = min(2, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(11 * ncols, 8.5 * nrows),
                             squeeze=False)

    for idx, (drug, subset) in enumerate(drug_subsets):
        ax = axes[idx // ncols][idx % ncols]
        auc_entries, seen_input_types, seen_drugs = draw_roc(ax, subset, model_filter, drug_linestyle)
        add_legends(ax, seen_input_types, model_filter, drug_linestyle, seen_drugs)
        add_auc_table(ax, auc_entries)
        model_label = f" [{MODEL_LABELS[model_filter]}]" if model_filter else ""
        ax.set_title(f"{drug.capitalize()}{model_label}", fontweight="bold", fontsize=13)
        ax.set_xlabel("False Positive Rate", fontsize=11)
        ax.set_ylabel("True Positive Rate", fontsize=11)

    # Hide any empty subplots
    for idx in range(n, nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(suptitle, fontweight="bold", fontsize=16, y=1.01)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"  -> Saved: {out_path}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Generate all plots
# ═════════════════════════════════════════════════════════════════════════════

drug_subsets = [(drug, {k: v for k, v in roc_lookup.items() if k[1] == drug})
                for drug in all_drugs]

# ── Per-drug: RF + LR ────────────────────────────────────────────────────────
print("── Per-drug plots (RF + LR) ──")
for drug, subset in drug_subsets:
    print(f"Plotting: {drug}")
    make_roc_plot(subset=subset,
                  title=f"Combined ROC — {drug.capitalize()} — All Input Types",
                  out_path=out_dir / f"combined_roc_{drug}.png")

# ── Per-drug: RF only ─────────────────────────────────────────────────────────
print("── Per-drug plots (RF only) ──")
for drug, subset in drug_subsets:
    print(f"Plotting: {drug} [RF]")
    make_roc_plot(subset=subset,
                  title=f"Combined ROC — {drug.capitalize()} — RF Only",
                  out_path=out_dir / f"combined_roc_{drug}_rf.png",
                  model_filter="rf")

# ── Per-drug: LR only ─────────────────────────────────────────────────────────
print("── Per-drug plots (LR only) ──")
for drug, subset in drug_subsets:
    print(f"Plotting: {drug} [LR]")
    make_roc_plot(subset=subset,
                  title=f"Combined ROC — {drug.capitalize()} — LR Only",
                  out_path=out_dir / f"combined_roc_{drug}_lr.png",
                  model_filter="lr")

# ── Overall: RF + LR ─────────────────────────────────────────────────────────
print("── Overall plot (RF + LR) ──")
make_roc_plot(subset=roc_lookup,
              title="Combined ROC — All Drugs x All Input Types [Both RF + LR]",
              out_path=out_dir / "combined_roc.png")

# ── Overall: RF only ──────────────────────────────────────────────────────────
print("── Overall plot (RF only) ──")
make_roc_plot(subset=roc_lookup,
              title="Combined ROC — All Drugs x All Input Types [RF Only]",
              out_path=out_dir / "combined_roc_rf.png",
              model_filter="rf")

# ── Overall: LR only ──────────────────────────────────────────────────────────
print("── Overall plot (LR only) ──")
make_roc_plot(subset=roc_lookup,
              title="Combined ROC — All Drugs x All Input Types [LR Only]",
              out_path=out_dir / "combined_roc_lr.png",
              model_filter="lr")

# ── Subplot compilations ──────────────────────────────────────────────────────
print("── Subplot compilation (RF + LR) ──")
make_subplot_compilation(drug_subsets,
                         suptitle="ROC Compilation — All Drugs x All Input Types [Both RF + LR]",
                         out_path=out_dir / "subplot_compilation.png")

print("── Subplot compilation (RF only) ──")
make_subplot_compilation(drug_subsets,
                         suptitle="ROC Compilation — All Drugs x All Input Types [RF Only]",
                         out_path=out_dir / "subplot_compilation_rf.png",
                         model_filter="rf")

print("── Subplot compilation (LR only) ──")
make_subplot_compilation(drug_subsets,
                         suptitle="ROC Compilation — All Drugs x All Input Types [LR Only]",
                         out_path=out_dir / "subplot_compilation_lr.png",
                         model_filter="lr")

# ── All drugs, all inputs, RF only (drug linestyle) ──────────────────────────
print("── All drugs, all inputs, RF only ──")
make_roc_plot(subset=roc_lookup,
              title="Combined ROC — All Drugs x All Input Types [RF only]",
              out_path=out_dir / "combined_roc_alldrugs_rf.png",
              model_filter="rf",
              drug_linestyle=True)

# ── All drugs, all inputs, LR only (drug linestyle) ──────────────────────────
print("── All drugs, all inputs, LR only ──")
make_roc_plot(subset=roc_lookup,
              title="Combined ROC — All Drugs x All Input Types [LR only]",
              out_path=out_dir / "combined_roc_alldrugs_lr.png",
              model_filter="lr",
              drug_linestyle=True)

print("Done!")