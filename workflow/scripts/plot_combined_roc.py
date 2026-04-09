"""
plot_combined_roc.py — Combined ROC curves
===================================================================================
Usage (standalone):
    python plot_combined_roc.py --ml-dir <path/to/ml/dir> [--out-dir <output/dir>]

When run via Snakemake, all inputs/outputs/params are taken from the rule.

Expects CSVs named:  {input_type}_{drug}_rf_roc_data.csv
                     {input_type}_{drug}_lr_roc_data.csv

Produces per drug:
  - {input_type}_{drug}_combined_roc.png       RF + LR
  - {input_type}_{drug}_combined_roc_rf.png    RF only
  - {input_type}_{drug}_combined_roc_lr.png    LR only

Legend:
  - Color  = input_type  (snp=blue, pan=yellow, card=red, snp_card=violet,
                           pan_card=orange, snp_pan=green, snp_pan_card=brown)
  - Style  = model       (RF=solid, LR=dashed)  [only in RF+LR plots]
  - AUC table box in lower-right corner of each plot
"""

import sys
import math
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
from pathlib import Path

# ── Input handling: Snakemake or CLI ─────────────────────────────────────────
if "snakemake" in dir():
    rf_path    = Path(snakemake.input.rf_roc_data)
    lr_path    = Path(snakemake.input.lr_roc_data)
    out_combined    = Path(snakemake.output.combined_roc)
    out_combined_rf = Path(snakemake.output.combined_roc_rf)
    out_combined_lr = Path(snakemake.output.combined_roc_lr)
    drug       = snakemake.params.drug
    input_type = snakemake.params.input_type
    out_dir    = out_combined.parent
else:
    import argparse
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

    # In CLI mode: glob-discover
    rf_path    = None
    lr_path    = None
    drug       = None
    input_type = None
    out_combined    = None
    out_combined_rf = None
    out_combined_lr = None

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

DRUG_ABBR = {
    "isoniazid": "INH",
    "rifampicin": "RIF",
    "ethambutol": "EMB",
    "streptomycin": "STM",
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
    "isoniazid":    "-",
    "rifampicin":   "--",
    "streptomycin": ":",
    "ethambutol":   "-.",
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

# ── Build roc_lookup ──────────────────────────────────────────────────────────
if "snakemake" in dir():
    # Snakemake: exactly one input_type/drug combination from the rule
    roc_lookup = {
        (input_type, drug): {
            "rf": rf_path,
            "lr": lr_path,
        }
    }
    all_drugs = [drug]
    print(f"Snakemake mode | input_type={input_type} | drug={drug}\n")
else:
    # CLI mode: glob-discover all matching CSVs exactly as before
    roc_lookup = {}
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
            drug_label = DRUG_ABBR.get(drug, drug)
            label = (
                f"{input_type} | {drug_label} {model.upper()}"
                if not drug_linestyle
                else f"{input_type} {drug_label}"
            )
            auc_entries.append((label, auc_val))
            print(f"  + {input_type} | {drug} | {model.upper()} AUC={auc_val:.3f}")

    ax.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.6)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.grid(True, alpha=0.3)

    return auc_entries, seen_input_types, seen_drugs


def add_legends(ax, seen_input_types, model_filter=None, drug_linestyle=False, seen_drugs=None):
    """input-type colour legend and line style legend to ax."""
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
    """small AUC value table in the lower-right corner of the plot."""
    if not auc_entries:
        return
    table_text = "\n".join(f"{lbl:<22} {auc:.3f}" for lbl, auc in sorted(auc_entries))
    ax.text(
        0.69, 0.02, table_text,
        transform=ax.transAxes,
        fontsize=6.5,
        verticalalignment="bottom",
        horizontalalignment="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8, edgecolor="grey"),
    )


# ── figure helper ──────────────────────────────────────────────────
def make_roc_plot(subset, title, out_path, model_filter=None, drug_linestyle=False):
    fig, ax = plt.subplots(figsize=(11, 8.5))
    auc_entries, seen_input_types, seen_drugs = draw_roc(ax, subset, model_filter, drug_linestyle)
    add_legends(ax, seen_input_type