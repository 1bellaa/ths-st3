"""
replot_feature_importance.py
============================
Re-generates feature importance PNGs using already-saved *_features.csv files.
Annotates raw feature names → gene names using tbdb.bed and Panaroo table.
Does NOT rerun any ML.

File conventions (must match your Snakemake outputs):
    features CSV  →  {feature_dir}/{input_type}_{drug}_{model}_features.csv
    features PNG  →  {plot_dir}/{input_type}_{drug}_{model}_features.png
"""
"""
replot_feature_importance.py
============================
Re-generates feature importance PNGs using already-saved *_features.csv files.
Annotates raw feature names -> gene names using tbdb.bed and Panaroo table.
Does NOT rerun any ML.

Run via Snakemake only. All inputs/outputs/params taken from the rule:
    params:
        feature_dir    = str(ML_DIR)
        plot_dir       = str(ML_DIR)
        tbdb_bed       = config.get("tbdb_bed", None)
        h37rv_gff      = config.get("h37rv_gff", None)
        pan_gene_table = str(RESULTS_DIR / "pangenome" / "gene_presence_absence.csv")
        input_types    = ML_INPUT_TYPES
        drugs          = DRUGS
        models         = ["rf", "lr"]
        dpi            = config.get("replot_dpi", 200)
    output:
        combined_rf_single = ML_DIR / "combined_rf_single_features.png",
        combined_rf_combo  = ML_DIR / "combined_rf_combo_features.png",
        combined_lr_single = ML_DIR / "combined_lr_single_features.png",
        combined_lr_combo  = ML_DIR / "combined_lr_combo_features.png",
        done               = touch(ML_DIR / ".replot_features.done"),
"""

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"

feature_dir    = Path(snakemake.params.feature_dir)
plot_dir       = Path(snakemake.params.plot_dir)
tbdb_bed       = snakemake.params.get("tbdb_bed",       None)
h37rv_gff      = snakemake.params.get("h37rv_gff",      None)
pan_gene_table = snakemake.params.get("pan_gene_table", None)
input_types    = list(snakemake.params.input_types)
drugs          = list(snakemake.params.drugs)
models         = list(snakemake.params.models)
dpi            = int(snakemake.params.get("dpi", 200))

plot_dir.mkdir(parents=True, exist_ok=True)

def msg(m): print(m, flush=True)

# 1. Build annotation maps 

SNP_RE = re.compile(r"^.+?_(\d+)(?:_[A-Za-z]+_[A-Za-z]+)?$")

pos_map:     dict[int, str] = {}
gff_pos_map: dict[int, str] = {}
pan_map:     dict[str, str] = {}

if tbdb_bed and Path(tbdb_bed).exists():
    msg(f"\n📋 Loading tbdb.bed: {tbdb_bed}")
    bed = pd.read_csv(tbdb_bed, sep="\t", header=None, low_memory=False)
    for _, row in bed.iterrows():
        try:
            start = int(row[1]); end = int(row[2])
            locus = str(row[3]).strip()
            gene  = str(row[4]).strip()
            label = gene if gene and gene not in (".", "-", "nan", locus) else locus
            for pos in range(start, end + 1):
                if pos not in pos_map:
                    pos_map[pos] = label
        except (ValueError, IndexError):
            continue
    msg(f"   {len(pos_map):,} positions mapped")
else:
    msg("WARNING: tbdb.bed not provided / not found -- SNP features keep raw position names")

if h37rv_gff and Path(h37rv_gff).exists():
    msg(f"\n📋 Loading H37Rv GFF: {h37rv_gff}")
    with open(h37rv_gff) as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            parts = line.strip().split("\t")
            if len(parts) < 9 or parts[2] not in ("gene", "CDS"):
                continue
            start = int(parts[3]); end = int(parts[4])
            attrs = parts[8]
            gene_name = locus_tag = ""
            for attr in attrs.split(";"):
                if attr.startswith("gene="):
                    gene_name = attr[5:].strip()
                if attr.startswith("locus_tag="):
                    locus_tag = attr[10:].strip()
            label = gene_name if gene_name else locus_tag
            if not label:
                continue
            for pos in range(start, end + 1):
                if pos not in gff_pos_map:
                    gff_pos_map[pos] = label
    msg(f"   GFF: {len(gff_pos_map):,} positions mapped")
else:
    msg("WARNING: h37rv_gff not provided -- unannotated SNPs will keep raw position names")

if pan_gene_table and Path(pan_gene_table).exists():
    msg(f"\n📋 Loading Panaroo table: {pan_gene_table}")
    pan = pd.read_csv(pan_gene_table, low_memory=False)
    for _, row in pan.iterrows():
        group     = str(row["Gene"]).strip()
        gene_name = str(row.get("Non-unique Gene name", "")).strip()
        if gene_name and gene_name.lower() not in ("nan", "", "hypothetical protein"):
            pan_map[group] = gene_name
    msg(f"   {len(pan_map)} groups with real Prokka gene names")
else:
    msg("WARNING: pan_gene_table not provided / not found -- pangenome features keep group IDs")


# 2. Name resolution 

def resolve(feat: str) -> tuple[str, str]:
    """Returns (gene_name, raw_feature). gene_name falls back to raw if unknown."""
    if feat.startswith("CARD_"):
        return feat[5:], feat
    if feat.startswith("country_"):
        return feat, feat
    if feat in pan_map:
        return pan_map[feat], feat
    m = SNP_RE.match(feat)
    if m:
        pos = int(m.group(1))
        if pos in pos_map:
            return pos_map[pos], feat
        if pos in gff_pos_map:
            return gff_pos_map[pos], feat
        return feat, feat
    return feat, feat


def make_display_labels(features: list[str]) -> list[str]:
    from collections import Counter
    resolved   = [resolve(f) for f in features]
    gene_names = [r[0] for r in resolved]
    raw_names  = [r[1] for r in resolved]
    counts     = Counter(gene_names)
    labels     = []
    for gene, raw in zip(gene_names, raw_names):
        if counts[gene] > 1:
            m = SNP_RE.match(raw)
            if m:
                labels.append(f"{gene} ({m.group(1)})")
            elif raw.startswith("group_"):
                labels.append(f"{gene} ({raw.split('_')[-1]})")
            else:
                labels.append(f"{gene} ({raw})")
        else:
            labels.append(gene)
    return labels


# 3. Per-combo individual plots 

n_done = n_skip = n_miss = 0

for input_type in input_types:
    for drug in drugs:
        for model_type in models:
            stem     = f"{input_type}_{drug}_{model_type}"
            csv_path = feature_dir / f"{stem}_features.csv"
            png_path = plot_dir    / f"{stem}_features.png"

            if not csv_path.exists():
                msg(f"   SKIP (no CSV): {csv_path.name}")
                n_miss += 1
                continue

            df = pd.read_csv(csv_path)
            if df.empty or "feature" not in df.columns or "importance" not in df.columns:
                msg(f"   WARNING: Empty/malformed CSV: {csv_path.name}")
                n_skip += 1
                continue

            df = df.reindex(df["importance"].abs().sort_values(ascending=False).index).head(10)
            df = df.reset_index(drop=True)
            df["gene_name"] = make_display_labels(df["feature"].tolist())
            n_resolved = (df["gene_name"] != df["feature"]).sum()

            label_name = "Random Forest" if model_type == "rf" else "Logistic Regression"
            imp_label  = "Importance"    if model_type == "rf" else "Coefficient"

            msg(f"   {stem}: {n_resolved}/{len(df)} names resolved -> {png_path.name}")

            fig, ax = plt.subplots(figsize=(8, 5))

            if model_type == "lr":
                colors = ["#d62728" if v > 0 else "#2ca02c" for v in df["importance"].values]
                ax.barh(range(len(df)), df["importance"].values, color=colors, alpha=0.85)
                ax.axvline(0, color="black", lw=0.8)
                ax.legend(handles=[
                    mpatches.Patch(fc="#d62728", alpha=0.85, label="Promotes Resistance"),
                    mpatches.Patch(fc="#2ca02c", alpha=0.85, label="Promotes Susceptibility"),
                ], fontsize=9)
            else:
                colors = sns.color_palette("YlGn", len(df))[::-1]
                ax.barh(range(len(df)), df["importance"].values, color=colors, alpha=0.85)

            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df["gene_name"].values, fontsize=9)
            ax.invert_yaxis()
            ax.set_xlabel(imp_label)
            ax.set_title(
                f"Top 10 Features -- {label_name} -- {drug.capitalize()} [{input_type}]",
                fontweight="bold",
            )
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=dpi, bbox_inches="tight")
            plt.close()

            # Write gene_name back to CSV so downstream scripts see it
            full_csv = pd.read_csv(csv_path)
            full_csv["gene_name"] = make_display_labels(full_csv["feature"].tolist())
            full_csv.to_csv(csv_path, index=False)

            n_done += 1


# 4. Combined subplot figures 

SINGLE_TYPES = ["snp", "pan", "card"]
COMBO_TYPES  = ["snp_pan", "snp_card", "pan_card", "snp_pan_card"]

INPUT_TYPE_LABELS = {
    "snp":          "SNP",
    "pan":          "Pangenome",
    "card":         "CARD",
    "snp_pan":      "SNP + Pangenome",
    "snp_card":     "SNP + CARD",
    "pan_card":     "Pangenome + CARD",
    "snp_pan_card": "SNP + Pan + CARD",
}


def _draw_subplot(ax, df, model_type, drug, input_type):
    imp_label = "Importance" if model_type == "rf" else "Coefficient"
    if df is None or df.empty:
        ax.text(0.5, 0.5, "No data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9, color="grey")
        ax.set_title(f"{drug.capitalize()}\n[{INPUT_TYPE_LABELS.get(input_type, input_type)}]",
                     fontsize=8, fontweight="bold")
        ax.axis("off")
        return
    if model_type == "lr":
        colors = ["#d62728" if v > 0 else "#2ca02c" for v in df["importance"].values]
        ax.barh(range(len(df)), df["importance"].values, color=colors, alpha=0.85)
        ax.axvline(0, color="black", lw=0.6)
    else:
        colors = sns.color_palette("YlGn", len(df))[::-1]
        ax.barh(range(len(df)), df["importance"].values, color=colors, alpha=0.85)
    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["gene_name"].values, fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel(imp_label, fontsize=7)
    ax.set_title(f"{drug.capitalize()}\n[{INPUT_TYPE_LABELS.get(input_type, input_type)}]",
                 fontsize=8, fontweight="bold")
    ax.tick_params(axis="x", labelsize=7)
    ax.grid(True, axis="x", alpha=0.3)


def make_combined_figure(model_type, input_type_group, group_tag, out_path):
    n_rows     = len(input_type_group)
    n_cols     = len(drugs)
    label_name = "Random Forest" if model_type == "rf" else "Logistic Regression"

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for r, input_type in enumerate(input_type_group):
        for c, drug in enumerate(drugs):
            ax       = axes[r][c]
            csv_path = feature_dir / f"{input_type}_{drug}_{model_type}_features.csv"
            df       = None
            if csv_path.exists():
                try:
                    raw = pd.read_csv(csv_path)
                    if not raw.empty and "feature" in raw.columns and "importance" in raw.columns:
                        raw = raw.reindex(
                            raw["importance"].abs().sort_values(ascending=False).index
                        ).head(10).reset_index(drop=True)
                        raw["gene_name"] = make_display_labels(raw["feature"].tolist())
                        df = raw
                except Exception as e:
                    msg(f"   WARNING: Could not load {csv_path.name}: {e}")
            _draw_subplot(ax, df, model_type, drug, input_type)

        axes[r][0].set_ylabel(
            INPUT_TYPE_LABELS.get(input_type, input_type),
            fontsize=9, fontweight="bold", labelpad=8,
        )

    fig.suptitle(
        f"{label_name} -- Top 10 Features\n({group_tag.replace('_', ' + ')})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    if model_type == "lr":
        fig.legend(handles=[
            mpatches.Patch(fc="#d62728", alpha=0.85, label="Promotes Resistance"),
            mpatches.Patch(fc="#2ca02c", alpha=0.85, label="Promotes Susceptibility"),
        ], loc="lower right", fontsize=9, framealpha=0.8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    msg(f"Combined figure -> {out_path}")


msg(f"\n{'='*55}")
msg("  Building combined subplot figures...")
msg(f"{'='*55}")

make_combined_figure("rf", SINGLE_TYPES, "single", snakemake.output.combined_rf_single)
make_combined_figure("rf", COMBO_TYPES,  "combo",  snakemake.output.combined_rf_combo)
make_combined_figure("lr", SINGLE_TYPES, "single", snakemake.output.combined_lr_single)
make_combined_figure("lr", COMBO_TYPES,  "combo",  snakemake.output.combined_lr_combo)

# ── 5. Summary ────────────────────────────────────────────────────────────────

msg(f"\n{'='*55}")
msg(f"  Re-plotted : {n_done}")
msg(f"  Missing CSV: {n_miss}")
msg(f"  Skipped    : {n_skip}")
msg(f"{'='*55}")