"""
replot_feature_importance.py
============================
Re-generates feature importance PNGs using already-saved *_features.csv files.
Annotates raw feature names → gene names using tbdb.bed and Panaroo table.
Does NOT rerun any ML.

Usage:
    python replot_feature_importance.py \
        --feature_dir   results/ml \
        --plot_dir      results/ml \
        --tbdb_bed      reference/tbdb/tbdb.bed \
        --pan_gene_table results/pangenome/gene_presence_absence.csv \
        --input_types   snp pan card snp_card snp_pan pan_card snp_pan_card \
        --drugs         isoniazid rifampicin ethambutol streptomycin \
        --models        rf lr \
        --dpi           200

File conventions (must match your Snakemake outputs):
    features CSV  →  {feature_dir}/{input_type}_{drug}_{model}_features.csv
    features PNG  →  {plot_dir}/{input_type}_{drug}_{model}_features.png
"""

import argparse
import re
import sys
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


p = argparse.ArgumentParser(description="Re-plot feature importance with gene names")
p.add_argument("--feature_dir",    required=True,  help="Dir containing *_features.csv files")
p.add_argument("--plot_dir",       default=None,   help="Dir to write PNGs (default: same as feature_dir)")
p.add_argument("--tbdb_bed",       default=None)
p.add_argument("--h37rv_gff",      default=None,
               help="Full H37Rv GFF for whole-genome position → gene lookup")
p.add_argument("--pan_gene_table", default=None)
p.add_argument("--input_types",    nargs="+",
               default=["snp","pan","card","snp_card","snp_pan","pan_card","snp_pan_card"])
p.add_argument("--drugs",          nargs="+",
               default=["isoniazid","rifampicin","ethambutol","streptomycin",
                        "pyrazinamide","levofloxacin"])
p.add_argument("--models",         nargs="+", default=["rf","lr"])
p.add_argument("--dpi",            type=int,  default=200)
p.add_argument("--dry_run",        action="store_true",
               help="Print what would be done without writing files")
args = p.parse_args()

feature_dir = Path(args.feature_dir)
plot_dir    = Path(args.plot_dir) if args.plot_dir else feature_dir
plot_dir.mkdir(parents=True, exist_ok=True)

def msg(m): print(m, flush=True)

# 1. Build annotation maps 
SNP_RE = re.compile(r"^.+?_(\d+)(?:_[A-Za-z]+_[A-Za-z]+)?$")

pos_map: dict[int, str] = {}   # genomic position → gene name
pan_map: dict[str, str] = {}   # group_XXXX       → gene name

if args.tbdb_bed and Path(args.tbdb_bed).exists():
    msg(f"\n📋 Loading tbdb.bed: {args.tbdb_bed}")
    bed = pd.read_csv(args.tbdb_bed, sep="\t", header=None, low_memory=False)
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
    msg("⚠️  tbdb.bed not provided / not found — SNP features keep raw position names")

gff_pos_map: dict[int, str] = {}   # whole-genome fallback

if args.h37rv_gff and Path(args.h37rv_gff).exists():
    msg(f"\n📋 Loading H37Rv GFF: {args.h37rv_gff}")
    with open(args.h37rv_gff) as fh:
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
    msg("⚠️  h37rv_gff not provided — unannotated SNPs will keep raw position names")

if args.pan_gene_table and Path(args.pan_gene_table).exists():
    msg(f"\n📋 Loading Panaroo table: {args.pan_gene_table}")
    pan = pd.read_csv(args.pan_gene_table, low_memory=False)
    for _, row in pan.iterrows():
        group     = str(row["Gene"]).strip()
        gene_name = str(row.get("Non-unique Gene name", "")).strip()
        # Prokka gene name only — skip Annotation fallback (too vague)
        if gene_name and gene_name.lower() not in ("nan", "", "hypothetical protein"):
            pan_map[group] = gene_name
        # else: don't add → resolve() returns raw group ID
    msg(f"   {len(pan_map)} groups with real Prokka gene names")
else:
    msg("⚠️  pan_gene_table not provided / not found — pangenome features keep group IDs")


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
    resolved   = [resolve(f) for f in features]
    gene_names = [r[0] for r in resolved]
    raw_names  = [r[1] for r in resolved]

    from collections import Counter
    counts = Counter(gene_names)

    labels = []
    for gene, raw in zip(gene_names, raw_names):
        if counts[gene] > 1:
            # Extract just the numeric position from the raw feature name
            m = SNP_RE.match(raw)
            if m:
                pos = m.group(1)                  # e.g. "4215433"
                labels.append(f"{gene} ({pos})")  # → "dnaA (1742)"
            elif raw.startswith("group_"):
                group_num = raw.split("_")[-1]    # e.g. "group_1874" → "1874"
                labels.append(f"{gene} ({group_num})")
            else:
                labels.append(f"{gene} ({raw})")  # fallback
        else:
            labels.append(gene)
    return labels

# 2. Iterate over every (input_type, drug, model) combo 
n_done = n_skip = n_miss = 0

for input_type in args.input_types:
    for drug in args.drugs:
        for model_type in args.models:
            stem     = f"{input_type}_{drug}_{model_type}"
            csv_path = feature_dir / f"{stem}_features.csv"
            png_path = plot_dir    / f"{stem}_features.png"

            if not csv_path.exists():
                msg(f"   ⚪ SKIP (no CSV): {csv_path.name}")
                n_miss += 1
                continue

            df = pd.read_csv(csv_path)
            if df.empty or "feature" not in df.columns or "importance" not in df.columns:
                msg(f"   ⚠️  Empty/malformed CSV: {csv_path.name}")
                n_skip += 1
                continue

            # Take top 10 by |importance| (CSV may already be sorted, but be safe)
            df = df.reindex(df["importance"].abs().sort_values(ascending=False).index).head(10)
            df = df.reset_index(drop=True)

            # Resolve gene names
            #df["gene_name"] = df["feature"].apply(resolve)
            df["gene_name"] = make_display_labels(df["feature"].tolist())
            n_resolved = (df["gene_name"] != df["feature"]).sum()

            label_name = "Random Forest" if model_type == "rf" else "Logistic Regression"
            imp_label  = "Importance"    if model_type == "rf" else "Coefficient"

            msg(f"   🔬 {stem}: {n_resolved}/{len(df)} names resolved → {png_path.name}")

            if args.dry_run:
                for _, row in df.iterrows():
                    arrow = "→" if row["gene_name"] != row["feature"] else "  "
                    msg(f"      {row['feature']:<40s} {arrow} {row['gene_name']}")
                n_done += 1
                continue

            # ── Plot ──────────────────────────────────────────────────────────
            fig, ax = plt.subplots(figsize=(8, 5))

            if model_type == "lr":
                colors = ["#d62728" if v > 0 else "#2ca02c"
                          for v in df["importance"].values]
                ax.barh(range(len(df)), df["importance"].values,
                        color=colors, alpha=0.85)
                ax.axvline(0, color="black", lw=0.8)
                ax.legend(handles=[
                    mpatches.Patch(fc="#d62728", alpha=0.85, label="↑ Promotes Resistance"),
                    mpatches.Patch(fc="#2ca02c", alpha=0.85, label="↓ Promotes Susceptibility"),
                ], fontsize=9)
            else:
                colors = sns.color_palette("YlGn", len(df))[::-1]
                ax.barh(range(len(df)), df["importance"].values,
                        color=colors, alpha=0.85)

            ax.set_yticks(range(len(df)))
            ax.set_yticklabels(df["gene_name"].values, fontsize=9)   # ← gene names here
            ax.invert_yaxis()
            ax.set_xlabel(imp_label)
            ax.set_title(
                f"Top 10 Features — {label_name} — {drug.capitalize()} [{input_type}]",
                fontweight="bold",
            )
            ax.grid(True, axis="x", alpha=0.3)
            plt.tight_layout()
            plt.savefig(png_path, dpi=args.dpi, bbox_inches="tight")
            plt.close()

            # Also write gene_name back to the CSV so annotate_features.py sees it
            full_csv = pd.read_csv(csv_path)
            #full_csv["gene_name"] = full_csv["feature"].apply(resolve)
            full_csv["gene_name"] = make_display_labels(full_csv["feature"].tolist())
            full_csv.to_csv(csv_path, index=False)

            n_done += 1


# ── 4. Combined subplot figures ───────────────────────────────────────────────
# Layout: rows = input_types, cols = drugs
# Figure A (per model): single input types — snp, pan, card
# Figure B (per model): combo input types  — snp_pan, snp_card, pan_card, snp_pan_card

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
    """Draw one feature importance bar chart into ax."""
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


def make_combined_figure(model_type, input_type_group, group_tag):
    """Build and save one combined subplot figure."""
    if args.dry_run:
        msg(f"   DRY RUN — would build combined: {model_type}_{group_tag}")
        return

    drugs       = args.drugs
    n_rows      = len(input_type_group)
    n_cols      = len(drugs)
    label_name  = "Random Forest" if model_type == "rf" else "Logistic Regression"

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(4.5 * n_cols, 4 * n_rows),
        squeeze=False,
    )

    for r, input_type in enumerate(input_type_group):
        for c, drug in enumerate(drugs):
            ax       = axes[r][c]
            csv_path = feature_dir / f"{input_type}_{drug}_{model_type}_features.csv"

            df = None
            if csv_path.exists():
                try:
                    raw = pd.read_csv(csv_path)
                    if not raw.empty and "feature" in raw.columns and "importance" in raw.columns:
                        raw = raw.reindex(
                            raw["importance"].abs().sort_values(ascending=False).index
                        ).head(10).reset_index(drop=True)
                        #raw["gene_name"] = raw["feature"].apply(resolve)
                        raw["gene_name"] = make_display_labels(raw["feature"].tolist())
                        df = raw
                except Exception as e:
                    msg(f"   ⚠️  Could not load {csv_path.name}: {e}")

            _draw_subplot(ax, df, model_type, drug, input_type)

        # Row label on the left-most subplot
        axes[r][0].set_ylabel(
            INPUT_TYPE_LABELS.get(input_type, input_type),
            fontsize=9, fontweight="bold", labelpad=8,
        )

    fig.suptitle(
        f"{label_name} — Top 10 Features\n({group_tag.replace('_', ' + ')})",
        fontsize=13, fontweight="bold", y=1.01,
    )

    # Shared LR legend — one instance bottom-right
    if model_type == "lr":
        fig.legend(
            handles=[
                mpatches.Patch(fc="#d62728", alpha=0.85, label="↑ Promotes Resistance"),
                mpatches.Patch(fc="#2ca02c", alpha=0.85, label="↓ Promotes Susceptibility"),
            ],
            loc="lower right", fontsize=9, framealpha=0.8,
        )

    plt.tight_layout()
    out_path = plot_dir / f"combined_{model_type}_{group_tag}_features.png"
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close()
    msg(f"✅ Combined figure → {out_path}")


if not args.dry_run:
    msg(f"\n{'='*55}")
    msg("  Building combined subplot figures...")
    msg(f"{'='*55}")

for model_type in args.models:
    make_combined_figure(model_type, SINGLE_TYPES, "single")
    make_combined_figure(model_type, COMBO_TYPES,  "combo")

# ── 3. Summary ────────────────────────────────────────────────────────────────
msg(f"\n{'='*55}")
msg(f"  ✅ Re-plotted : {n_done}")
msg(f"  ⚪ Missing CSV: {n_miss}")
msg(f"  ⚠️  Skipped    : {n_skip}")
msg(f"{'='*55}")
if args.dry_run:
    msg("  DRY RUN — no files written")