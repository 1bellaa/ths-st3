"""
annotate_features.py
====================
Maps raw feature names to human-readable gene names using:

  SNP features ("Chromosome_761155" or "Chromosome_761155_C_G"):
    → tbdb.bed: col[4] = gene name (dnaA, rpoB, etc.)
       BED columns: Chromosome  start  end  locus(Rv)  gene_name  drugs

  Pangenome features ("group_00123"):
    → Panaroo gene_presence_absence.csv
      - "Non-unique Gene name" if not empty
      - else "Annotation" (functional description)

  CARD features ("CARD_rpoB2"):
    → Already named, strip "CARD_" prefix

Outputs:
  annotated_features.csv  — all features with gene_name column
  annotation_summary.png  — coverage breakdown chart

Run standalone:
    python workflow/scripts/annotate_features.py \
        --feature_dir results/ml \
        --input_types snp pan card \
        --drugs isoniazid rifampicin ethambutol streptomycin \
        --model rf \
        --tbdb_bed reference/tbdb/tbdb.bed \
        --pan_gene_table results/pangenome/gene_presence_absence.csv \
        --out results/ml
"""

import argparse
import re
import sys
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Mode: Snakemake or CLI ────────────────────────────────────────────────────
try:
    feature_files  = snakemake.input.feature_files
    out_csv        = snakemake.output.annotated_csv
    out_plot       = snakemake.output.annotation_plot
    tbdb_bed       = snakemake.params.get("tbdb_bed",       None)
    pan_gene_table = snakemake.params.get("pan_gene_table", None)
    log_file       = snakemake.log[0]
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_file, "w")
    def msg(m): print(m); log.write(m+"\n"); log.flush()
except NameError:
    p = argparse.ArgumentParser()
    p.add_argument("--feature_dir",    required=True)
    p.add_argument("--input_types",    nargs="+",
                   default=["snp","pan","card","snp_card","snp_pan","pan_card","snp_pan_card"])
    p.add_argument("--drugs",          nargs="+",
                   default=["isoniazid","rifampicin","ethambutol","streptomycin"])
    p.add_argument("--model",          default="rf")
    p.add_argument("--tbdb_bed",       default=None)
    p.add_argument("--pan_gene_table", default=None)
    p.add_argument("--out",            default="results/ml")
    args = p.parse_args()
    fd   = Path(args.feature_dir)
    feature_files  = [str(fd / f"{it}_{d}_{args.model}_features.csv")
                      for it in args.input_types for d in args.drugs]
    out_csv        = str(Path(args.out) / "annotated_features.csv")
    out_plot       = str(Path(args.out) / "annotation_summary.png")
    tbdb_bed       = args.tbdb_bed
    pan_gene_table = args.pan_gene_table
    def msg(m): print(m, flush=True)

msg("=" * 60)
msg("  Annotating features → gene names")
msg("=" * 60)

# ── 1. Load all feature CSVs ──────────────────────────────────────────────────
all_feat = []
for f in feature_files:
    try:
        df = pd.read_csv(f)
        if not df.empty and "feature" in df.columns:
            all_feat.append(df)
    except Exception:
        pass

if not all_feat:
    msg("❌  No feature files found — exiting")
    pd.DataFrame().to_csv(out_csv, index=False)
    sys.exit(0)

feat_df = pd.concat(all_feat, ignore_index=True).drop_duplicates(subset=["feature"])
feat_df["gene_name"]         = ""
feat_df["annotation_source"] = "unannotated"
msg(f"   Total unique features: {len(feat_df)}")

# ── 2. Build position → gene map from tbdb.bed ────────────────────────────────
# tbdb.bed columns: Chromosome  start  end  locus(Rv)  gene_name  drugs
# gene_name is column index 4 (e.g. dnaA, rpoB, gyrA)
tbdb_pos_map = {}   # position (int) → gene name

if tbdb_bed and Path(tbdb_bed).exists():
    msg(f"\n📋 Loading tbdb.bed: {tbdb_bed}")
    try:
        bed = pd.read_csv(tbdb_bed, sep="\t", header=None, low_memory=False)
        # Columns: 0=chrom, 1=start, 2=end, 3=locus(Rv), 4=gene_name, 5=drugs
        for _, row in bed.iterrows():
            try:
                start     = int(row[1])
                end       = int(row[2])
                locus     = str(row[3]).strip()   # e.g. Rv0001
                gene_name = str(row[4]).strip()   # e.g. dnaA
                # Use gene_name if meaningful, else locus
                label = gene_name if (gene_name and gene_name not in (".", "-", "nan", locus)) \
                        else locus
                for pos in range(start, end + 1):
                    if pos not in tbdb_pos_map:
                        tbdb_pos_map[pos] = label
            except (ValueError, IndexError):
                continue
        msg(f"   tbdb.bed: {len(tbdb_pos_map):,} positions mapped")
        # Show examples
        sample_pos = list(tbdb_pos_map.items())[:3]
        for pos, gene in sample_pos:
            msg(f"   Example: pos {pos} → {gene}")
    except Exception as e:
        msg(f"   ⚠️  tbdb.bed error: {e}")
else:
    msg("\n⚠️  tbdb.bed not provided or not found — SNP features will keep position labels")

# ── 3. Build pangenome group → gene name map ──────────────────────────────────
# Panaroo gene_presence_absence.csv:
#   Gene  |  Non-unique Gene name  |  Annotation  |  sample1  |  sample2  ...
# "Non-unique Gene name" = short gene name if available (e.g. dnaA)
# "Annotation"           = functional description (e.g. "DNA replication protein")
pan_map = {}   # group_id (e.g. group_3240) → best gene name

if pan_gene_table and Path(pan_gene_table).exists():
    msg(f"\n📋 Loading Panaroo gene_presence_absence.csv: {pan_gene_table}")
    try:
        pan = pd.read_csv(pan_gene_table, low_memory=False)
        # Confirm expected columns
        msg(f"   Columns: {list(pan.columns[:6])}")

        for _, row in pan.iterrows():
            group     = str(row["Gene"]).strip()
            gene_name = str(row.get("Non-unique Gene name", "")).strip()
            annotation = str(row.get("Annotation", "")).strip()

            # Priority: Non-unique Gene name > Annotation > group ID
            if gene_name and gene_name.lower() not in ("nan", "", "hypothetical protein"):
                label = gene_name
            elif annotation and annotation.lower() not in ("nan", "", "hypothetical protein"):
                # Truncate long descriptions
                label = annotation[:50] if len(annotation) > 50 else annotation
            else:
                label = group   # fallback to group ID

            pan_map[group] = label

        msg(f"   Panaroo: {len(pan_map)} group → gene name entries")
        named = sum(1 for v in pan_map.values() if not v.startswith("group_"))
        msg(f"   Named (non-group): {named} / {len(pan_map)}")
        for k, v in list(pan_map.items())[:3]:
            msg(f"   Example: {k} → {v}")
    except Exception as e:
        msg(f"   ⚠️  Panaroo table error: {e}")
else:
    msg("\n⚠️  pan_gene_table not provided or not found — pangenome features will keep group IDs")

# ── 4. Annotate each feature ──────────────────────────────────────────────────
# Patterns:
#   SNP:      Chromosome_761155  or  Chromosome_761155_C_G
#   Pangenome: group_3240
#   CARD:     CARD_rpoB2

SNP_PATTERN = re.compile(r"^(.+?)_(\d+)(?:_[A-Za-z]+_[A-Za-z]+)?$")
PAN_PATTERN = re.compile(r"^group_\d+$", re.IGNORECASE)

sources = []

for idx, row in feat_df.iterrows():
    feat = str(row["feature"])

    # ── CARD — already named ──────────────────────────────────────────────────
    if feat.startswith("CARD_"):
        feat_df.at[idx, "gene_name"]         = feat[5:]   # strip CARD_ prefix
        feat_df.at[idx, "annotation_source"] = "CARD"
        sources.append("CARD")
        continue

    # ── Country one-hot ───────────────────────────────────────────────────────
    if feat.startswith("country_"):
        feat_df.at[idx, "gene_name"]         = feat
        feat_df.at[idx, "annotation_source"] = "metadata"
        sources.append("metadata")
        continue

    # ── Pangenome group ───────────────────────────────────────────────────────
    if PAN_PATTERN.match(feat) or feat in pan_map:
        label = pan_map.get(feat, "")
        if label and label != feat:
            feat_df.at[idx, "gene_name"]         = label
            feat_df.at[idx, "annotation_source"] = "panaroo"
            sources.append("panaroo")
        else:
            feat_df.at[idx, "gene_name"]         = feat
            feat_df.at[idx, "annotation_source"] = "pan_no_name"
            sources.append("pan_no_name")
        continue

    # ── SNP feature: extract position, look up in tbdb.bed ───────────────────
    snp_m = SNP_PATTERN.match(feat)
    if snp_m:
        pos = int(snp_m.group(2))
        if pos in tbdb_pos_map:
            gene = tbdb_pos_map[pos]
            feat_df.at[idx, "gene_name"]         = gene
            feat_df.at[idx, "annotation_source"] = "tbdb_bed"
            sources.append("tbdb_bed")
        else:
            # Position not in any known gene region
            feat_df.at[idx, "gene_name"]         = feat
            feat_df.at[idx, "annotation_source"] = "position_only"
            sources.append("position_only")
        continue

    # ── Fallback ──────────────────────────────────────────────────────────────
    feat_df.at[idx, "gene_name"]         = feat
    feat_df.at[idx, "annotation_source"] = "unannotated"
    sources.append("unannotated")

feat_df["annotation_source"] = sources

# ── 5. Summary ────────────────────────────────────────────────────────────────
source_counts = feat_df["annotation_source"].value_counts()
n_annotated   = len(feat_df[~feat_df["annotation_source"].isin(
    ["position_only", "unannotated", "pan_no_name"]
)])
msg(f"\n✅ Annotated {n_annotated}/{len(feat_df)} features")
msg("\n📊 By source:")
for src, cnt in source_counts.items():
    msg(f"   {src:20s}: {cnt}")

feat_df.to_csv(out_csv, index=False)
msg(f"\n💾 Saved → {out_csv}")

# ── 6. Summary plot ───────────────────────────────────────────────────────────
SOURCE_COLORS = {
    "tbdb_bed":      "#e74c3c",
    "CARD":          "#2ecc71",
    "panaroo":       "#9b59b6",
    "position_only": "#f39c12",
    "pan_no_name":   "#d7bde2",
    "metadata":      "#95a5a6",
    "unannotated":   "#bdc3c7",
}
colors = [SOURCE_COLORS.get(s, "#7f8c8d") for s in source_counts.index]

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

axes[0].bar(range(len(source_counts)), source_counts.values,
            color=colors, alpha=0.85, edgecolor="white")
axes[0].set_xticks(range(len(source_counts)))
axes[0].set_xticklabels(source_counts.index, rotation=30, ha="right", fontsize=9)
axes[0].set_ylabel("Feature count")
axes[0].set_title("Annotation Sources", fontweight="bold")
for i, cnt in enumerate(source_counts.values):
    axes[0].text(i, cnt + 0.3, str(cnt), ha="center", va="bottom", fontsize=9)
axes[0].grid(True, axis="y", alpha=0.3)

axes[1].pie(source_counts.values, labels=source_counts.index,
            colors=colors, autopct="%1.1f%%", startangle=140,
            wedgeprops={"edgecolor": "white", "linewidth": 1.2})
axes[1].set_title(
    f"Annotation Coverage\n({n_annotated}/{len(feat_df)} features annotated)",
    fontweight="bold"
)

fig.suptitle("Feature Annotation Summary", fontweight="bold", fontsize=13)
plt.tight_layout()
Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_plot, dpi=180, bbox_inches="tight")
plt.close()
msg(f"✅ Annotation summary plot → {out_plot}")
try: log.close()
except: pass