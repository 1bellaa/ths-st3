"""
merge_metadata.py — Attach resistance phenotype labels to feature matrices
and produce summary visualisations.

Two-file strategy:
  metadata.xlsx    → isolate name, country, ena_sample, ena_experiment, ena_run
                     (used for downloading; the Snakefile also builds
                      sample_isolate_map.tsv from this file)

  master_data.xlsx → per-drug resistance labels
                     Sheets: ISONIAZID, RIFAMPICIN, ETHAMBUTOL, STREPTOMYCIN …
                     Each sheet: columns "isolate name" + "resistance phenotype (pDST)"
                     Values: R → 1, S → 0

Join strategy (isolate name as anchor):
  accession (matrix row) → isolate name (via sample_isolate_map.tsv)
                         → resistance labels (via master_data.xlsx drug sheets)
                         → country / ena_* cols (via metadata.xlsx)

Memory-efficient changes vs original:
  1. Feature matrices (SNP, pan, CARD) are loaded with dtype=np.int8.
     Why: all values are 0/1 binary. int8 = 1 byte vs default int64 = 8 bytes.
     For a 2484 × 50000 SNP matrix: int8 → 124 MB, int64 → 992 MB.
     Labels columns (NaN-capable floats) are handled separately and are tiny.
 
  2. Each joined output DataFrame is deleted immediately after writing with `del`.
     Why: without del, Python's GC may not free the joined copy before the next
     join is built, causing peak RAM to hold multiple matrix copies at once.
 
  3. `combine()` uses concat on int8 matrices, which stays compact.                         

Outputs:
  input_snp.csv       SNP features + labels + country
  input_pan.csv       Pangenome features + labels + country (empty if skipped)
  input_snp_pan.csv   Combined features + labels + country (empty if skipped)
  input_card.csv      CARD features + labels (empty if skipped)
  input_snp_card.csv  SNP + CARD features + labels (empty if skipped)
  input_pan_card.csv  Pangenome + CARD features + labels (empty if skipped)
  input_snp_pan_card.csv  SNP + Pangenome + CARD features + labels (empty if skipped)

  country_distribution.png   — pie chart of sample countries
  resistance_summary.png     — table image of R/S/no_label counts per drug
  country_distribution.png   — pie chart of sample countries
  resistance_summary.png     — table image of R/S/no_label counts per drug
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
 
# Snakemake bindings 
snp_matrix_path    = snakemake.input.snp_matrix
pan_matrix_path    = snakemake.input.pan_matrix
card_matrix_path   = snakemake.input.card_matrix
metadata_file      = snakemake.input.metadata
master_data_file   = snakemake.input.master_data
sample_isolate_map = snakemake.input.sample_isolate_map
out_snp            = snakemake.output.snp
out_pan            = snakemake.output.pan
out_snp_pan        = snakemake.output.snp_pan
out_card           = snakemake.output.card
out_snp_card       = snakemake.output.snp_card
out_pan_card       = snakemake.output.pan_card
out_snp_pan_card   = snakemake.output.snp_pan_card
out_country_plot   = snakemake.output.country_plot
out_resistance_plot= snakemake.output.resistance_plot
drugs              = snakemake.params.drugs
skip_pangenome     = snakemake.params.skip_pangenome
skip_card          = snakemake.params.skip_card
log_file           = snakemake.log[0]
 
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
log = open(log_file, "w")
 
def msg(m):
    print(m, flush=True)
    log.write(m + "\n")
    log.flush()
 
msg("=" * 60)
msg("  Merging resistance labels into feature matrices")
msg(f"   metadata     : {metadata_file}")
msg(f"   master_data  : {master_data_file}")
msg(f"   skip_pangenome = {skip_pangenome}")
msg("=" * 60)
 
# Load sample → isolate name map 
msg("\n  Loading sample → isolate name map...")
map_df = pd.read_csv(sample_isolate_map, sep="\t", dtype=str)
map_df["accession"]    = map_df["accession"].str.strip().str.upper()
map_df["isolate_name"] = map_df["isolate_name"].str.strip()
acc_to_isolate = dict(zip(map_df["accession"], map_df["isolate_name"]))
msg(f"   {len(acc_to_isolate)} accession → isolate name entries loaded")
 
# Load feature matrices 
msg("\n📂 Loading SNP matrix (int8)...")
snp_df = pd.read_csv(snp_matrix_path, index_col=0, dtype=np.int8)
snp_df.index = snp_df.index.astype(str).str.strip().str.upper()
msg(f"   {snp_df.shape[0]} samples × {snp_df.shape[1]} SNP features  "
    f"[{snp_df.values.nbytes / 1e6:.1f} MB]")
 
if not skip_pangenome:
    msg("\n📂 Loading Pangenome matrix (int8)...")
    pan_df = pd.read_csv(pan_matrix_path, index_col=0, dtype=np.int8)
    pan_df.index = pan_df.index.astype(str).str.strip().str.upper()
    msg(f"   {pan_df.shape[0]} samples × {pan_df.shape[1]} gene features  "
        f"[{pan_df.values.nbytes / 1e6:.1f} MB]")
 
matrix_ids = list(snp_df.index)
 
# Load metadata.xlsx 
msg("\n Loading metadata.xlsx...")
meta_xl  = pd.ExcelFile(metadata_file)
meta_raw = meta_xl.parse(meta_xl.sheet_names[0])
meta_raw.columns = [str(c).strip() for c in meta_raw.columns]
meta_raw["isolate name"] = meta_raw["isolate name"].astype(str).str.strip()
meta_raw = meta_raw.drop_duplicates(subset=["isolate name"]).set_index("isolate name")
meta_keep = [c for c in ["country","ena_sample","ena_experiment","ena_run"] if c in meta_raw.columns]
meta_base = meta_raw[meta_keep].copy()
msg(f"   {len(meta_base)} isolates  |  columns kept: {meta_keep}")
 
# Load master_data.xlsx → per-drug resistance labels 
msg("\n💊 Loading resistance labels from master_data.xlsx...")
master_xl = pd.ExcelFile(master_data_file)
msg(f"   Available sheets: {master_xl.sheet_names}")
RESISTANCE_COL = "resistance phenotype (pDST)"
 
for drug in drugs:
    drug_upper = drug.upper()
    colname    = f"{drug.lower()}_resistance"
    if drug_upper not in master_xl.sheet_names:
        msg(f"   ⚠️  Sheet '{drug_upper}' not found — skipping")
        continue
    drug_df = master_xl.parse(drug_upper, header=0)
    drug_df.columns = [str(c).strip() for c in drug_df.columns]
    if "isolate name" not in drug_df.columns or RESISTANCE_COL not in drug_df.columns:
        msg(f"   ⚠️  Sheet '{drug_upper}' missing required columns — skipping")
        continue
    drug_df = drug_df[["isolate name", RESISTANCE_COL]].copy()
    drug_df["isolate name"] = drug_df["isolate name"].astype(str).str.strip()
    drug_df = drug_df[drug_df["isolate name"].str.strip() != "nan"]
    drug_df[colname] = drug_df[RESISTANCE_COL].map({"R": 1, "S": 0})
    drug_df = drug_df.drop_duplicates(subset=["isolate name"]).set_index("isolate name")[[colname]]
    meta_base = meta_base.join(drug_df, how="left")
    r     = int((drug_df[colname] == 1).sum())
    s     = int((drug_df[colname] == 0).sum())
    total = len(drug_df.dropna(subset=[colname]))
    msg(f"   ✓ {drug_upper:15s}  R={r:4d}  S={s:4d}  total={total}")
 
# Map accessions → labels 
msg("\n🔗 Mapping matrix accessions → isolate names → labels...")
matched   = [s for s in matrix_ids if s in acc_to_isolate]
unmatched = [s for s in matrix_ids if s not in acc_to_isolate]
msg(f"   Matched   : {len(matched)}/{len(matrix_ids)}")
if unmatched:
    msg(f"   Unmatched : {unmatched}")
 
label_rows = []
for acc in matrix_ids:
    isolate = acc_to_isolate.get(acc)
    if isolate and isolate in meta_base.index:
        row = meta_base.loc[isolate].copy()
    else:
        row = pd.Series({col: None for col in meta_base.columns})
    row.name = acc
    label_rows.append(row)
 
labels_df = pd.DataFrame(label_rows)
labels_df.index.name = "sample"
 
# Log resistance distribution 
msg("\n📊 Resistance distribution (full dataset):")
resist_rows = []
for drug in drugs:
    colname = f"{drug.lower()}_resistance"
    if colname in labels_df.columns:
        r       = int((labels_df[colname] == 1).sum())
        s       = int((labels_df[colname] == 0).sum())
        missing = int(labels_df[colname].isna().sum())
        total   = r + s + missing
        msg(f"   {drug.upper():15s}  R={r:3d}  S={s:3d}  no_label={missing}   total={total}")
        resist_rows.append({"Drug": drug.upper(), "Resistant (R)": r,
                             "Susceptible (S)": s, "No Label": missing, "Samples": total})
 
if "country" in labels_df.columns:
    msg("\n🌍 Country distribution:")
    for country, n in labels_df["country"].value_counts().items():
        msg(f"   {country}: {n}")
 
# Pie chart — country distribution 
Path(out_country_plot).parent.mkdir(parents=True, exist_ok=True)
 
if "country" in labels_df.columns and labels_df["country"].notna().any():
    country_counts = labels_df["country"].value_counts()
    # Group into Top 10 + Others
    if len(country_counts) > 10:
        top10  = country_counts.iloc[:10]
        others = pd.Series({"Others": country_counts.iloc[10:].sum()})
        country_counts = pd.concat([top10, others])
    n_slices = len(country_counts)
    cmap     = plt.cm.RdYlBu_r
    colors   = [cmap(i / max(n_slices - 1, 1)) for i in range(n_slices)]
    if "Others" in country_counts.index:
        colors[-1] = "#cccccc"
    fig, ax = plt.subplots(figsize=(8, 7))
    wedges, texts, autotexts = ax.pie(
        country_counts.values, 
        labels=country_counts.index,
        autopct="%1.1f%%", 
        startangle=140, 
        pctdistance=0.82,
        colors=colors, 
        wedgeprops={"edgecolor": "white", "linewidth": 1.2},
    )
    for t in autotexts:
        t.set_fontsize(9)
    ax.set_title(
        f"Country Distribution  (Top 10 + Others,  n={len(labels_df)})",
        fontweight="bold", fontsize=13, pad=16,
    )
    plt.tight_layout()
    plt.savefig(out_country_plot, dpi=180, bbox_inches="tight")
    plt.close()
    msg(f"\n✅ Country pie chart → {out_country_plot}")
else:
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.text(0.5, 0.5, "No country data available",
            ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    plt.savefig(out_country_plot, dpi=120, bbox_inches="tight")
    plt.close()
    msg(f"\n⚠️  No country data — empty plot saved → {out_country_plot}")
 
# Table image — resistance distribution 
if resist_rows:
    resist_df = pd.DataFrame(resist_rows)
    fig, ax = plt.subplots(figsize=(7, max(2.5, len(resist_df) * 0.55 + 1.2)))
    ax.axis("off")
    tbl = ax.table(
        cellText=resist_df.values,
        colLabels=resist_df.columns,
        cellLoc="center",
        loc="center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1.2, 1.8)
    # Header row styling
    for j in range(len(resist_df.columns)):
        tbl[0, j].set_facecolor("#2c3e50")
        tbl[0, j].set_text_props(color="white", fontweight="bold")
    # Data rows — colour code R/S columns
    for i in range(1, len(resist_df) + 1):
        tbl[i, 0].set_facecolor("#ecf0f1")   # drug name
        tbl[i, 1].set_facecolor("#fadbd8")   # R — light red
        tbl[i, 2].set_facecolor("#d5f5e3")   # S — light green
        tbl[i, 3].set_facecolor("#fef9e7")   # no label — light yellow
    ax.set_title("Resistance Phenotype Distribution",
                 fontweight="bold", fontsize=13, pad=12)
    plt.tight_layout()
    plt.savefig(out_resistance_plot, dpi=180, bbox_inches="tight")
    plt.close()
    msg(f"✅ Resistance table image → {out_resistance_plot}")
else:
    fig, ax = plt.subplots(figsize=(4, 2))
    ax.text(0.5, 0.5, "No resistance data available",
            ha="center", va="center", transform=ax.transAxes)
    ax.axis("off")
    plt.savefig(out_resistance_plot, dpi=120, bbox_inches="tight")
    plt.close()
 
# Load CARD binary matrix 
if not skip_card:
    msg(f"\n📋 Loading CARD binary matrix (int8): {card_matrix_path}")
    card_df = pd.read_csv(card_matrix_path, index_col=0, dtype=np.int8)
    card_df.index = card_df.index.astype(str).str.strip()
    card_df.columns = [f"CARD_{c}" for c in card_df.columns]
    msg(f"   CARD matrix: {card_df.shape[0]} samples × {card_df.shape[1]} genes  "
        f"[{card_df.values.nbytes / 1e6:.1f} MB]")
else:
    card_df = None
    msg("\n⏭️  CARD skipped")
 
# Build and write output CSVs 
def build_and_write(matrix_df, label, outpath):
    """Join features + labels and write immediately; delete the join result."""
    merged = matrix_df.join(labels_df, how="left")
    msg(f"   {label}: {merged.shape[0]} rows × {merged.shape[1]} cols  "
        f"[{merged.memory_usage(deep=False).sum() / 1e6:.1f} MB]")
    merged.to_csv(outpath)
    del merged                      # ← free memory before next join
 
def combine(*dfs):
    """Inner-join feature matrices on common samples."""
    common = dfs[0].index
    for df in dfs[1:]:
        common = common.intersection(df.index)
    if len(common) == 0:
        msg("   ⚠️  No common samples across matrices")
        return pd.DataFrame()
    return pd.concat([df.loc[common] for df in dfs], axis=1)
 
Path(out_snp).parent.mkdir(parents=True, exist_ok=True)
 
msg("\n💾 Writing input_snp.csv...")
build_and_write(snp_df, "SNP", out_snp)
 
if skip_pangenome:
    msg("\n⏭️  Pangenome skipped — writing empty placeholders")
    pd.DataFrame().to_csv(out_pan)
    pd.DataFrame().to_csv(out_snp_pan)
else:
    msg("\n💾 Writing input_pan.csv...")
    build_and_write(pan_df, "Pangenome", out_pan)
    msg("\n💾 Writing input_snp_pan.csv...")
    snp_pan = combine(snp_df, pan_df)
    build_and_write(snp_pan, "SNP+Pan", out_snp_pan)
    del snp_pan
 
if skip_card:
    msg("\n⏭️  CARD skipped — writing empty placeholders")
    pd.DataFrame().to_csv(out_card)
    pd.DataFrame().to_csv(out_snp_card)
    pd.DataFrame().to_csv(out_pan_card)
    pd.DataFrame().to_csv(out_snp_pan_card)
else:
    msg("\n💾 Writing input_card.csv...")
    build_and_write(card_df, "CARD", out_card)
 
    msg("\n💾 Writing input_snp_card.csv...")
    snp_card = combine(snp_df, card_df)
    build_and_write(snp_card, "SNP+CARD", out_snp_card)
    del snp_card
 
    if skip_pangenome:
        msg("\n⏭️  pan_card / snp_pan_card skipped (no pangenome)")
        pd.DataFrame().to_csv(out_pan_card)
        pd.DataFrame().to_csv(out_snp_pan_card)
    else:
        msg("\n💾 Writing input_pan_card.csv...")
        pan_card = combine(pan_df, card_df)
        build_and_write(pan_card, "Pan+CARD", out_pan_card)
        del pan_card
 
        msg("\n💾 Writing input_snp_pan_card.csv...")
        snp_pan_card = combine(snp_df, pan_df, card_df)
        build_and_write(snp_pan_card, "SNP+Pan+CARD", out_snp_pan_card)
        del snp_pan_card
 
msg("\n🎉 Metadata merge complete")
log.close()