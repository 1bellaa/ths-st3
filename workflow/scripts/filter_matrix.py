"""
filter_matrix.py — Filter SNP or Pangenome binary matrix by allele/gene frequency
====================================================================================
Called as a Snakemake script for two rules:
  - filter_snp_matrix       (params.matrix_type = "snp")
  - filter_pangenome_matrix (params.matrix_type = "pangenome")

For SNP matrix:
  - Keeps sites where minor allele frequency >= maf_min (default 0.005)
  - Keeps sites where minor allele frequency <= maf_max (default 0.995)
  - This removes ultra-rare SNPs and near-fixed sites

For Pangenome matrix:
  - Keeps genes present in >= maf_min fraction of samples (default 0.01)
  - Keeps genes present in <= maf_max fraction of samples (default 0.99)
  - This removes very rare accessory genes and core genes present in almost all
  - Uses chunked reading to handle large pangenome matrices with low RAM usage

[CHANGED] vs original filter_matrix.py:
  - Was a standalone script with hardcoded filenames — now a proper Snakemake
    script that reads all paths/params from snakemake.* objects.
  - The three separate code sections (SNP filter, pangenome filter, merge) are
    now split: this script handles SNP + pangenome filtering only.
    The merge step is handled by merge_metadata.py (which already does it).
  - Fixed: original SNP section wrote to "snp_matrix_filter.csv" (typo) but
    printed "snp_matrix_filtered.csv" — now outputs to snakemake.output.filtered.
  - Fixed: original pangenome section read from hardcoded 'panaroo_matrix.csv'
    instead of the actual pipeline output path.
  - Added proper logging to snakemake.log[0].
"""

import sys
from pathlib import Path

import pandas as pd

# SNAKEMAKE BINDINGS
input_matrix  = snakemake.input.matrix
output_file   = snakemake.output.filtered
matrix_type   = snakemake.params.matrix_type   # "snp" or "pangenome"
maf_min       = snakemake.params.maf_min
maf_max       = snakemake.params.maf_max
log_file      = snakemake.log[0]

Path(log_file).parent.mkdir(parents=True, exist_ok=True)
log = open(log_file, "w")

def msg(m):
    print(m, flush=True)
    log.write(m + "\n")
    log.flush()

CHUNK_SIZE = 50   # rows per chunk for memory-efficient processing, can adjust based on available RAM

msg("=" * 60)
msg(f"🔬 Filtering {matrix_type} matrix")
msg(f"   Input   : {input_matrix}")
msg(f"   Output  : {output_file}")
msg(f"   maf_min : {maf_min}")
msg(f"   maf_max : {maf_max}")
msg("=" * 60)

# CALCULATE FREQUENCY OF EACH COLUMN (SNP OR GENE) 
msg("\n[1/2] Calculating column frequencies (chunked read)...")
col_sums      = None
total_samples = 0

for chunk in pd.read_csv(input_matrix, index_col=0, chunksize=CHUNK_SIZE):
    # Coerce everything to numeric; non-numeric → NaN → 0
    chunk_num = chunk.apply(pd.to_numeric, errors="coerce").fillna(0)

    if col_sums is None:
        col_sums = pd.Series(0.0, index=chunk_num.columns)

    col_sums = col_sums.add(chunk_num.sum(axis=0), fill_value=0)
    total_samples += len(chunk)
    print(f"   Processed {total_samples} samples...", end="\r", flush=True)

print()
msg(f"   ✓ {total_samples} samples analysed")

if total_samples == 0:
    msg("❌ No samples found in matrix — check input file")
    log.close()
    sys.exit(1)

freq = col_sums / total_samples
cols_to_keep = freq[(freq >= maf_min) & (freq <= maf_max)].index.tolist()

n_total  = len(freq)
n_rare   = int((freq < maf_min).sum())
n_fixed  = int((freq > maf_max).sum())
n_keep   = len(cols_to_keep)

msg(f"\n📊 Filtering statistics:")
msg(f"   Total features   : {n_total}")
msg(f"   Removed (rare)   : {n_rare}  (freq < {maf_min})")
msg(f"   Removed (fixed)  : {n_fixed}  (freq > {maf_max})")
msg(f"   Kept             : {n_keep}")

if n_keep == 0:
    msg("❌ No features passed the frequency thresholds — "
        "check maf_min / maf_max in config.yaml")
    log.close()
    sys.exit(1)

# WRITE FILTERED MATRIX 
msg("\n[2/2] Writing filtered matrix (chunked write)...")
Path(output_file).parent.mkdir(parents=True, exist_ok=True)
first = True

for i, chunk in enumerate(
    pd.read_csv(input_matrix, index_col=0, chunksize=CHUNK_SIZE)
):
    # Keep only the surviving columns; fill NaN with 0 and cast to int
    chunk_filtered = chunk[cols_to_keep].fillna(0).astype(int)
    chunk_filtered.to_csv(
        output_file,
        mode="w" if first else "a",
        header=first,
    )
    first = False
    print(f"   Written chunk {i + 1}...", end="\r", flush=True)

print()
msg(f"\n✅ Filtered {matrix_type} matrix saved → {output_file}")
msg(f"   Shape: {total_samples} samples × {n_keep} features")
log.close()
