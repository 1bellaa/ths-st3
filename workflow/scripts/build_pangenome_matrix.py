"""
Convert Panaroo gene_presence_absence.csv to a binary sample×gene matrix.
Snakemake script — no subprocess calls.
"""

import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# SNAKEMAKE BINDINGS
pan_csv       = snakemake.input.pan_csv
output_matrix = snakemake.output.matrix
log_file      = snakemake.log[0]

Path(log_file).parent.mkdir(parents=True, exist_ok=True)
log = open(log_file, "w")

def msg(m):
    print(m)
    log.write(m + "\n")
    log.flush()

msg("🧬 Building pangenome matrix from Panaroo output...")

META_COLS  = ["Non-unique Gene name", "Annotation"]
CHUNK_SIZE = 500   # genes per chunk 

# Get sample column names from the header row 
# To know n_samples before allocating the numpy matrix
header_df   = pd.read_csv(pan_csv, index_col=0, nrows=0, low_memory=False)
sample_cols = [c for c in header_df.columns if c not in META_COLS]
n_samples   = len(sample_cols)
msg(f"   Sample columns detected: {n_samples}")
msg(f"   Meta columns dropped   : {[c for c in META_COLS if c in header_df.columns]}")

if n_samples == 0:
    msg("❌ No sample columns found — check Panaroo output format")
    log.close()
    sys.exit(1)

# Count total genes by scanning only the index column (gene names) in chunks
gene_names = []
for chunk in pd.read_csv(pan_csv, index_col=0, chunksize=CHUNK_SIZE,
                          usecols=[0], low_memory=False):
    gene_names.extend(chunk.index.tolist())

n_genes = len(gene_names)
msg(f"   Total genes: {n_genes}")

# Allocate compact numpy matrix: samples × genes
binary_matrix = np.zeros((n_samples, n_genes), dtype=np.uint8)
sample_to_idx = {s: i for i, s in enumerate(sample_cols)}
msg(f"   Allocated numpy matrix: {binary_matrix.nbytes / 1e6:.1f} MB  "
    f"({n_samples} samples × {n_genes} genes × uint8)")

# Fill matrix chunk by chunk
gene_offset = 0
chunks_done = 0

for chunk in pd.read_csv(pan_csv, index_col=0, chunksize=CHUNK_SIZE,
                          low_memory=False):
    # Drop meta columns if present in this chunk
    chunk = chunk.drop(columns=[c for c in META_COLS if c in chunk.columns],
                       errors="ignore")

    # Reorder to canonical sample order; fill missing columns with NaN
    chunk = chunk.reindex(columns=sample_cols)

    # Binary: non-NaN AND non-empty-string → 1, otherwise → 0
    # .notna() catches NaN (absent gene); (chunk != "") catches explicit empty strings
    binary_chunk = (chunk.notna() & (chunk != "")).values.astype(np.uint8)
    # binary_chunk shape: (chunk_genes × n_samples) → transpose to (n_samples × chunk_genes)
    chunk_genes = binary_chunk.shape[0]
    binary_matrix[:, gene_offset:gene_offset + chunk_genes] = binary_chunk.T

    gene_offset += chunk_genes
    chunks_done += 1
    if chunks_done % 10 == 0:
        msg(f"   Processed {gene_offset}/{n_genes} genes...")

msg(f"   ✓ All {n_genes} genes processed")

# Write matrix row-by-row to avoid another large in-memory copy from DataFrame to CSV
msg(f"\n📝 Writing binary matrix → {output_matrix}")
Path(output_matrix).parent.mkdir(parents=True, exist_ok=True)

with open(output_matrix, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["sample"] + gene_names)
    for i, sample in enumerate(sample_cols):
        writer.writerow([sample] + binary_matrix[i].tolist())

msg(f"📊 Pangenome matrix: {n_samples} samples × {n_genes} genes")
msg(f"   Core genes (present in ≥95% samples): "
    f"{int((binary_matrix.mean(axis=0) >= 0.95).sum())}")
msg(f"   Accessory genes (present in <95%): "
    f"{int((binary_matrix.mean(axis=0) < 0.95).sum())}")
msg("✅ Pangenome matrix saved")
log.close()