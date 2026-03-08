"""
Convert Panaroo gene_presence_absence.csv to a binary sample×gene matrix.
Snakemake script — no subprocess calls.
"""

from pathlib import Path
import pandas as pd
import sys

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

# tama ba to pacheck na lang pls nakalimutan ko na tysm
# Panaroo gene_presence_absence.csv layout:
#   Gene, Non-unique Gene name, Annotation, sample1, sample2, ...
#   Present = gene ID string | Absent = ""
df = pd.read_csv(pan_csv, index_col=0, low_memory=False)

# DROP META COLUMNS IF PRESENT (e.g. "Non-unique Gene name", "Annotation")
meta_cols = ["Non-unique Gene name", "Annotation"]
df = df.drop(columns=[c for c in meta_cols if c in df.columns], errors="ignore")

# CONVERT TO BINARY: non-empty string → 1, empty → 0 
binary = df.notna() & (df != "")
binary = binary.astype(int)

# TRANSPOSE: genes are columns, samples are rows
binary = binary.T
binary.index.name = "sample"

msg(f"📊 Pangenome matrix: {binary.shape[0]} samples × {binary.shape[1]} genes")
binary.to_csv(output_matrix)
msg("✅ Pangenome matrix saved")
log.close()
