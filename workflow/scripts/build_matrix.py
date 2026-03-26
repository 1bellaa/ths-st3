"""
Build binary SNP matrix from all per-sample VCF files.
Snakemake script — no subprocess calls.

Memory-efficient two-pass approach:
  Pass 1: Read all VCFs once to collect the set of all unique SNP sites.
          Only store a (sample_name, vcf_path) list — NOT per-sample site data.
  Pass 2: Re-read each VCF and write one CSV row at a time directly to disk.
          Peak RAM = one row (n_sites integers) at a time, not the full matrix.

Why the original was OOM:
  sample_sites dict  → 2484 samples × ~2000 site-tuples × ~200 bytes = ~1 GB
  matrix_data list   → 2484 × n_sites Python ints                    = ~1–2 GB
  pd.DataFrame(...)  → another full copy of the matrix               = ~1–2 GB
  Total peak: 3–5 GB just to build and write the matrix.
"""

import csv
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd
from cyvcf2 import VCF
from tqdm import tqdm

# SNAKEMAKE BINDINGS
vcf_dir       = Path(snakemake.params.vcf_dir)
output_matrix = snakemake.output.matrix
output_stats  = snakemake.output.stats
log_file      = snakemake.log[0]

Path(log_file).parent.mkdir(parents=True, exist_ok=True)
log = open(log_file, "w")

def msg(m):
    print(m)
    log.write(m + "\n")
    log.flush()

msg("🧬 Building SNP matrix...")

# FIND VCF FILES
vcf_files = list(vcf_dir.glob("*.targets.vcf.gz")) or list(vcf_dir.glob("*.vcf.gz"))
if not vcf_files:
    msg("❌ No VCF files found")
    log.close()
    sys.exit(1)
msg(f"📂 Found {len(vcf_files)} VCF files")

# Collect all unique sites + per-sample stats (no per-sample site storage) 
all_sites    = set()    # unique (CHROM, POS, REF, ALT) tuples — small (~10K entries)
sample_vcfs  = []       # ordered list of (sample_name, vcf_path) — lightweight
sample_stats = {}       # per-sample stats for the stats file

for vcf_file in tqdm(vcf_files, desc="Collecting unique sites..."):
    sample_name = vcf_file.stem.replace(".targets", "").replace(".vcf", "")
    sample_vcfs.append((sample_name, vcf_file))
    try:
        reader     = VCF(str(vcf_file))
        n_snps     = 0
        n_variants = 0
        for rec in reader:
            n_variants += 1
            if len(rec.REF) == 1:
                for alt in rec.ALT:
                    if alt and len(alt) == 1:
                        site = (rec.CHROM, rec.POS, rec.REF, str(alt))
                        all_sites.add(site)
                        n_snps += 1
        sample_stats[sample_name] = {
            "total_variants": n_variants,
            "snps":           n_snps,
            "vcf_size_mb":    vcf_file.stat().st_size / 1024 / 1024,
        }
    except Exception as e:
        msg(f"⚠️  Error reading {sample_name}: {e}")
        sample_vcfs[-1]             = (sample_name, vcf_file)   # keep entry for pass 2
        sample_stats[sample_name]   = {"error": str(e)}

msg(f"✅ {len(all_sites)} unique SNP sites across {len(sample_vcfs)} samples")

# Build fast lookup: site-tuple → column index
all_sites_sorted = sorted(all_sites)
site_to_idx      = {s: i for i, s in enumerate(all_sites_sorted)}
columns          = [f"{c}_{p}_{r}_{a}" for c, p, r, a in all_sites_sorted]
n_sites          = len(all_sites_sorted)

# Re-read each VCF and write one CSV row immediately 
# Write each sample's row directly to the CSV file. 
msg(f"\n📐 Writing matrix ({len(sample_vcfs)} × {n_sites}) row-by-row...")
Path(output_matrix).parent.mkdir(parents=True, exist_ok=True)

with open(output_matrix, "w", newline="") as fh:
    writer = csv.writer(fh)
    writer.writerow(["sample"] + columns)

    for sample_name, vcf_file in tqdm(sample_vcfs, desc="Pass 2/2  writing rows"):
        if "error" in sample_stats.get(sample_name, {}):
            # Write a zero row for failed samples
            writer.writerow([sample_name] + [0] * n_sites)
            continue
        try:
            reader  = VCF(str(vcf_file))
            row_vec = [0] * n_sites          # one row — ~n_sites × 28 bytes ≈ 80 KB
            for rec in reader:
                if len(rec.REF) == 1:
                    for alt in rec.ALT:
                        if alt and len(alt) == 1:
                            site = (rec.CHROM, rec.POS, rec.REF, str(alt))
                            idx  = site_to_idx.get(site)
                            if idx is not None:
                                row_vec[idx] = 1
            writer.writerow([sample_name] + row_vec)
        except Exception as e:
            msg(f"⚠️  Error in pass 2 for {sample_name}: {e}")
            writer.writerow([sample_name] + [0] * n_sites)

msg("✅ Matrix saved")

# STATISTICS (computed from sample_stats — no DataFrame needed)
n_samples  = len(sample_vcfs)
snp_counts = [
    sample_stats[s]["snps"]
    for s, _ in sample_vcfs
    if "snps" in sample_stats.get(s, {})
]

chrom_counts = defaultdict(int)
for c, p, r, a in all_sites_sorted:
    chrom_counts[c] += 1

mean_snps   = sum(snp_counts) / len(snp_counts) if snp_counts else 0
sorted_snps = sorted(snp_counts)
median_snps = sorted_snps[len(sorted_snps) // 2] if sorted_snps else 0
# Sparsity estimate: fraction of zeros = 1 - (total SNPs in all samples / total cells)
total_ones  = sum(snp_counts)
sparsity    = (1 - total_ones / (n_samples * n_sites) * 100) if n_sites and n_samples else 0

msg(f"📊 Matrix: {n_samples} samples × {n_sites} sites  (sparsity ≈ {sparsity:.1f}%)")

lines = [
    "=" * 60, "SNP MATRIX STATISTICS", "=" * 60,
    f"Total samples:    {n_samples}",
    f"Total SNP sites:  {n_sites}",
    f"Matrix shape:     {n_samples} × {n_sites}",
    f"Sparsity (est.):  {sparsity:.2f}%",
    "",
    "SNPs per sample (from variant calling):",
    f"  Mean:   {mean_snps:.0f}",
    f"  Median: {median_snps:.0f}",
    f"  Min:    {min(snp_counts, default=0)}",
    f"  Max:    {max(snp_counts, default=0)}",
    "",
    "=" * 60, "PER-SAMPLE STATISTICS", "=" * 60,
    f"{'Sample':<25} {'SNPs':>8} {'Variants':>10} {'VCF MB':>8}",
    "-" * 60,
]
for sname, _ in sample_vcfs:
    stats = sample_stats.get(sname, {})
    if "error" in stats:
        lines.append(f"{sname:<25}  ERROR: {stats['error']}")
    else:
        lines.append(
            f"{sname:<25} {stats.get('snps',0):>8} "
            f"{stats.get('total_variants',0):>10} "
            f"{stats.get('vcf_size_mb',0):>7.1f}"
        )
lines += ["", "SNP Distribution by Chromosome:"]
for chrom, cnt in sorted(chrom_counts.items()):
    lines.append(f"  {chrom}: {cnt}")

with open(output_stats, "w") as f:
    f.write("\n".join(lines))
for line in lines:
    msg(line)

msg("🎉 SNP matrix complete")
log.close()