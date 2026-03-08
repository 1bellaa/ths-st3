"""
Build binary SNP matrix from all per-sample VCF files.
Snakemake script — no subprocess calls.
"""

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

# CREATE LOG DIRECTORY
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
log = open(log_file, "w")

# HELPER FUNCTION 
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

# COLLECT SNP SITES 
all_sites    = set()
sample_sites = {}
sample_stats = {}

for vcf_file in tqdm(vcf_files, desc="Reading VCFs"):
    sample_name = vcf_file.stem.replace(".targets", "").replace(".vcf", "")
    try:
        reader = VCF(str(vcf_file))
        sites  = []
        n_variants = 0
        for rec in reader:
            n_variants += 1
            if len(rec.REF) == 1:
                for alt in rec.ALT:
                    if alt and len(alt) == 1:
                        site = (rec.CHROM, rec.POS, rec.REF, str(alt))
                        sites.append(site)
                        all_sites.add(site)
        sample_sites[sample_name] = sites
        sample_stats[sample_name] = {
            "total_variants": n_variants,
            "snps": len(sites),
            "vcf_size_mb": vcf_file.stat().st_size / 1024 / 1024,
        }
    except Exception as e:
        msg(f"⚠️  Error reading {sample_name}: {e}")
        sample_sites[sample_name] = []
        sample_stats[sample_name] = {"error": str(e)}

msg(f"✅ {len(all_sites)} unique SNP sites")

# BUILD SNP MATRIX 
all_sites   = sorted(all_sites)
matrix_data = []
sample_names = []

for name, sites in tqdm(sample_sites.items(), desc="Building matrix"):
    sites_set = set(sites)
    matrix_data.append([1 if s in sites_set else 0 for s in all_sites])
    sample_names.append(name)

columns = [f"{c}_{p}_{r}_{a}" for c, p, r, a in all_sites]
df = pd.DataFrame(matrix_data, columns=columns, index=sample_names)
df.index.name = "sample"

msg(f"📊 Matrix: {df.shape[0]} samples × {df.shape[1]} sites")
df.to_csv(output_matrix)
msg("✅ Matrix saved")

# STATISTICS
chrom_counts = defaultdict(int)
for c, p, r, a in all_sites:
    chrom_counts[c] += 1

lines = [
    "=" * 60, "SNP MATRIX STATISTICS", "=" * 60,
    f"Total samples:    {len(sample_names)}",
    f"Total SNP sites:  {len(all_sites)}",
    f"Matrix shape:     {df.shape[0]} × {df.shape[1]}",
    f"Sparsity:         {(df == 0).sum().sum() / df.size * 100:.2f}%",
    "",
    "SNPs per sample:",
    f"  Mean:   {df.sum(axis=1).mean():.0f}",
    f"  Median: {df.sum(axis=1).median():.0f}",
    f"  Min:    {df.sum(axis=1).min():.0f}",
    f"  Max:    {df.sum(axis=1).max():.0f}",
    "",
    "=" * 60, "PER-SAMPLE STATISTICS", "=" * 60,
    f"{'Sample':<25} {'SNPs':>8} {'Variants':>10} {'VCF MB':>8}",
    "-" * 60,
]
for sname, stats in sorted(sample_stats.items()):
    if "error" in stats:
        lines.append(f"{sname:<25}  ERROR: {stats['error']}")
    else:
        lines.append(
            f"{sname:<25} {stats['snps']:>8} {stats['total_variants']:>10} "
            f"{stats['vcf_size_mb']:>7.1f}"
        )
lines += ["", "SNP Distribution by Chromosome:"]
for chrom, cnt in sorted(chrom_counts.items()):
    lines.append(f"  {chrom}: {cnt}")

with open(output_stats, "w") as f:
    f.write("\n".join(lines))
for l in lines:
    msg(l)

msg("🎉 SNP matrix complete")
log.close()
