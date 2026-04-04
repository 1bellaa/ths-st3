"""
preprocess.py — Trim + Align + Variant Call per sample
=======================================================
Runs: Trimmomatic PE → BWA-MEM → samtools sort/index → bcftools mpileup|call
Outputs a bgzipped + tabix-indexed VCF to snakemake.output.vcf/.tbi.
Cleans up trimmed FASTQs, SAM, and BAM internally after each step.

Called as a Snakemake script (snakemake.* objects available).
USES-SUBPROCESS: calls trimmomatic, bwa, samtools, bcftools, tabix via subprocess.
All tools must be available inside workflow/envs/align.yaml conda environment.
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
import gzip

# ── Snakemake bindings ────────────────────────────────────────────────────────
sample     = snakemake.params.sample
r1         = snakemake.input.r1
r2         = snakemake.input.r2
ref_genome = snakemake.input.ref
output_vcf = snakemake.output.vcf
output_tbi = snakemake.output.tbi
log_file   = snakemake.log[0]
threads    = snakemake.threads
tmp_dir    = Path(snakemake.params.tmp_dir) / sample

# ── Trimmomatic parameters ────────────────────────────────────────────────────
LEADING       = snakemake.config.get("trimmomatic_leading",       3)
TRAILING      = snakemake.config.get("trimmomatic_trailing",      3)
SLIDINGWINDOW = snakemake.config.get("trimmomatic_slidingwindow", "4:20")
MINLEN        = snakemake.config.get("trimmomatic_minlen",        25)
ADAPTER_CFG   = snakemake.config.get("trimmomatic_adapter",       "TruSeq3-PE-2.fa")

# ── Setup ─────────────────────────────────────────────────────────────────────
Path(log_file).parent.mkdir(parents=True, exist_ok=True)
Path(output_vcf).parent.mkdir(parents=True, exist_ok=True)
tmp_dir.mkdir(parents=True, exist_ok=True)

log = open(log_file, "w")

def log_msg(msg):
    print(msg, flush=True)
    log.write(msg + "\n")
    log.flush()

def run(cmd, description, timeout=5400):
    """Run a subprocess list command; exit(1) on failure."""
    log_msg(f"  ▶ {description}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if result.returncode != 0:
        log_msg(f"❌ {description} failed (exit {result.returncode})")
        if result.stderr:
            log_msg(f"   STDERR: {result.stderr[-1200:]}")
        log.close()
        sys.exit(1)
    return result

def run_shell(cmd, description, timeout=7200):
    """
    Run a shell pipeline command (uses shell=True so conda PATH is inherited).
    stderr is written directly to the log file, not captured, so it appears
    in the log even if the process crashes mid-pipe.
    """
    log_msg(f"  ▶ {description}")
    log_msg(f"    CMD: {cmd}")
    result = subprocess.run(
        cmd, shell=True,
        executable="/bin/bash",   # ensure bash so pipefail works
        stdout=None, stderr=None, # inherit — goes to terminal / snakemake log
        timeout=timeout
    )
    if result.returncode != 0:
        log_msg(f"❌ {description} failed (exit {result.returncode})")
        log_msg(f"   Check log: {log_file}")
        log.close()
        sys.exit(1)
    return result

log_msg("=" * 60)
log_msg(f"🔬 Preprocessing sample: {sample}")
log_msg(f"   Threads : {threads}")
log_msg(f"   Ref     : {ref_genome}")
log_msg(f"   Tmp dir : {tmp_dir}")
log_msg("=" * 60)

# ==================== VALIDATE INPUTS ====================
log_msg("📦 Validating input FASTQ files...")
for fq in [r1, r2]:
    try:
        with gzip.open(fq, "rb") as f:
            f.read(1000)
        size_mb = Path(fq).stat().st_size / (1024 * 1024)
        log_msg(f"   ✓ {Path(fq).name} ({size_mb:.1f} MB)")
    except Exception as e:
        log_msg(f"❌ FASTQ corrupted or missing: {fq}\n   Error: {e}")
        log.close()
        sys.exit(1)

# ==================== CHECK BWA INDEX ====================
log_msg("🔍 Checking BWA index files...")
for ext in [".amb", ".ann", ".bwt", ".pac", ".sa"]:
    idx = Path(str(ref_genome) + ext)
    if not idx.exists():
        log_msg(f"❌ BWA index file missing: {idx}")
        log_msg(f"   Fix: cd reference && bwa index {ref_genome}")
        log.close()
        sys.exit(1)
log_msg("   ✓ All BWA index files present")

# ==================== LOCATE ADAPTER FILE ====================
log_msg("🔍 Locating Trimmomatic adapter file...")
conda_prefix = os.environ.get("CONDA_PREFIX", "")
adapter_candidates = [
    ADAPTER_CFG,
    f"{conda_prefix}/share/trimmomatic/adapters/{ADAPTER_CFG}",
    f"{conda_prefix}/share/trimmomatic/{ADAPTER_CFG}",
    f"{Path.home()}/miniconda3/share/trimmomatic/adapters/{ADAPTER_CFG}",
    f"{Path.home()}/miniconda3/share/trimmomatic/adapters/TruSeq3-PE-2.fa",
    "/usr/share/trimmomatic/TruSeq3-PE-2.fa",
    "/opt/conda/share/trimmomatic/adapters/TruSeq3-PE-2.fa",
]
adapter_file = next((p for p in adapter_candidates if p and Path(p).exists()), None)
if adapter_file:
    log_msg(f"   ✓ Adapter file: {adapter_file}")
else:
    log_msg("   ⚠️  No adapter file found — adapter trimming will be skipped")

# ==================== STEP 1: TRIMMOMATIC ====================
log_msg("\n1️⃣  Quality trimming with Trimmomatic...")

trimmed_r1 = tmp_dir / f"{sample}_R1_paired.fastq.gz"
trimmed_r2 = tmp_dir / f"{sample}_R2_paired.fastq.gz"
unp_r1     = tmp_dir / f"{sample}_R1_unpaired.fastq.gz"
unp_r2     = tmp_dir / f"{sample}_R2_unpaired.fastq.gz"

trim_cmd = [
    "trimmomatic", "PE",
    "-threads", str(threads),
    "-phred33",
    str(r1), str(r2),
    str(trimmed_r1), str(unp_r1),
    str(trimmed_r2), str(unp_r2),
]
if adapter_file:
    trim_cmd += [f"ILLUMINACLIP:{adapter_file}:2:30:10"]
trim_cmd += [
    f"LEADING:{LEADING}",
    f"TRAILING:{TRAILING}",
    f"SLIDINGWINDOW:{SLIDINGWINDOW}",
    f"MINLEN:{MINLEN}",
]

run(trim_cmd, "Trimmomatic PE", timeout=3600)
log_msg(f"   ✓ R1 paired: {trimmed_r1.stat().st_size / 1e6:.1f} MB")
log_msg(f"   ✓ R2 paired: {trimmed_r2.stat().st_size / 1e6:.1f} MB")

for f in [unp_r1, unp_r2]:
    f.unlink(missing_ok=True)

# ==================== STEP 2: BWA-MEM ====================
log_msg("\n2️⃣  Aligning reads with BWA-MEM...")

sorted_bam = tmp_dir / f"{sample}.sorted.bam"
rg = f"@RG\\tID:{sample}\\tSM:{sample}\\tPL:ILLUMINA"

# shell=True so conda PATH is inherited; stderr appended to log file
bwa_cmd = (
    f"set -euo pipefail; "
    f"bwa mem -t {threads} -R '{rg}' {ref_genome} "
    f"{trimmed_r1} {trimmed_r2} 2>> {log_file} "
    f"| samtools sort -@ {threads} -o {sorted_bam} - 2>> {log_file}"
)
run_shell(bwa_cmd, "BWA-MEM | samtools sort", timeout=7200)

run(["samtools", "index", str(sorted_bam)], "samtools index BAM")
log_msg(f"   ✓ BAM: {sorted_bam.stat().st_size / 1e6:.0f} MB")

for f in [trimmed_r1, trimmed_r2]:
    f.unlink(missing_ok=True)
log_msg("   ✓ Trimmed FASTQs deleted")

# ==================== STEP 3: BCFTOOLS VARIANT CALLING ====================
log_msg("\n3️⃣  Calling variants with BCFtools...")

raw_vcf = tmp_dir / f"{sample}.vcf.gz"

vcf_cmd = (
    f"set -euo pipefail; "
    f"bcftools mpileup "
    f"-f {ref_genome} -Q 20 -q 30 "
    f"-a FORMAT/DP,FORMAT/AD -Ou {sorted_bam} 2>> {log_file} "
    f"| bcftools call -mv -Oz -o {raw_vcf} 2>> {log_file}"
)
run_shell(vcf_cmd, "bcftools mpileup | bcftools call", timeout=3600)

log_msg(f"   ✓ VCF: {raw_vcf.stat().st_size / 1024:.1f} KB")

sorted_bam.unlink(missing_ok=True)
Path(str(sorted_bam) + ".bai").unlink(missing_ok=True)
log_msg("   ✓ BAM/BAI deleted")

# ==================== STEP 4: TABIX INDEX ====================
log_msg("\n4️⃣  Indexing VCF with tabix...")
run(["tabix", "-p", "vcf", str(raw_vcf)], "tabix index VCF")

# ==================== STEP 5: MOVE TO OUTPUT ====================
log_msg("\n5️⃣  Moving outputs to final location...")
shutil.move(str(raw_vcf), str(output_vcf))
shutil.move(str(raw_vcf) + ".tbi", str(output_tbi))
log_msg(f"   ✓ VCF → {output_vcf}")
log_msg(f"   ✓ TBI → {output_tbi}")

# ==================== STEP 6: VERIFY ====================
log_msg("\n6️⃣  Verifying VCF integrity...")
try:
    with gzip.open(output_vcf, "rb") as f:
        lines = [f.readline().decode() for _ in range(60)]
    n_header   = sum(1 for l in lines if l.startswith("#"))
    n_variants = sum(1 for l in lines if l.strip() and not l.startswith("#"))
    log_msg(f"   ✓ Header lines : {n_header}")
    log_msg(f"   ✓ Variant lines: {n_variants}+ (first 60 lines)")
    if n_variants == 0:
        log_msg("   ⚠️  No variants in first 60 lines — check coverage/ref match")
except Exception as e:
    log_msg(f"   ⚠️  VCF verification warning: {e}")

# ==================== CLEANUP TMP ====================
for f in tmp_dir.glob("*"):
    try:
        f.unlink()
    except Exception:
        pass
try:
    tmp_dir.rmdir()
except Exception:
    pass
log_msg("   ✓ Temp directory cleaned")

log_msg("\n" + "=" * 60)
log_msg(f"🎉 SUCCESS: {sample}")
log_msg(f"   VCF size: {Path(output_vcf).stat().st_size / 1024:.1f} KB")
log_msg("=" * 60)
log.close()