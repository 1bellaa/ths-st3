"""
TB Drug Resistance Genomics Pipeline
=====================================
Workflow:
  Download → tbprofiler.py (Trim + BWA + BCFtools) → SNP Matrix
  → CARD AMR Screen → summarize_card.py
  → SPAdes → Bakta → Panaroo → Pangenome Matrix
  → filter_matrix.py (SNP + Pangenome)
  → merge_metadata.py → ML Models (RF + LR)

Sample identification strategy:
  The pipeline uses `isolate name` (from master_data.xlsx sheet "All") as the
  stable anchor. For every isolate, the best available run accession is picked
  (ena_run → ena_experiment → ena_sample, in that order) and used as the sample
  ID throughout the pipeline (file names, matrix row index, etc.).
  A mapping file (sample_isolate_map.tsv) is written at startup so
  merge_metadata.py can reverse-map any accession back to its isolate name,
  and from there join resistance labels — regardless of whether the downloaded
  FASTQ comes back as ERR, SRR, or any other accession prefix.

Storage management:
  - Raw FASTQs        → temp(), deleted after preprocess_sample writes VCF
  - Trimmed FASTQs / BAM inside tbprofiler.py → deleted by the script itself
  - CARD BAMs         → temp(), deleted after coverage file is written
  - Assembly FASTAs   → temp(), deleted after Bakta consumes them
  - Annotation GFFs   → temp(), deleted after Panaroo consumes them
  - VCFs              → kept until cleanup_vcfs rule runs after snp_matrix.csv
  - Assembly/annot dirs → deleted by cleanup_pangenome_intermediates

Skip flags (set in config.yaml or with --config on the command line):
  skip_assembly:   true  → skip SPAdes
  skip_annotation: true  → skip Bakta  (also forces skip_pangenome)
  skip_pangenome:  true  → skip Panaroo + pangenome matrix + pan ML inputs
  skip_card:       true  → skip CARD screening + card_summary
  skip_ml:         true  → skip all ML rules
"""

import pandas as pd
from pathlib import Path
import os

# CONFIGURATION 
configfile: "config.yaml"

# DIRECTORIES
FASTQ_DIR    = Path(config["fastq_dir"])
RESULTS_DIR  = Path(config["results_dir"])
VCF_DIR      = Path(config["vcf_dir"])
BAM_DIR      = Path(config["bam_dir"])
TRIM_DIR     = Path(config["trim_dir"])
ASSEMBLY_DIR = Path(config["assembly_dir"])
ANNOT_DIR    = Path(config["annot_dir"])
CARD_DIR     = Path(config["card_dir"])
ML_DIR       = Path(config["ml_dir"])
LOGS_DIR     = Path(config["logs_dir"])
TMP_DIR      = Path(config["tmp_dir"])

REF          = config["reference_genome"]
CARD_BT2     = config["card_bowtie2_index"]
CARD_LENGTHS = config["card_lengths"]
METADATA     = config["metadata"]
MASTERDATA   = config["master_data"]
DRUGS        = config["drugs"]

# SKIP FLAGS
SKIP_ASSEMBLY   = config.get("skip_assembly",   False)
SKIP_ANNOTATION = config.get("skip_annotation", False)
SKIP_PANGENOME  = config.get("skip_pangenome",  False)
SKIP_CARD       = config.get("skip_card",       False)
SKIP_ML         = config.get("skip_ml",         False)

if SKIP_ASSEMBLY or SKIP_ANNOTATION:
    SKIP_PANGENOME = True

ML_INPUT_TYPES = ["snp"] if SKIP_PANGENOME else ["snp", "pan", "snp_pan"]

# LOAD SAMPLES 
# Strategy:
#   1. Read sheet "All" from metadata.xlsx (try header=1 then header=0).
#   2. For each row, pick the best run accession in order:
#        ena_run → ena_experiment → ena_sample
#      This becomes the SAMPLE ID used for all file names.
#   3. Build ISOLATE_MAP: accession → isolate name
#      This is written to sample_isolate_map.tsv so merge_metadata.py
#      can join resistance labels by isolate name, not accession.
#
# Why isolate name?
#   Accession IDs can differ between ENA and SRA mirrors (ERR vs SRR).
#   The isolate name is the stable, study-assigned identifier that appears
#   in every drug resistance sheet, so it is used as the join key.

ACC_PREFIXES = ["ERR","SRR","DRR","ERX","SRX","ERS","SRS","SAMN","SAMEA"]

def _load_metadata(path):
    xl    = pd.ExcelFile(path)
    sheet = "All" if "All" in xl.sheet_names else xl.sheet_names[0]
    print(f"   Metadata sheet  : '{sheet}'")
    print(f"   Available sheets: {xl.sheet_names}")
    # Try row-2 header first 
    for hdr in [1, 0]:
        df = pd.read_excel(path, sheet_name=sheet, header=hdr)
        if "ena_run" in df.columns:
            print(f"   Header row      : {hdr+1}  (header={hdr})")
            return df
    # Last resort: scan for the row that contains "ena_run"
    raw = pd.read_excel(path, sheet_name=sheet, header=None)
    for i, row in raw.iterrows():
        if "ena_run" in row.values:
            df = pd.read_excel(path, sheet_name=sheet, header=i)
            print(f"   Header row      : {i+1}  (auto-detected)")
            return df
    print("⚠️  Could not find 'ena_run' column — returning empty DataFrame")
    print(f"   Columns seen: {list(pd.read_excel(path, sheet_name=sheet, header=0).columns)}")
    return pd.DataFrame()


meta_df_global = _load_metadata(METADATA)

SAMPLES      = []   # list of accession strings used as sample IDs
ISOLATE_MAP  = {}   # accession → isolate name

for _, row in meta_df_global.iterrows():
    isolate = str(row.get("isolate name", "")).strip()
    if not isolate or isolate.lower() in ("nan", "none", ""):
        continue
    for col in ["ena_run", "ena_experiment", "ena_sample"]:
        val = str(row.get(col, "")).strip().upper()
        if val and val.lower() not in ("nan", "none", "") and \
           any(val.startswith(p) for p in ACC_PREFIXES):
            SAMPLES.append(val)
            ISOLATE_MAP[val] = isolate
            break

SAMPLES = sorted(set(SAMPLES))

if len(SAMPLES) == 0:
    print("❌  No samples found! Check:")
    print(f"    • metadata path in config.yaml: {METADATA}")
    print("    • Sheet 'All' exists and has an 'ena_run' column")
    print("    • ena_run values start with ERR/SRR/DRR/ERX/ERS/SAMN/SAMEA")
    raise ValueError(f"No samples loaded from {METADATA}. See diagnostics above.")

print(f"📊 Found {len(SAMPLES)} samples  (e.g. {SAMPLES[:3]})")
print(f"   skip_assembly={SKIP_ASSEMBLY}  skip_card={SKIP_CARD}  "
      f"skip_pangenome={SKIP_PANGENOME}  skip_ml={SKIP_ML}")

# Write sample→isolate map so merge_metadata.py can use it without re-reading Excel
_map_path = Path(config["results_dir"]) / "ml" / "sample_isolate_map.tsv"
_map_path.parent.mkdir(parents=True, exist_ok=True)
pd.DataFrame(
    list(ISOLATE_MAP.items()), columns=["accession", "isolate_name"]
).to_csv(_map_path, sep="\t", index=False)
print(f"   Sample→isolate map written to {_map_path}")


# ── Wildcard constraints ───────────────────────────────────────────────────────
wildcard_constraints:
    sample     = r"[A-Za-z0-9]+",
    drug       = r"[a-z]+",
    input_type = r"snp|pan|snp_pan",
    model      = r"rf|lr",


# ==================== TARGET HELPERS ====================
def card_targets():
    if SKIP_CARD:
        return []
    return (
        expand(str(CARD_DIR / "{sample}.card_coverage.txt"), sample=SAMPLES)
        + [str(RESULTS_DIR / "card_summary.csv"),
           str(RESULTS_DIR / "card_binary_matrix.csv")]
    )

def assembly_targets():
    if SKIP_ASSEMBLY:
        return []
    return expand(str(ASSEMBLY_DIR / "{sample}" / "contigs.fasta"), sample=SAMPLES)

def annotation_targets():
    if SKIP_ANNOTATION:
        return []
    return expand(str(RESULTS_DIR / ".{sample}.annotate.done"), sample=SAMPLES)

def pangenome_targets():
    if SKIP_PANGENOME:
        return []
    return [str(ML_DIR / "pangenome_matrix_filtered.csv")]

def ml_targets():
    if SKIP_ML:
        return []
    out = []
    for it in ML_INPUT_TYPES:
        for drug in DRUGS:
            for model in ["rf", "lr"]:
                out += [
                    str(ML_DIR / f"{it}_{drug}_{model}_roc.png"),
                    str(ML_DIR / f"{it}_{drug}_{model}_top10_features.png"),
                    str(ML_DIR / f"{it}_{drug}_{model}_metrics.csv"),
                    str(ML_DIR / f"{it}_{drug}_{model}_best_hyperparams.csv"),
                    str(ML_DIR / f"{it}_{drug}_{model}_split_dist.png"),
                ]
    return out

# ==================== RULE ALL ====================
rule all:
    input:
        str(ML_DIR / "snp_matrix_filtered.csv"),
        card_targets(),
        # assembly_targets(),
        # annotation_targets(),
        pangenome_targets(),
        ml_targets(),


# ===========================================================
# STEP 1: DOWNLOAD FASTQ FROM ENA
# Raw FASTQs are temp() — deleted after preprocess_sample writes VCF.
# ===========================================================
rule download_fastq:
    output:
        r1   = temp(FASTQ_DIR / "{sample}_1.fastq.gz"),
        r2   = temp(FASTQ_DIR / "{sample}_2.fastq.gz"),
        done = touch(FASTQ_DIR / ".{sample}.done"),
    params:
        sample        = "{sample}",
        metadata_file = METADATA,
    log:
        LOGS_DIR / "download" / "{sample}.log",
    threads: 1
    retries: 3
    resources:
        downloads = 2, # change concurrent download
    conda:
        "workflow/envs/download.yaml"
    script:
        "workflow/scripts/download.py"


# ===========================================================
# STEP 2: TRIM + ALIGN + VARIANT CALL  (tbprofiler.py)
# Runs Trimmomatic → BWA-MEM → BCFtools inside the script.
# The script deletes trimmed FASTQs and BAMs itself after each step.
# ===========================================================
rule preprocess_sample:
    input:
        r1  = FASTQ_DIR / "{sample}_1.fastq.gz",
        r2  = FASTQ_DIR / "{sample}_2.fastq.gz",
        ref = REF,
    output:
        vcf  = VCF_DIR / "{sample}.targets.vcf.gz",
        tbi  = VCF_DIR / "{sample}.targets.vcf.gz.tbi",
        flag = touch(RESULTS_DIR / ".{sample}.preprocess.done"),
    params:
        sample  = "{sample}",
        tmp_dir = TMP_DIR,
        ref     = REF,
    log:
        LOGS_DIR / "preprocess" / "{sample}.log",
    threads: config.get("tbprofiler_threads", 4)
    conda:
        "workflow/envs/align.yaml"
    script:
        "workflow/scripts/tbprofiler.py"


# ===========================================================
# STEP 3: CARD AMR SCREENING  
# CARD BAM is temp() — deleted after coverage file is written.
# ===========================================================
if not SKIP_CARD:
    rule card_screen:
        input:
            r1 = FASTQ_DIR / "{sample}_1.fastq.gz",
            r2 = FASTQ_DIR / "{sample}_2.fastq.gz",
        output:
            bam      = temp(CARD_DIR / "{sample}.card.bam"),
            coverage = CARD_DIR / "{sample}.card_coverage.txt",
        params:
            index  = CARD_BT2,
            sample = "{sample}",
        log:
            LOGS_DIR / "card" / "{sample}.log",
        threads: config.get("bowtie2_threads", 4)
        conda:
            "workflow/envs/card.yaml"
        shell:
            """
            bowtie2 \
                -x {params.index} \
                -1 {input.r1} \
                -2 {input.r2} \
                -p {threads} \
                --very-sensitive-local \
                --no-unal \
            2> {log} \
            | samtools sort -@ {threads} -o {output.bam} - 2>> {log}
            samtools index {output.bam} 2>> {log}
            samtools coverage {output.bam} > {output.coverage} 2>> {log}
            """

    rule summarize_card:
        input:
            expand(str(CARD_DIR / "{sample}.card_coverage.txt"), sample=SAMPLES),
        output:
            summary       = RESULTS_DIR / "card_summary.csv",
            binary_matrix = RESULTS_DIR / "card_binary_matrix.csv",
        log:
            LOGS_DIR / "card_summary.log",
        threads: 1
        conda:
            "workflow/envs/ml.yaml"
        script:
            "workflow/scripts/summarize_card.py"


# ===========================================================
# STEPS 4-6: ASSEMBLY → ANNOTATION → PANGENOME  (all skippable)
# Assembly FASTAs and annotation GFFs are temp() — auto-deleted
# once Panaroo has consumed them.
# ===========================================================
if not SKIP_ASSEMBLY:
    rule assemble_genome:
        input:
            r1 = FASTQ_DIR / "{sample}_1.fastq.gz",
            r2 = FASTQ_DIR / "{sample}_2.fastq.gz",
        output:
            fasta = temp(ASSEMBLY_DIR / "{sample}" / "contigs.fasta"),
        params:
            outdir = str(ASSEMBLY_DIR / "{sample}"),
        log:
            LOGS_DIR / "spades" / "{sample}.log",
        threads: config.get("spades_threads", 8)
        resources:
            mem_mb = config.get("spades_mem_mb", 16000),
        conda:
            "workflow/envs/assembly.yaml"
        # --only-assembler later if we want to skip read error correction (but it seems to help even with clean Illumina data, so leaving it in for now)
        shell:
            """
            spades.py -1 {input.r1} -2 {input.r2} \
                -o {params.outdir} -t {threads} \
                --isolate \
                -m $(({resources.mem_mb} / 1000)) \
            > {log} 2>&1
            """

if not SKIP_ANNOTATION:
    rule annotate_genome:
        input:
            fasta = ASSEMBLY_DIR / "{sample}" / "contigs.fasta",
        output:
            gff = temp(ANNOT_DIR / "{sample}" / "{sample}.gff3"),
            gbk = temp(ANNOT_DIR / "{sample}" / "{sample}.gbff"),
        params:
            outdir    = str(ANNOT_DIR / "{sample}"),
            prefix    = "{sample}",
            db        = config.get("bakta_db", "db/bakta_db"),
            mincontig = config.get("bakta_min_contig", 500),
        log:
            LOGS_DIR / "bakta" / "{sample}.log",
        threads: config.get("bakta_threads", 4)
        resources:
            mem_mb = config.get("bakta_mem_mb", 12000),
        conda:
            "workflow/envs/annotation.yaml"
        shell:
            """
            bakta --db {params.db} --output {params.outdir} \
                --prefix {params.prefix} --threads {threads} \
                --min-contig-length {params.mincontig} --force \
                {input.fasta} > {log} 2>&1
            """

if not SKIP_PANGENOME:
    rule panaroo:
        input:
            gffs = expand(str(ANNOT_DIR / "{sample}" / "{sample}.gff3"), sample=SAMPLES),
        output:
            matrix = RESULTS_DIR / "pangenome" / "gene_presence_absence.csv",
            core   = RESULTS_DIR / "pangenome" / "core_gene_alignment.aln",
        params:
            outdir      = str(RESULTS_DIR / "pangenome"),
            mode        = config.get("panaroo_mode", "strict"),
            core_thresh = config.get("panaroo_core_threshold", 0.98),
        log:
            LOGS_DIR / "panaroo.log",
        threads: config.get("panaroo_threads", 8)
        conda:
            "workflow/envs/pangenome.yaml"
        shell:
            """
            panaroo -i {input.gffs} -o {params.outdir} \
                --clean-mode {params.mode} -t {threads} \
                --core_threshold {params.core_thresh} > {log} 2>&1
            """

    rule build_pangenome_matrix:
        input:
            pan_csv = RESULTS_DIR / "pangenome" / "gene_presence_absence.csv",
        output:
            matrix = ML_DIR / "pangenome_matrix.csv",
        log:   LOGS_DIR / "pangenome_matrix.log"
        conda: "workflow/envs/ml.yaml"
        script: "workflow/scripts/build_pangenome_matrix.py"

    rule filter_pangenome_matrix:
        input:
            matrix = ML_DIR / "pangenome_matrix.csv",
        output:
            filtered = ML_DIR / "pangenome_matrix_filtered.csv",
        params:
            maf_min     = config.get("pangenome_maf_min", 0.01),
            maf_max     = config.get("pangenome_maf_max", 0.99),
            matrix_type = "pangenome",
        log:     LOGS_DIR / "filter_pangenome_matrix.log"
        threads: config.get("matrix_threads", 8)
        conda:   "workflow/envs/ml.yaml"
        script:  "workflow/scripts/filter_matrix.py"


# ===========================================================
# STEP 7: COLLECT VCFs → BUILD SNP MATRIX → FILTER
# ===========================================================
rule collect_vcfs:
    input:
        expand(str(VCF_DIR / "{sample}.targets.vcf.gz"), sample=SAMPLES),
    output:
        touch(RESULTS_DIR / "vcf_collection.done"),


rule build_snp_matrix:
    input:
        flag = RESULTS_DIR / "vcf_collection.done",
    output:
        matrix = ML_DIR / "snp_matrix.csv",
        stats  = ML_DIR / "snp_matrix_stats.txt",
    params:
        vcf_dir = str(VCF_DIR),
    log:     LOGS_DIR / "snp_matrix.log"
    threads: config.get("matrix_threads", 8)
    conda:   "workflow/envs/ml.yaml"
    script:  "workflow/scripts/build_matrix.py"


rule filter_snp_matrix:
    input:
        matrix = ML_DIR / "snp_matrix.csv",
    output:
        filtered = ML_DIR / "snp_matrix_filtered.csv",
    params:
        maf_min     = config.get("snp_maf_min", 0.005),
        maf_max     = config.get("snp_maf_max", 0.995),
        matrix_type = "snp",
    log:     LOGS_DIR / "filter_snp_matrix.log"
    threads: config.get("matrix_threads", 8)
    conda:   "workflow/envs/ml.yaml"
    script:  "workflow/scripts/filter_matrix.py"


# ===========================================================
# STEP 8: CLEANUP — free storage after matrices are built
# ===========================================================
rule cleanup_vcfs:
    input:
        matrix = ML_DIR / "snp_matrix.csv",
    output:
        touch(RESULTS_DIR / "vcfs_cleaned.done"),
    log: LOGS_DIR / "cleanup_vcfs.log"
    shell:
        """
        echo "🗑️  Removing VCF files..." | tee {log}
        rm -f {VCF_DIR}/*.targets.vcf.gz {VCF_DIR}/*.targets.vcf.gz.tbi 2>> {log} || true
        echo "✅ VCF cleanup complete" | tee -a {log}
        """

if not SKIP_PANGENOME:
    rule cleanup_pangenome_intermediates:
        input:
            matrix = ML_DIR / "pangenome_matrix.csv",
        output:
            touch(RESULTS_DIR / "pangenome_intermediates_cleaned.done"),
        log: LOGS_DIR / "cleanup_pangenome.log"
        shell:
            """
            echo "🗑️  Removing assembly + annotation dirs..." | tee {log}
            rm -rf {ASSEMBLY_DIR} {ANNOT_DIR} 2>> {log} || true
            echo "✅ Done" | tee -a {log}
            """


# ===========================================================
# STEP 9: MERGE METADATA
# Passes sample_isolate_map.tsv to merge_metadata.py so it can
# join by isolate name instead of accession.
# ===========================================================
def _merge_inputs(wildcards):
    d = {
        "snp_matrix":       str(ML_DIR / "snp_matrix_filtered.csv"),
        "metadata":         METADATA,
        "master_data":      MASTERDATA,
        "sample_isolate_map": str(ML_DIR / "sample_isolate_map.tsv"),
    }
    d["pan_matrix"] = (
        str(ML_DIR / "pangenome_matrix_filtered.csv")
        if not SKIP_PANGENOME
        else str(ML_DIR / "snp_matrix_filtered.csv")
    )
    return d


rule merge_metadata:
    input:
        unpack(_merge_inputs),
    output:
        snp             = ML_DIR / "input_snp.csv",
        pan             = ML_DIR / "input_pan.csv",
        snp_pan         = ML_DIR / "input_snp_pan.csv",
        country_plot    = RESULTS_DIR / "country_distribution.png",
        resistance_plot = RESULTS_DIR / "resistance_summary.png",
    params:
        drugs          = DRUGS,
        skip_pangenome = SKIP_PANGENOME,
    log:   LOGS_DIR / "merge_metadata.log"
    conda: "workflow/envs/ml.yaml"
    script: "workflow/scripts/merge_metadata.py"


# ===========================================================
# STEPS 10-11: ML — RANDOM FOREST + LOGISTIC REGRESSION
# ===========================================================
if not SKIP_ML:
    rule run_rf:
        input:
            data = ML_DIR / "input_{input_type}.csv",
        output:
            roc              = ML_DIR / "{input_type}_{drug}_rf_roc.png",
            features         = ML_DIR / "{input_type}_{drug}_rf_top10_features.png",
            metrics          = ML_DIR / "{input_type}_{drug}_rf_metrics.csv",
            best_hyperparams = ML_DIR / "{input_type}_{drug}_rf_best_hyperparams.csv",
            split_dist       = ML_DIR / "{input_type}_{drug}_rf_split_dist.png",
        params:
            drug         = "{drug}",
            input_type   = "{input_type}",
            model        = "rf",
            random_state = config.get("random_state", 42),
            n_iter       = config.get("rf_n_iter", 20),
            cv_folds     = config.get("cv_folds", 5),
        log:     LOGS_DIR / "ml" / "{input_type}_{drug}_rf.log"
        threads: config.get("ml_threads", 4)
        conda:   "workflow/envs/ml.yaml"
        script:  "workflow/scripts/run_ml.py"

    rule run_lr:
        input:
            data = ML_DIR / "input_{input_type}.csv",
        output:
            roc              = ML_DIR / "{input_type}_{drug}_lr_roc.png",
            features         = ML_DIR / "{input_type}_{drug}_lr_top10_features.png",
            metrics          = ML_DIR / "{input_type}_{drug}_lr_metrics.csv",
            best_hyperparams = ML_DIR / "{input_type}_{drug}_lr_best_hyperparams.csv",
            split_dist       = ML_DIR / "{input_type}_{drug}_lr_split_dist.png",
        params:
            drug         = "{drug}",
            input_type   = "{input_type}",
            model        = "lr",
            random_state = config.get("random_state", 42),
            n_iter       = config.get("lr_n_iter", 20),
            cv_folds     = config.get("cv_folds", 5),
        log:     LOGS_DIR / "ml" / "{input_type}_{drug}_lr.log"
        threads: 1
        conda:   "workflow/envs/ml.yaml"
        script:  "workflow/scripts/run_ml.py"


onsuccess:
    print("\n🎉 Pipeline completed successfully!")
    print(f"   SNP matrix  → {ML_DIR}/snp_matrix_filtered.csv")
    if not SKIP_PANGENOME:
        print(f"   Pan matrix  → {ML_DIR}/pangenome_matrix_filtered.csv")
    if not SKIP_CARD:
        print(f"   CARD        → {RESULTS_DIR}/card_summary.csv")
    if not SKIP_ML:
        print(f"   ML results  → {ML_DIR}/")

onerror:
    print("\n❌ Pipeline failed — check logs/ for details.")
