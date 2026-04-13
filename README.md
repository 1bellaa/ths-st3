# TB Drug Resistance Prediction Pipeline

A Snakemake workflow for predicting *Mycobacterium tuberculosis* drug resistance from whole-genome sequencing data using SNPs, pangenome features, and antimicrobial resistance gene screening.

## Contents

- [Description](#description)
- [Directory Structure](#directory-structure)
- [Installation](#installation)
- [Database Setup](#database-setup)
- [Input & Output](#input-and-output)
- [Usage](#usage)
- [Workflow](#workflow)
- [Configuration](#configuration)
- [Skip Flags](#skip-flags)
- [Citation](#citation)
- [FAQ](#faq)
- [Information](#information)

## Description

### Comprehensive Feature Engineering

The pipeline constructs three independent genomic feature sets and evaluates their predictive power individually and in combination:

- **SNP Matrix (Core Genome Variation)**: Reference-based variant calling identifies single nucleotide polymorphisms associated with drug resistance. Variants are filtered by minor allele frequency (0.5% ≤ MAF ≤ 99.5%) to retain informative sites while excluding sequencing errors and near-fixed positions.

- **Pangenome Matrix (Accessory Genome)**: De novo assembly followed by graph-based pangenome clustering captures gene presence/absence patterns. Genes present in 1-99% of isolates are retained, focusing on accessory genome variation absent from the H37Rv reference.

- **CARD AMR Gene Screening**: Direct alignment to the Comprehensive Antibiotic Resistance Database identifies known antimicrobial resistance genes via coverage-based detection (≥80% coverage, ≥5× depth).

### Systematic Feature Combinations

Seven feature matrices are constructed for comparative analysis:

1. **SNP only** — Core genome variation
2. **Pangenome only** — Accessory genome variation
3. **SNP + Pangenome** — Combined chromosomal features
4. **Resistome only** — Known AMR genes
5. **SNP + Resistome** — Point mutations + AMR genes
6. **Pangenome + Resistome** — Accessory genes + AMR genes  
7. **SNP + Pangenome + Resistome** — Complete feature set

This design allows evaluation of: (a) core vs. accessory gene mechanisms, (b) known vs. novel resistance determinants, and (c) effects of combined features.

### Machine Learning Classification

Two machine learning (ML) algorithms are evaluated for each drug-feature combination:

- **Random Forest**: Ensemble decision tree classifier capturing non-linear interactions and feature importance via Gini impurity. Hyperparameter optimization via randomized search (20 iterations, 5-fold CV).

- **Logistic Regression**: Regularized linear classifier (elastic net penalty) providing interpretable coefficients. Features scaled via MaxAbsScaler prior to training.

Models are trained on balanced datasets (RandomUnderSampler) with stratified 60/20/20 train/validation/test splits. Performance metrics include ROC AUC, sensitivity, specificity, precision, and F1-score.

### Biological Annotation

Top predictive features are mapped to functional context:

- **SNP features**: Annotated with gene names from TB Drug Resistance Database (TBDB)
- **Pangenome features**: Annotated with gene names and functional descriptions from Panaroo
- **esistome features**: Annotated with ARO identifiers from CARD database

### Fast

Preprocessing optimized for high-throughput analysis with automatic cleanup of intermediate files. Typical runtimes on a laptop (4 cores, 16 GB RAM):
- Variant calling: ~X min/sample
- Resistome screening: ~X min/sample  
- _De novo_ assembly: ~X min/sample

Storage management via Snakemake temp() directives automatically removes raw FASTQs, alignment BAMs, and assembly intermediates after downstream processing.

### Standardized & Reproducible

- **Conda environments**: Each pipeline stage runs in an isolated environment with pinned software versions
- **Snakemake workflow**: Declarative workflow definition ensures reproducibility and automatic parallelization
- **Version logging**: All tool versions and parameters documented in output files
- **Random seed control**: Fixed random state (42) for ML reproducibility

### Comprehensive Output

- Filtered feature matrices (SNP, pangenome, Resistome)
- Per-drug performance metrics (ROC AUC, sensitivity, specificity)
- Feature importance rankings with biological annotations
- ROC curves comparing models and feature types
- Hyperparameter optimization results




## Directory Structure
```
tb_pipeline/
├── Snakefile                         # Main workflow
├── config.yaml                       # All parameters (edit this)
├── metadata.xlsx                     # Sample metadata 
├── master_data.xlsx                  # Isolate information + resistance phenotypes
├── reference/
│   └── tbdb/
│       └── tbdb.fasta                # TB reference genome
├── db/
│   ├── card/
│   │   ├── card.fasta                # CARD database FASTA
│   │   ├── card.*.bt2          
│   │   └── card_lengths.txt          # Gene lengths
│   └── bakta_db/                     # Bakta annotation database
└── workflow/
    ├── scripts/
    │   ├── download.py               # ENA FASTQ downloader
    │   ├── build_matrix.py           # SNP binary matrix builder
    │   ├── build_pangenome_matrix.py # Panaroo → binary matrix
    │   ├── merge_metadata.py         # Attach resistance labels
    │   ├── filter_matrix.py          # Filters the SNP and pangenome matrix before merging
    │   ├── summarize_card.py
    │   ├── preprocess.py             # Run the SNP preprocessing   
    │   └── run_ml.py                 # RF + LR training & plots
    │   └── plot_feature_venn.py               
    │   └── plot_combined_roc.py                
    └── envs/
        ├── download.yaml
        ├── align.yaml
        ├── card.yaml
        ├── assembly.yaml
        ├── annotation.yaml
        ├── pangenome.yaml
        └── ml.yaml
```

## Installation

The pipeline requires Conda/Mamba and Snakemake. All bioinformatics tools and Python dependencies are automatically installed via Conda environments.


### Conda / Mamba

```bash
# Install Mamba
conda install -n base -c conda-forge mamba
```

### Snakemake

```bash
conda create -n snakemake -c conda-forge -c bioconda snakemake=7.32.0
conda activate snakemake
```

### Pipeline Download

```bash
git clone https://github.com/1bellaa/ths-st3.git
cd tb-resistance-pipeline
```

## Database Setup

### TB Reference Genome

**Option 1: TB-Profiler database (recommended)**

```bash
# Install TB-Profiler
conda install -c bioconda tb-profiler

# Download TBDB reference
tb-profiler update_tbdb

# Copy to pipeline directory
mkdir -p reference/tbdb
cp ~/.tb_profiler/tbdb.fasta reference/tbdb/tbdb.fasta

# Index with BWA
bwa index reference/tbdb/tbdb.fasta
```

### CARD Database

```bash
mkdir -p db/card
cd db/card

# Download CARD nucleotide FASTA
wget https://card.mcmaster.ca/latest/data
tar -xvf data
cp nucleotide_fasta_protein_homolog_model.fasta card.fasta

# Build Bowtie2 index
bowtie2-build card.fasta card

# Generate gene lengths file
python3 << 'EOF'
from Bio import SeqIO
with open('card_lengths.txt', 'w') as out:
    for rec in SeqIO.parse('card.fasta', 'fasta'):
        out.write(f'{rec.id}\t{len(rec.seq)}\n')
EOF

cd ../..
```

### Bakta Database

Required if running pangenome analysis (skip_assembly: false).

```bash
mkdir -p db/bakta_db

# Light database (~3.2 GB) - faster, sufficient for M. tuberculosis
bakta_db download --output db/bakta_db --type light

# Full database (~40 GB) - more comprehensive
# bakta_db download --output db/bakta_db --type full
```

## Input and Output

### Input Files

**metadata.xlsx** (sheet "All" with columns):
- `isolate name`: Stable isolate identifier
- `country`: Country of origin
- `ena_run` / `ena_experiment` / `ena_sample`: ENA accessions (at least one required)

Example:

| isolate name | country | ena_run   | ena_experiment | ena_sample |
|--------------|---------|-----------|----------------|------------|
| 0001         | India   | ERR123456 | ERX123456      | ERS123456  |
| 0002         | UK      | SRR234567 | SRX234567      | SRS234567  |

**master_data.xlsx** (separate sheets per drug):
- Sheet names: `ISONIAZID`, `RIFAMPICIN`, `ETHAMBUTOL`, `STREPTOMYCIN`
- Columns: `isolate name`, `resistance phenotype (pDST)` (values: R/S)

Example (sheet "ISONIAZID"):

| isolate name | resistance phenotype (pDST) |
|--------------|-----------------------------|
| 0001         | R                           |
| 0002         | S                           |

### Output Files

```
data/results/
├── ml/
│   ├── snp_matrix_filtered.csv              # Binary SNP matrix
│   ├── pangenome_matrix_filtered.csv        # Binary gene presence/absence
│   ├── input_snp.csv                        # SNP features + resistance labels
│   ├── input_pan.csv                        # Pangenome features + labels
│   ├── input_snp_pan.csv                    # Combined features + labels
│   ├── input_card.csv                       # CARD features + labels
│   ├── input_snp_card.csv                   # SNP + CARD + labels
│   ├── input_pan_card.csv                   # Pan + CARD + labels
│   ├── input_snp_pan_card.csv               # All features + labels
│   │
│   ├── snp_isoniazid_rf_roc.png             # ROC curve plots
│   ├── snp_isoniazid_rf_features.csv        # Top 10 feature importances
│   ├── snp_isoniazid_rf_metrics.csv         # Performance metrics
│   ├── snp_isoniazid_rf_best_hyperparams.csv
│   ├── snp_isoniazid_combined_roc.png       # RF vs LR comparison
│   ├── snp_model_summary.png                # Multi-drug performance
│   ├── annotated_features.csv               # Features mapped to gene names
│   └── ...
│
├── plots/
│   ├── isoniazid_roc_by_input_type.png      # Compare SNP vs Pan vs CARD
│   ├── isoniazid_data_distribution.png      # Train/Val/Test splits
│   └── snp_rf_feature_venn.png              # Feature overlap across drugs
│
├── card_summary.csv                          # CARD gene detection summary
├── card_binary_matrix.csv                    # CARD presence/absence matrix
├── country_distribution.png                  # Geographic distribution
└── resistance_summary.png                    # Resistance phenotype counts
```

### Metrics File Schema

```csv
drug,input_type,model,split,accuracy,precision,recall,f1,specificity,auc_roc
isoniazid,snp,rf,test,0.89,0.87,0.92,0.89,0.85,0.94
```

### Feature Importance Schema

```csv
feature,importance,drug,input_type,model
Chromosome_761155_C_T,0.082,rifampicin,snp,rf
group_3240,0.045,isoniazid,pan,rf
CARD_rpoB2,0.071,rifampicin,card,rf
```

## Usage

### Dry Run

Validate workflow before execution:

```bash
# Activate Snakemake environment
conda activate snakemake

# Check rule execution plan
snakemake --dry-run

# Visualize rule graph
snakemake --dag | dot -Tpng > dag.png
```

### Local Execution

```bash
# Run on local machine with 8 cores
snakemake --cores 8 --use-conda

# Verbose output
snakemake --cores 8 --use-conda --verbose

# Continue on sample failures
snakemake --cores 8 --use-conda --keep-going
```

### Cluster Execution (HPC)

To run the pipeline on an DOST-ASTI HPC cluster using SLURM, we use a wrapper script to submit the entire Snakemake workflow as a single batch job.

#### 1. The Submission Script
Create a file (e.g., `run_snakemake.sbatch`) with your cluster configurations. This script loads the environment and executes Snakemake across the allocated cores.

```bash
#!/bin/bash
#SBATCH --partition=batch
#SBATCH --qos=batch_default
#SBATCH --ntasks=1
#SBATCH --nodes=1
#SBATCH --cpus-per-task=86
#SBATCH --mem=100G
#SBATCH --output=logs/snakemake_%j.out

# Load environment
module load anaconda
source activate snakemake_env

# Benchmarking
start_time=$(date +%s.%N)
echo "Started on: $(date)"

# Execute Snakemake
snakemake --profile workflow/profiles/default --cores 86 --jobs 20

# Benchmarking wrap-up
end_time=$(date +%s.%N)
run_time=$(python -c "print($end_time - $start_time)")
echo "Finished on: $(date)"
echo "Total runtime (sec): ${run_time}"
```

#### 2. Launching the Workflow

Once your script is configured, submit it to the scheduler:

```bash

sbatch run_snakemake.sbatch

    Note: This method runs Snakemake on a single heavy-duty node. Ensure the --cores count in your snakemake command matches the --cpus-per-task requested in the SBATCH headers.
```

## Workflow

### Overview

```
ENA Download → Quality Trimming → Alignment/Assembly → Feature Matrices → ML Training
```

### Detailed Stages

**1. Data Download** (`download_fastq`)
- Queries ENA API for FASTQ URLs
- Downloads paired-end reads with retry logic (3 attempts)
- Validates gzip integrity

**2. Preprocessing** (`preprocess_sample`)
- Trimmomatic: Adapter removal, quality trimming (Q20, min 25 bp)
- BWA-MEM: Alignment to H37Rv reference
- BCFtools: Variant calling (mapping quality ≥30, base quality ≥20)
- Output: bgzipped + tabix-indexed VCF

**3. CARD Screening** (`summarize_card`)
- Bowtie2: Alignment to CARD database (very-sensitive-local)
- SAMtools: Calculate per-gene coverage
- Detection threshold: ≥80% coverage, ≥5× mean depth
- Output: Binary AMR gene matrix

**4. De Novo Assembly** (`assemble_genome`)  
*Optional: skip with `skip_assembly: true`*
- SPAdes: Isolate mode, automatic k-mer selection
- Output: Contigs in FASTA format

**5. Genome Annotation** (`annotate_genome`)  
*Optional: skip with `skip_annotation: true`*
- Bakta: Rapid bacterial annotation
- Light database: ~3 GB, sufficient for TB
- Output: GFF3 annotation files

**6. Pangenome Clustering** (`panaroo`)  
*Optional: skip with `skip_pangenome: true`*
- Panaroo: Graph-based pangenome construction
- Strict mode: Minimizes false positives
- Core gene threshold: 98%
- Output: Gene presence/absence matrix

**7. Feature Matrix Construction**
- SNP matrix: Parse VCFs, build binary matrix
- Pangenome matrix: Convert Panaroo output
- Filter by frequency: SNP (>0.5%), genes (1-99%)

**8. Metadata Merging** (`merge_metadata`)
- Map accessions → isolate names (handles ENA/SRA mismatches)
- Join resistance phenotypes from master_data.xlsx
- One-hot encode country
- Create 7 feature combinations

**9. Machine Learning** (`run_rf`, `run_lr`)
- Split: 60% train / 20% val / 20% test (stratified)
- Balance: RandomUnderSampler on training set
- Hyperparameter search: RandomizedSearchCV (20 iter, 5-fold CV)
- Metrics: ROC AUC, sensitivity, specificity, F1

**10. Visualization & Annotation**
- Map SNPs → genes (TBDB)
- Map gene clusters → functional annotation (Panaroo)
- ROC curves, feature importance plots, Venn diagrams

### Workflow Diagram

```
                      ┌─────────────┐
                      │ ENA Download│
                      └──────┬──────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
         │ BWA-MEM │    │ Bowtie2 │   │ SPAdes  │
         │(H37Rv)  │    │ (CARD)  │   │(de novo)│
         └────┬────┘    └────┬────┘   └────┬────┘
              │              │              │
         ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
         │BCFtools │    │Coverage │   │ Bakta   │
         │  (VCF)  │    │  Stats  │   │ (GFF3)  │
         └────┬────┘    └────┬────┘   └────┬────┘
              │              │              │
         ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
         │   SNP   │    │  CARD   │   │ Panaroo │
         │ Matrix  │    │ Matrix  │   │  (Pan)  │
         └────┬────┘    └────┬────┘   └────┬────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────▼────────┐
                    │ Filter Matrices │ (MAF thresholds)
                    └────────┬────────┘
                             │
                    ┌────────▼────────┐
                    │ Merge Metadata  │ (Attach resistance labels)
                    └────────┬────────┘
                             │
              7 Feature Combinations:
         • SNP only       • SNP+Pan       • SNP+Pan+CARD
         • Pan only       • SNP+CARD
         • CARD only      • Pan+CARD
                             │
                    ┌────────▼────────┐
                    │   ML Training   │ (RF + LR per drug)
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
         ┌────▼────┐    ┌────▼────┐   ┌────▼────┐
         │ROC Curve│    │ Feature │   │  Model  │
         │  Plots  │    │Rankings │   │ Metrics │
         └─────────┘    └─────────┘   └─────────┘
```

## Configuration

Edit `config.yaml` before running:

```yaml
# ========== FILE PATHS ==========
reference_genome: "reference/tbdb/tbdb.fasta"
card_bowtie2_index: "db/card/card"
card_lengths: "db/card/card_lengths.txt"
bakta_db: "db/bakta_db"
tbdb_bed: "reference/tbdb/tbdb.bed"
metadata: "metadata.xlsx"
master_data: "master_data.xlsx"

# ========== DRUGS ==========
drugs:
  - isoniazid
  - rifampicin
  - ethambutol
  - streptomycin

# ========== QUALITY CONTROL ==========
trimmomatic_leading: 3
trimmomatic_trailing: 3
trimmomatic_slidingwindow: "4:20"
trimmomatic_minlen: 25

# ========== FEATURE FILTERING ==========
snp_maf_min: 0.005      # 0.5%
snp_maf_max: 1.0        # 100%
pangenome_maf_min: 0.01 # 1%
pangenome_maf_max: 0.99 # 99%

# ========== MACHINE LEARNING ==========
random_state: 42
cv_folds: 5
rf_n_iter: 20
lr_n_iter: 20

# ========== COMPUTATIONAL RESOURCES ==========
preprocess_mem_mb: 4000 
card_mem_mb:       4000 
matrix_mem_mb:     4000
spades_mem_mb:    8000 
bakta_mem_mb:     8000 
panaroo_mem_mb:   16000

download_threads:   4
preprocess_threads: 4
bowtie2_threads:    4
spades_threads:     4 
bakta_threads:      8
panaroo_threads:    8
matrix_threads:     4
ml_threads:         4

# ========== SKIP FLAGS ==========
skip_assembly: false
skip_annotation: false
skip_pangenome: false
skip_card: false
skip_ml: false
```

### Resource Tuning

**Low RAM (≤16 GB)**:
```yaml
spades_mem_mb: 8000
spades_threads: 4
```

**High Performance cluster**:
```yaml
preprocess_threads: 16
spades_threads: 32
panaroo_threads: 16
```

## Skip Flags

### SNP-Only Pipeline (Fast)

```yaml
skip_assembly: true    
skip_card: true
skip_ml: false
```

**Runtime**: X min/sample

### Testing Configuration

For quick validation on small datasets:

```yaml
skip_assembly: true
skip_card: true
drugs:
  - isoniazid  
```

### Rerun ML Without Reprocessing

If `input_*.csv` files already exist:

```bash
snakemake --cores 4 --use-conda \
  --forcerun run_rf run_lr \
  --until annotate_features
```

## Citation

If you use this pipeline in your research, please cite: (as if lmao)

**Pipeline Paper** (if published):
> X

This pipeline is *made possible due to many execellent tools*, namely:

### Tools

- **Trimmomatic**: Bolger AM, Lohse M, Usadel B. Trimmomatic: a flexible trimmer for Illumina sequence data. *Bioinformatics*. 2014;30(15):2114-2120. https://doi.org/10.1093/bioinformatics/btu170

- **BWA**: Li H, Durbin R. Fast and accurate short read alignment with Burrows-Wheeler transform. *Bioinformatics*. 2009;25(14):1754-1760. https://doi.org/10.1093/bioinformatics/btp324

- **SAMtools/BCFtools**: Danecek P, Bonfield JK, Liddle J, et al. Twelve years of SAMtools and BCFtools. *GigaScience*. 2021;10(2):giab008. https://doi.org/10.1093/gigascience/giab008

- **Bowtie2**: Langmead B, Salzberg SL. Fast gapped-read alignment with Bowtie 2. *Nature Methods*. 2012;9(4):357-359. https://doi.org/10.1038/nmeth.1923

- **SPAdes**: Prjibelski A, Antipov D, Meleshko D, et al. Using SPAdes De Novo Assembler. *Current Protocols in Bioinformatics*. 2020;70(1):e102. https://doi.org/10.1002/cpbi.102

- **Bakta**: Schwengers O, Jelonek L, Dieckmann MA, et al. Bakta: rapid and standardized annotation of bacterial genomes via alignment-free sequence identification. *Microbial Genomics*. 2021;7(11):000685. https://doi.org/10.1099/mgen.0.000685

- **Panaroo**: Tonkin-Hill G, MacAlasdair N, Ruis C, et al. Producing polished prokaryotic pangenomes with the Panaroo pipeline. *Genome Biology*. 2020;21(1):180. https://doi.org/10.1186/s13059-020-02090-4

- **scikit-learn**: Pedregosa F, Varoquaux G, Gramfort A, et al. Scikit-learn: Machine Learning in Python. *Journal of Machine Learning Research*. 2011;12:2825-2830.

- **Snakemake**: Mölder F, Jablonski KP, Letcher B, et al. Sustainable data analysis with Snakemake. *F1000Research*. 2021;10:33. https://doi.org/10.12688/f1000research.29032.2

### Databases

- **CARD**: Alcock BP, Huynh W, Chalil R, et al. CARD 2023: expanded curation, support for machine learning, and resistome prediction at the Comprehensive Antibiotic Resistance Database. *Nucleic Acids Research*. 2023;51(D1):D690-D699. https://doi.org/10.1093/nar/gkac920

- **TBDB**: Phelan, J. TBDB: Standard database for the TBProfiler tool. GitHub repository. 2024. https://github.com/jodyphelan/tbdb

- **Bakta DB**: Schwengers, O. Bakta database. 2025. https://doi.org/10.5281/zenodo.14916843

## Information

The preprocessing of the genome data and running of the machine learning models were made possible by the Computing and Archiving Research Environment (COARE) service provided by the Department of Science and Technology's Advanced Science and Technology (DOST-ASTI). Using the High-Performance Computing (HPC) service of the organization made the processing significantly faster, given the limited computational resources of the group. Their service in addressing our requests are appreciated by the group.

---

**Developed by**: Rachel Lauren Manlapig, Andrea Euceli Loria
**Institution**: De La Salle University 
**Contact**: rachelmanlapig.acads@gmail.com, andreae.loria@gmail.com 
**License**: ??   
**Documentation**: https://github.com/1bellaa/ths-st3.git
