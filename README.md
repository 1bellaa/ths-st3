# TB Drug Resistance Genomics Pipeline

Snakemake workflow for whole-genome TB drug resistance analysis.

## Pipeline Overview

```
ENA Download → Trimmomatic → BWA-MEM → BCFtools
↘ Bowtie2 (CARD screening)
↘ SPAdes → Bakta → Panaroo

→ SNP Matrix → Pangenome Matrix → SNP + Pangenome Matrix 

→ ML Models (RF + LR)
↘ ROC Curves
↘ Best Hyperparameters
↘ Top 10 Features
```

### Tools and Versions

| Tool | Version | Purpose |
|---|---|---|
| Trimmomatic | 0.40 | Read quality trimming |
| BWA-MEM | 0.7.19 | Reference genome alignment |
| SAMtools | 1.22.1 | BAM processing and indexing |
| BCFtools | 1.22 | SNP/variant calling |
| Bowtie2 | 2.5.4 | CARD AMR database screening |
| SPAdes | 3.15.5 | De novo genome assembly |
| Bakta | 1.11.4 | Genome annotation |
| Panaroo | 1.3.0 | Pangenome analysis |

‼️‼️‼️‼️‼️pls add other dependencies used based on what's written in the yaml

---

## Directory Structure

```
tb_pipeline/
├── Snakefile                   # Main workflow
├── config.yaml                 # All parameters (edit this)
├── metadata.xlsx               # Sample metadata 
├── master_data.xlsx            # Isolate information + resistance phenotypes
├── reference/tbdb
│   └── tbdb.fasta              # TB reference genome
├── db/
│   ├── card/
│   │   ├── card.fasta          # CARD database FASTA
│   │   ├── card.*.bt2          
│   │   └── card_lengths.txt    # Gene lengths
│   └── bakta_db/               # Bakta annotation database
└── workflow/
    └──scripts/
│   │  ├── download.py             # ENA FASTQ downloader
│   │  ├── build_matrix.py         # SNP binary matrix builder
│   │  ├── build_pangenome_matrix.py  # Panaroo → binary matrix
│   │  ├── merge_metadata.py       # Attach resistance labels
│   │  ├── card_screen.py
    │  ├── filter_matrix.py        # filters the snp and pangenome matrix before merging
    │  ├── summarize_card.py
    │  ├── tbprofiler.py           # run the snp preprocessing   
    │  └── run_ml.py               # RF + LR training & plots 
    └──envs/
        ├── download.yaml
        ├── trim.yaml
        ├── align.yaml
        ├── card.yaml
        ├── assembly.yaml
        ├── annotation.yaml
        ├── pangenome.yaml
        └── ml.yaml
```

---

## Prerequisites

- conda
- snakemake
- TBDB reference genome
```bash
mkdir -p reference
tb-profiler update_tbdb
cp ~/.tbprofiler/tbdb/tbdb.fasta reference/tbdb.fasta
```
- CARD database
- Metadata file

## Configuration

Edit `config.yaml` before running. Key settings:

```yaml
# Paths — adjust if your files are elsewhere
reference_genome:   "reference/tbdb.fasta"
card_db:            "db/card/card.fasta"
card_bowtie2_index: "db/card/card"
card_lengths:       "db/card/card_lengths.txt"
bakta_db:           "db/bakta_db"
metadata:           "metadata.xlsx"

# Drugs to analyse
drugs:
  - isoniazid
  - rifampicin
  - ethambutol
  - streptomycin

# Thread counts — tune to your cluster/machine
bwa_threads:    8
spades_threads: 8
ml_threads:     4

# ML settings
cv_folds:   5
rf_n_iter:  20
lr_n_iter:  20
```

---

## Running the Pipeline
TBD
---

## Citation
add links or smth maybe docs???
- [Trimmomatic]()
- [BWA]()
- [SAMtools / BCFtools]()
- [Bowtie2]()
- [SPAdes]()
- [Bakta]()
- [Panaroo]()
- [CARD]()
- [Snakemake]()
