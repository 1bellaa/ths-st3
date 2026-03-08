#!/usr/bin/env python3
"""
Summarize CARD AMR gene detection across all samples from samtools coverage TSV.

Outputs:
  card_summary.csv          — long-format: sample, gene, aro_id, num_reads,
                              coverage_percent, mean_depth
  card_binary_matrix.csv    — wide-format binary presence/absence matrix
                              rows = samples, cols = AMR genes, values = 0/1
"""

import sys
import pandas as pd
from pathlib import Path
from datetime import datetime

MIN_COVERAGE  = 80.0
MIN_MEANDEPTH =  5.0

def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}", flush=True)
    with open(snakemake.log[0], "a") as f:
        f.write(f"[{timestamp}] {msg}\n")

def main():
    card_files  = [Path(f) for f in snakemake.input]
    out_summary = Path(snakemake.output.summary)
    out_matrix  = Path(snakemake.output.binary_matrix)

    log("=" * 60)
    log(f"Summarizing CARD results  ({len(card_files)} samples)")
    log(f"  Min coverage  : {MIN_COVERAGE}%   Min mean depth: {MIN_MEANDEPTH}x")
    log("=" * 60)

    all_data = []

    for card_file in sorted(card_files):
        try:
            df = pd.read_csv(card_file, sep="\t")
        except pd.errors.EmptyDataError:
            log(f"  ⚠️  {card_file.name} is empty — skipping")
            continue
        except Exception as e:
            log(f"  ⚠️  Could not read {card_file.name}: {e}")
            continue

        df.columns = [c.lstrip("#") for c in df.columns]
        sample = card_file.stem.replace(".card_coverage", "")

        for _, row in df.iterrows():
            rname_parts = str(row["rname"]).split("|")
            if len(rname_parts) >= 6:
                aro_id    = rname_parts[4]
                gene_name = rname_parts[5]
            else:
                aro_id    = "Unknown"
                gene_name = row["rname"]

            cov_pct    = float(row.get("coverage",  0.0))
            mean_depth = float(row.get("meandepth", 0.0))
            num_reads  = int(row.get("numreads",    0))
            detected   = int(cov_pct >= MIN_COVERAGE and mean_depth >= MIN_MEANDEPTH)

            all_data.append({
                "sample":           sample,
                "gene":             gene_name,
                "aro_id":           aro_id,
                "num_reads":        num_reads,
                "coverage_percent": round(cov_pct, 3),
                "mean_depth":       round(mean_depth, 3),
                "detected":         detected,
            })

    # ── Long-format summary ───────────────────────────────────────────────────
    if all_data:
        df_summary = pd.DataFrame(all_data).sort_values(["sample", "gene"])
        detected_df = df_summary[df_summary["detected"] == 1]
        log(f"\n📊 Summary:")
        log(f"   Total gene-sample pairs   : {len(df_summary)}")
        log(f"   Detected                  : {len(detected_df)}")
        log(f"   Samples with ≥1 AMR gene  : {detected_df['sample'].nunique()}")
        log(f"   Unique AMR genes detected  : {detected_df['gene'].nunique()}")
        if not detected_df.empty:
            log("\n   Top 10 most prevalent AMR genes:")
            top = (detected_df.groupby("gene")["sample"]
                   .nunique().sort_values(ascending=False).head(10))
            for gene, cnt in top.items():
                log(f"     {gene}: {cnt} sample(s)")
    else:
        df_summary = pd.DataFrame(columns=[
            "sample","gene","aro_id","num_reads","coverage_percent","mean_depth","detected"])
        log("  ⚠️  No CARD data found")

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    df_summary.drop(columns=["detected"]).to_csv(out_summary, index=False)
    log(f"\n✅ Long-format summary → {out_summary}")

    # ── Binary presence/absence matrix ────────────────────────────────────────
    log("\n📐 Building binary presence/absence matrix...")
    if not df_summary.empty:
        binary = (
            df_summary[["sample","gene","detected"]]
            .pivot_table(index="sample", columns="gene",
                         values="detected", aggfunc="max", fill_value=0)
        )
        binary.columns.name = None
        binary.index.name   = "sample"

        n_genes   = binary.shape[1]
        n_samples = binary.shape[0]
        log(f"   Matrix: {n_samples} samples × {n_genes} genes")

        prevalence = (binary.sum() / n_samples * 100).sort_values(ascending=False)
        log("\n   Gene prevalence (% samples with gene detected):")
        for gene, pct in prevalence[prevalence > 0].head(15).items():
            log(f"     {gene}: {pct:.1f}%")
    else:
        binary = pd.DataFrame()
        log("   ⚠️  No detected genes — writing empty matrix")

    out_matrix.parent.mkdir(parents=True, exist_ok=True)
    binary.to_csv(out_matrix)
    log(f"✅ Binary matrix → {out_matrix}")
    log("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"❌ Fatal error: {e}")
        import traceback
        log(traceback.format_exc())
        sys.exit(1)