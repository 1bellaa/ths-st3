#!/usr/bin/env python3
"""
Summarize CARD AMR gene detection across all samples from samtools coverage TSV.

Outputs:
  card_summary.csv          — long-format: sample, gene, aro_id, num_reads,
                              coverage_percent, mean_depth
  card_binary_matrix.csv    — wide-format binary presence/absence matrix
                              rows = samples, cols = AMR genes, values = 0/1

Memory-efficient version: streams long-format rows directly to disk and builds
the binary matrix as a compact dict-of-sets, avoiding loading all data at once.
"""

import sys
import csv
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

def parse_gene_info(rname):
    """Parse ARO ID and gene name from CARD rname field."""
    parts = str(rname).split("|")
    if len(parts) >= 6:
        return parts[4], parts[5]
    return "Unknown", rname

def main():
    card_files  = [Path(f) for f in snakemake.input]
    out_summary = Path(snakemake.output.summary)
    out_matrix  = Path(snakemake.output.binary_matrix)

    log("=" * 60)
    log(f"Summarizing CARD results  ({len(card_files)} samples)")
    log(f"  Min coverage  : {MIN_COVERAGE}%   Min mean depth: {MIN_MEANDEPTH}x")
    log("=" * 60)

    out_summary.parent.mkdir(parents=True, exist_ok=True)
    out_matrix.parent.mkdir(parents=True, exist_ok=True)

    # ── Tracking stats (no giant list in memory) ──────────────────────────────
    # binary_hits: sample → set of detected gene names
    binary_hits   = {}
    # gene_samples: gene → count of samples (for prevalence log)
    gene_samples  = {}
    total_pairs   = 0
    total_detected = 0
    skipped       = 0

    SUMMARY_COLS = [
        "sample", "gene", "aro_id", "num_reads",
        "coverage_percent", "mean_depth",
    ]

    # ── Stream long-format rows directly to CSV ───────────────────────────────
    with open(out_summary, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=SUMMARY_COLS)
        writer.writeheader()

        for i, card_file in enumerate(sorted(card_files)):
            if (i + 1) % 200 == 0:
                log(f"  Processed {i+1}/{len(card_files)} files …")

            try:
                df = pd.read_csv(card_file, sep="\t")
            except pd.errors.EmptyDataError:
                log(f"  ⚠️  {card_file.name} is empty — skipping")
                skipped += 1
                continue
            except Exception as e:
                log(f"  ⚠️  Could not read {card_file.name}: {e}")
                skipped += 1
                continue

            df.columns = [c.lstrip("#") for c in df.columns]
            sample = card_file.stem.replace(".card_coverage", "")
            binary_hits.setdefault(sample, set())

            for _, row in df.iterrows():
                aro_id, gene_name = parse_gene_info(row["rname"])
                cov_pct    = float(row.get("coverage",  0.0))
                mean_depth = float(row.get("meandepth", 0.0))
                num_reads  = int(row.get("numreads",    0))
                detected   = cov_pct >= MIN_COVERAGE and mean_depth >= MIN_MEANDEPTH

                total_pairs += 1

                # Write only rows with any reads (saves disk space too)
                if num_reads > 0 or detected:
                    writer.writerow({
                        "sample":           sample,
                        "gene":             gene_name,
                        "aro_id":           aro_id,
                        "num_reads":        num_reads,
                        "coverage_percent": round(cov_pct, 3),
                        "mean_depth":       round(mean_depth, 3),
                    })

                if detected:
                    total_detected += 1
                    binary_hits[sample].add(gene_name)
                    gene_samples[gene_name] = gene_samples.get(gene_name, 0) + 1

            # Free the per-file DataFrame immediately
            del df

    log(f"\n📊 Summary:")
    log(f"   Total gene-sample pairs   : {total_pairs}")
    log(f"   Detected                  : {total_detected}")
    log(f"   Samples with ≥1 AMR gene  : {sum(1 for s in binary_hits if binary_hits[s])}")
    log(f"   Unique AMR genes detected  : {len(gene_samples)}")
    if skipped:
        log(f"   Skipped files             : {skipped}")

    if gene_samples:
        log("\n   Top 10 most prevalent AMR genes:")
        top = sorted(gene_samples.items(), key=lambda x: x[1], reverse=True)[:10]
        for gene, cnt in top:
            log(f"     {gene}: {cnt} sample(s)")

    log(f"\n✅ Long-format summary → {out_summary}")

    # ── Binary presence/absence matrix ────────────────────────────────────────
    log("\n📐 Building binary presence/absence matrix...")

    all_genes = sorted(gene_samples.keys())
    all_samples = sorted(binary_hits.keys())

    n_samples = len(all_samples)
    n_genes   = len(all_genes)
    log(f"   Matrix: {n_samples} samples × {n_genes} genes")

    # Write matrix row-by-row — never materialise the full DataFrame
    with open(out_matrix, "w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["sample"] + all_genes)
        for sample in all_samples:
            detected_genes = binary_hits[sample]
            row = [sample] + [1 if g in detected_genes else 0 for g in all_genes]
            writer.writerow(row)

    # Log prevalence from the compact gene_samples dict (no DataFrame needed)
    if gene_samples and n_samples > 0:
        log("\n   Gene prevalence (% samples with gene detected):")
        top_prev = sorted(gene_samples.items(), key=lambda x: x[1], reverse=True)[:15]
        for gene, cnt in top_prev:
            if cnt > 0:
                log(f"     {gene}: {cnt/n_samples*100:.1f}%")

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