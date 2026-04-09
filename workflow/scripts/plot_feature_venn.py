"""
plot_combined_feature_venn.py
=============================
Aggregation rule: combined RF, LR, and RF+LR feature Venn diagrams
across ALL input types and drugs.

Usage (CLI/standalone):
    python plot_combined_feature_venn.py \
        --ml-dir /path/to/ml/results \
        --out-dir /path/to/output

When run via Snakemake, all inputs/outputs/params are taken from the rule.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from venn import venn

# ── Input handling: Snakemake or CLI ─────────────────────────────────────────
if "snakemake" in dir():
    input_types = list(snakemake.params.input_types)
    drugs       = list(snakemake.params.drugs)
    ml_dir      = Path(snakemake.params.ml_dir)
    out_dir     = Path(snakemake.params.out_dir)

    out_combined_rf_venn    = Path(snakemake.output.combined_rf_venn)
    out_combined_rf_csv     = Path(snakemake.output.combined_rf_csv)
    out_combined_lr_venn    = Path(snakemake.output.combined_lr_venn)
    out_combined_lr_csv     = Path(snakemake.output.combined_lr_csv)
    out_combined_rf_lr_venn = Path(snakemake.output.combined_rf_lr_venn)
    out_combined_rf_lr_csv  = Path(snakemake.output.combined_rf_lr_csv)
else:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ml-dir",  required=True)
    parser.add_argument("--out-dir", default="results/plots")
    args = parser.parse_args()
    ml_dir  = Path(args.ml_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    input_types = ["snp", "pan", "card", "snp_pan", "pan_card", "snp_card", "snp_pan_card"]
    drugs       = ["isoniazid", "rifampicin", "ethambutol", "streptomycin"]

    out_combined_rf_venn    = out_dir / "combined_rf_feature_venn.png"
    out_combined_rf_csv     = out_dir / "combined_rf_feature_venn.csv"
    out_combined_lr_venn    = out_dir / "combined_lr_feature_venn.png"
    out_combined_lr_csv     = out_dir / "combined_lr_feature_venn.csv"
    out_combined_rf_lr_venn = out_dir / "combined_rf_lr_feature_venn.png"
    out_combined_rf_lr_csv  = out_dir / "combined_rf_lr_feature_venn.csv"

# ── Constants ─────────────────────────────────────────────────────────────────
drug_codes = {
    "isoniazid":   "INH",
    "rifampicin":  "RIF",
    "ethambutol":  "EMB",
    "streptomycin":"STM",
}

def msg(m): print(m, flush=True)

# ── Helpers ───────────────────────────────────────────────────────────────────
def save_table_png(drug_features, drug_imp, drugs, out_png, title, min_drugs=2, include_model=False, out_csv=None):
    all_genes = set().union(*drug_features.values())
    records = []
    for gene in all_genes:
        membership = {d: gene in drug_features[d] for d in drugs}
        n_present = sum(membership.values())
        if n_present >= min_drugs:
            row = {"Gene": gene, "n_drugs": n_present}
            for d in drugs:
                if membership[d]:
                    val_model = drug_imp[d][gene]
                    if include_model and isinstance(val_model, tuple):
                        val, model = val_model
                        row[d] = f"{val:.3f} ({model})"
                    else:
                        val = val_model if isinstance(val_model, float) else val_model[0]
                        row[d] = f"{val:.3f}"
                else:
                    row[d] = "-"
            records.append(row)

    if not records:
        print(f"⚠️ No genes found for {title}")
        return

    df = pd.DataFrame(records).sort_values(["n_drugs"], ascending=False)

    if out_csv:
        df.to_csv(out_csv, index=False)
        print(f"✅ Saved CSV → {out_csv}")

    fig, ax = plt.subplots(figsize=(12, max(4, len(df) * 0.4)))
    ax.axis("off")
    table = ax.table(cellText=df.values, colLabels=df.columns, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.auto_set_column_width(col=list(range(len(df.columns))))
    plt.title(title, fontsize=12)
    plt.savefig(out_png, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"✅ Saved table PNG → {out_png}")


# ── Combined per model (RF only or LR only) ───────────────────────────────────
def process_combined_model(all_feature_files, model_name, out_plot, out_csv, threshold=0.005):
    msg(f"\n📊 Combined {model_name} — all input types")

    drug_features = {}
    drug_imp = {}

    for drug in drugs:
        combined = {}
        for files in all_feature_files:
            try:
                df = pd.read_csv(files[drug])
                if df.empty or "gene_name" not in df.columns:
                    continue
                grouped = df.groupby("gene_name")["importance"].max()
                grouped = grouped[grouped >= threshold]
                for g, v in grouped.items():
                    combined[g] = max(v, combined.get(g, 0))
            except Exception as e:
                msg(f"⚠️ {drug} file error: {e}")
                continue
        drug_features[drug] = set(combined.keys())
        drug_imp[drug] = combined
        msg(f"{drug}: {len(combined)} genes above threshold {threshold}")

    all_genes = set().union(*drug_features.values())
    if not all_genes:
        msg(f"⚠️ No genes passed threshold for {model_name}")
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    records = []
    for gene in all_genes:
        membership = {d: gene in drug_features[d] for d in drugs}
        n_present = sum(membership.values())
        imp_max = max([drug_imp[d].get(gene, 0) for d in drugs if gene in drug_imp[d]])
        records.append({
            "gene": gene,
            "n_drugs": n_present,
            "importance_max": round(imp_max, 5),
            **{f"in_{d}": membership[d] for d in drugs}
        })

    feat_df = pd.DataFrame(records).sort_values(
        ["importance_max", "n_drugs"], ascending=[False, False]
    )
    feat_df.to_csv(out_csv, index=False)

    venn_dict = {d: drug_features[d] for d in drugs}
    plt.figure(figsize=(10, 10))
    venn(venn_dict)
    plt.title(f"{model_name} — COMBINED INPUTS", fontsize=13)

    common_genes = [g for g in all_genes if all(g in drug_features[d] for d in drugs)]
    shared_genes = [g for g in all_genes if sum(g in drug_features[d] for d in drugs) >= 2 and g not in common_genes]

    common_annot = [
        f"• {g} ({max([drug_imp[d][g] for d in drugs if g in drug_imp[d]]):.5f})"
        for g in common_genes[:10]
    ]
    shared_annot = [
        f"• {g} ({max([drug_imp[d][g] for d in drugs if g in drug_imp[d]]):.5f})"
        for g in shared_genes[:10]
    ]

    annotation_text = []
    if common_annot:
        annotation_text.append("Shared in ALL drugs:")
        annotation_text += common_annot
    if shared_annot:
        annotation_text.append("\nShared (≥2 drugs):")
        annotation_text += shared_annot

    if annotation_text:
        plt.gcf().text(1.05, 0.2, "\n".join(annotation_text), fontsize=8, va="top")

    save_table_png(
        drug_features,
        {d: {g: (v, model_name) for g, v in drug_imp[d].items()} for d in drugs},
        drugs,
        out_plot.with_name(out_plot.stem.replace("_feature_venn", "_table") + ".png"),
        title=f"Combined {model_name} — Genes in ≥2 Drugs",
        include_model=True,
    )

    plt.savefig(out_plot, dpi=180, bbox_inches="tight")
    plt.close()
    msg(f"✅ Saved → {out_plot}")


# ── Combined RF+LR across all input types ─────────────────────────────────────
def process_combined_rf_lr(rf_all_files, lr_all_files, out_plot, out_csv, threshold=0.005):
    msg("\n📊 Combined RF+LR — all input types")

    drug_features = {}
    drug_imp = {}

    for drug in drugs:
        combined = {}
        for files_dict, model_name in zip([rf_all_files, lr_all_files], ["RF", "LR"]):
            for files in files_dict:
                try:
                    df = pd.read_csv(files[drug])
                    if df.empty or "gene_name" not in df.columns:
                        continue
                    grouped = df.groupby("gene_name")["importance"].max()
                    grouped = grouped[grouped >= threshold]
                    for g, v in grouped.items():
                        if g not in combined or v > combined[g][0]:
                            combined[g] = (v, model_name)
                except Exception as e:
                    msg(f"⚠️ {drug} file error: {e}")
                    continue

        combined = {g: val for g, val in combined.items() if val[0] >= threshold}
        drug_features[drug] = set(combined.keys())
        drug_imp[drug] = combined
        msg(f"{drug}: {len(combined)} genes above threshold {threshold}")

    all_genes = set().union(*drug_features.values())
    if not all_genes:
        msg("⚠️ No genes passed threshold for Combined RF+LR")
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    records = []
    for gene in all_genes:
        membership = {d: gene in drug_features[d] for d in drugs}
        n_present = sum(membership.values())
        imp_max = max([drug_imp[d][gene][0] for d in drugs if gene in drug_imp[d]])
        best_models = ",".join([drug_imp[d][gene][1] for d in drugs if gene in drug_imp[d]])
        records.append({
            "gene": gene,
            "n_drugs": n_present,
            "importance_max": round(imp_max, 5),
            **{f"in_{d}": membership[d] for d in drugs},
            "best_model": best_models,
        })

    feat_df = pd.DataFrame(records).sort_values(
        ["importance_max", "n_drugs"], ascending=[False, False]
    )
    feat_df.to_csv(out_csv, index=False)
    msg(f"✅ Saved CSV → {out_csv}")

    common_genes = [g for g in all_genes if all(g in drug_features[d] for d in drugs)]
    shared_genes = [g for g in all_genes if sum(g in drug_features[d] for d in drugs) >= 2 and g not in common_genes]

    common_annot = [
        f"• {g} ({max([drug_imp[d][g][0] for d in drugs if g in drug_imp[d]]):.5f}, "
        + ",".join(f"{drug_codes[d]}({drug_imp[d][g][1]})" for d in drugs if g in drug_features[d])
        + ")"
        for g in common_genes[:12]
    ]
    shared_annot = [
        f"• {g} ({max([drug_imp[d][g][0] for d in drugs if g in drug_features[d]]):.5f}, "
        + ",".join(f"{drug_codes[d]}({drug_imp[d][g][1]})" for d in drugs if g in drug_features[d])
        + ")"
        for g in shared_genes[:10]
    ]

    annotation_text = []
    if common_annot:
        annotation_text.append("Shared in ALL drugs:")
        annotation_text += common_annot
    if shared_annot:
        annotation_text.append("\nShared (≥2 drugs):")
        annotation_text += shared_annot

    plt.figure(figsize=(10, 10))
    venn_dict = {d: drug_features[d] for d in drugs}
    venn(venn_dict)
    plt.title("Combined RF+LR — all input types", fontsize=13)

    if annotation_text:
        plt.gcf().text(1.05, 0.2, "\n".join(annotation_text), fontsize=8, va="top")

    plt.savefig(out_plot, dpi=180, bbox_inches="tight")
    plt.close()
    msg(f"✅ Saved → {out_plot}")

    drug_imp_tuples = {d: {g: drug_imp[d][g] for g in drug_features[d]} for d in drugs}
    save_table_png(
        drug_features,
        drug_imp_tuples,
        drugs,
        out_plot.with_name(out_plot.stem.replace("_feature_venn", "_table") + ".png"),
        title="Combined RF+LR — Genes in ≥2 Drugs",
        include_model=True,
        out_csv=out_plot.with_name(out_plot.stem.replace("_feature_venn", "_table") + ".csv"),
    )


# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════

rf_all_files = [{d: ml_dir / f"{it}_{d}_rf_features.csv" for d in drugs} for it in input_types]
lr_all_files = [{d: ml_dir / f"{it}_{d}_lr_features.csv" for d in drugs} for it in input_types]

msg("── Combined RF only ──")
process_combined_model(rf_all_files, "RF", out_combined_rf_venn, out_combined_rf_csv)

msg("── Combined LR only ──")
process_combined_model(lr_all_files, "LR", out_combined_lr_venn, out_combined_lr_csv)

msg("── Combined RF+LR ──")
process_combined_rf_lr(rf_all_files, lr_all_files, out_combined_rf_lr_venn, out_combined_rf_lr_csv)

msg("\n🎉 ALL DONE")

# ── Combined RF+LR across all input types ──────────────────────────────────
def process_combined_rf_lr(rf_all_files, lr_all_files, out_plot, out_csv):
    msg("\n📊 Combined RF+LR — all input types")

    drug_features = {}
    drug_imp = {}  # store max importance per gene

    for drug in drugs:
        combined = {}
        for files_dict, model_name in zip([rf_all_files, lr_all_files], ["RF", "LR"]):
            for files in files_dict:
                try:
                    df = pd.read_csv(files[drug])
                    if df.empty or "gene_name" not in df.columns:
                        continue
                    grouped = df.groupby("gene_name")["importance"].max()
                    for g, v in grouped.items():
                        if g not in combined or v > combined[g][0]:
                            combined[g] = (v, model_name)
                except:
                    continue
        drug_features[drug] = set(combined.keys())
        drug_imp[drug] = combined

    all_genes = set().union(*drug_features.values())

    # Membership table
    records = []
    for gene in all_genes:
        membership = {d: gene in drug_features[d] for d in drugs}
        n_present = sum(membership.values())
        imp_max = max([drug_imp[d][gene][0] for d in drugs if gene in drug_imp[d]])
        best_models = ','.join([drug_imp[d][gene][1] for d in drugs if gene in drug_imp[d]])
        records.append({
            "gene": gene,
            "n_drugs": n_present,
            "importance_max": round(imp_max,5),
            **{f"in_{d}": membership[d] for d in drugs},
            "best_model": best_models
        })

    feat_df = pd.DataFrame(records).sort_values(
        ["importance_max", "n_drugs"], ascending=[False, False]
    )
    feat_df.to_csv(out_csv, index=False)

    # Venn
    venn_dict = {d: drug_features[d] for d in drugs}
    plt.figure(figsize=(10, 10))
    venn(venn_dict)
    plt.title("Combined RF+LR — all input types", fontsize=13)

    common_genes = [g for g in all_genes if all(g in drug_features[d] for d in drugs)]
    shared_genes = [g for g in all_genes if sum(g in drug_features[d] for d in drugs) >= 2 and g not in common_genes]

    common_annot = [f"• {g} ({max([drug_imp[d][g][0] for d in drugs if g in drug_imp[d]]):.5f}, {','.join([d+'('+drug_imp[d][g][1]+')' for d in drugs if g in drug_features[d]])})" for g in common_genes[:8]]
    shared_annot = [f"• {g} ({max([drug_imp[d][g][0] for d in drugs if g in drug_features[d]]):.5f}, {','.join([d+'('+drug_imp[d][g][1]+')' for d in drugs if g in drug_features[d]])})" for g in shared_genes[:12]]

    annotation_text = []
    if common_annot:
        annotation_text.append("Shared in ALL drugs:")
        annotation_text += common_annot
    if shared_annot:
        annotation_text.append("\nShared (≥2 drugs):")
        annotation_text += shared_annot

    if annotation_text:
        plt.gcf().text(1.05, 0.2, "\n".join(annotation_text), fontsize=8, va="top")

    plt.savefig(out_plot, dpi=180, bbox_inches="tight")
    plt.close()
    msg(f"✅ Saved → {out_plot}")


# ── MAIN LOOP ───────────────────────────────────────────────────────────────
for input_type in input_types:
    msg(f"\n==============================")
    msg(f"Processing input_type: {input_type}")
    msg(f"==============================")

    rf_files = [ml_dir / f"{input_type}_{d}_rf_features.csv" for d in drugs]
    lr_files = [ml_dir / f"{input_type}_{d}_lr_features.csv" for d in drugs]

    rf_plot = out_dir / f"{input_type}_rf_feature_venn.png"
    lr_plot = out_dir / f"{input_type}_lr_feature_venn.png"

    rf_csv  = out_dir / f"{input_type}_rf_feature_venn.csv"
    lr_csv  = out_dir / f"{input_type}_lr_feature_venn.csv"

    process_model(rf_files,'RF',rf_plot,rf_csv,input_type)
    process_model(lr_files,'LR',lr_plot,lr_csv,input_type)

# Combined RF only
rf_all_files = [{d: ml_dir / f"{it}_{d}_rf_features.csv" for d in drugs} for it in input_types]
process_combined_model(rf_all_files, 'RF', out_dir / "combined_rf_feature_venn.png",
                       out_dir / "combined_rf_feature_venn.csv")

# Combined LR only
lr_all_files = [{d: ml_dir / f"{it}_{d}_lr_features.csv" for d in drugs} for it in input_types]
process_combined_model(lr_all_files, 'LR', out_dir / "combined_lr_feature_venn.png",
                       out_dir / "combined_lr_feature_venn.csv")

# Combined RF+LR
process_combined_rf_lr(rf_all_files, lr_all_files,
                       out_dir / "combined_rf_lr_feature_venn.png",
                       out_dir / "combined_rf_lr_feature_venn.csv")

msg("\n🎉 ALL DONE")