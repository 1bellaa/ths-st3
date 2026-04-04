"""
plot_feature_venn.py
====================
Usage:
    python plot_feature_venn.py --ml-dir /path/to/ml/results --out-dir /path/to/output
"""

import argparse
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from venn import venn

# ── CLI ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument("--ml-dir", required=True)
parser.add_argument("--out-dir", default="results/plots")
args = parser.parse_args()

ml_dir = Path(args.ml_dir)
out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# ── DRUGS & INPUT TYPES ──────────────────────────────────────────────────────
drugs = ["isoniazid", "rifampicin", "ethambutol", "streptomycin"]
input_types = ["snp", "pan", "card", "snp_pan", "pan_card", "snp_card", "snp_pan_card"]

def msg(m): print(m, flush=True)

# ── Core function ────────────────────────────────────────────────────────────
def process_model(feature_files, model_name, out_plot, out_csv, input_type):
    msg(f"\n📊 {input_type.upper()} — {model_name}")

    drug_features = {}
    drug_imp = {}

    for f, drug in zip(feature_files, drugs):
        try:
            df = pd.read_csv(f)
            if df.empty or "gene_name" not in df.columns:
                msg(f"⚠️ {drug}: empty or missing 'gene_name'")
                continue
            grouped = df.groupby("gene_name")["importance"].max()
            drug_features[drug] = set(grouped.index)
            drug_imp[drug] = grouped.to_dict()
            msg(f"✔ {drug}: {len(drug_features[drug])} features")
        except Exception as e:
            msg(f"⚠️ {drug}: {e}")

    if not drug_features:
        pd.DataFrame().to_csv(out_csv, index=False)
        return

    present_drugs = list(drug_features.keys())
    all_genes = set().union(*drug_features.values())

    # ── Membership table ────────────────────────────────────────────────────
    records = []
    for gene in all_genes:
        membership = {d: gene in drug_features[d] for d in present_drugs}
        n_present = sum(membership.values())
        imp_max = max([drug_imp[d].get(gene, 0) for d in present_drugs])
        records.append({
            "gene": gene,
            "n_drugs": n_present,
            "importance_max": round(imp_max, 5),
            **{f"in_{d}": membership[d] for d in present_drugs}
        })

    feat_df = pd.DataFrame(records).sort_values(
        ["importance_max", "n_drugs"], ascending=[False, False]
    )
    feat_df.to_csv(out_csv, index=False)

    # ── Venn ───────────────────────────────────────────────────────────────
    venn_dict = {d: drug_features[d] for d in present_drugs}
    plt.figure(figsize=(8, 8))
    venn(venn_dict)
    plt.title(f"{model_name} — {input_type}", fontsize=12)

    # Annotation: max importance + shared genes
    common_genes = [g for g in all_genes if all(g in drug_features[d] for d in present_drugs)]
    shared_genes = [g for g in all_genes if sum(g in drug_features[d] for d in present_drugs) >= 2 and g not in common_genes]

    if common_genes:
        print(f"\n🔹 {model_name} — {input_type} — Shared in ALL drugs:")
        for g in common_genes:
            max_val = 0
            max_drug = ""
            for d in present_drugs:
                val = drug_imp[d].get(g, 0)
                if val > max_val:
                    max_val = val
                    max_drug = d
            print(f"• {g}: max importance {max_val:.5f} in {max_drug}")

    common_annot = [f"• {g} ({max([drug_imp[d].get(g,0) for d in present_drugs]):.5f}, {','.join([d for d in present_drugs if g in drug_features[d]])})" for g in common_genes[:8]]
    shared_annot = [f"• {g} ({max([drug_imp[d].get(g,0) for d in present_drugs if g in drug_features[d]]):.5f}, {','.join([d for d in present_drugs if g in drug_features[d]])})" for g in shared_genes[:12]]

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


# ── Combined per model ────────────────────────────────────────────────────
def process_combined_model(all_feature_files, model_name, out_plot, out_csv):
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
                for g, v in grouped.items():
                    combined[g] = max(v, combined.get(g, 0))
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
        imp_max = max([drug_imp[d].get(gene, 0) for d in drugs])
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

    # Venn plot
    venn_dict = {d: drug_features[d] for d in drugs}
    plt.figure(figsize=(10, 10))
    venn(venn_dict)
    plt.title(f"{model_name} — COMBINED INPUTS", fontsize=13)

    common_genes = [g for g in all_genes if all(g in drug_features[d] for d in drugs)]
    shared_genes = [g for g in all_genes if sum(g in drug_features[d] for d in drugs) >= 2 and g not in common_genes]

    common_annot = [f"• {g} ({max([drug_imp[d].get(g,0) for d in drugs]):.5f}, {','.join([d for d in drugs if g in drug_features[d]])})" for g in common_genes[:8]]
    shared_annot = [f"• {g} ({max([drug_imp[d].get(g,0) for d in drugs if g in drug_features[d]]):.5f}, {','.join([d for d in drugs if g in drug_features[d]])})" for g in shared_genes[:12]]

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