"""
plot_data_distribution.py
=========================
One figure per drug. Subplots: one per input_type.
Each subplot: stacked bar showing Train/Val/Test R/S split.
Reads directly from input_*.csv — does NOT require ML to have run.

Run standalone:
    python plot_data_distribution.py \
        --ml_dir results/ml \
        --drug isoniazid \
        --out results/plots
"""

import argparse
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Mode: Snakemake or CLI ────────────────────────────────────────────────────
try:
    input_files  = snakemake.input.input_files
    out_plot     = snakemake.output.plot
    drug         = snakemake.params.drug
    input_types  = snakemake.params.input_types
    RANDOM_STATE = snakemake.params.get("random_state", 42)
    log_file     = snakemake.log[0]
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    log = open(log_file, "w")
    def msg(m): print(m); log.write(m+"\n"); log.flush()
except NameError:
    p = argparse.ArgumentParser()
    p.add_argument("--ml_dir",     required=True)
    p.add_argument("--drug",       required=True)
    p.add_argument("--input_types", nargs="+",
                   default=["snp","pan","snp_pan","card","snp_card","pan_card","snp_pan_card"])
    p.add_argument("--out",        default="results/plots")
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()
    ml_dir       = Path(args.ml_dir)
    drug         = args.drug
    input_types  = args.input_types
    RANDOM_STATE = args.random_state
    input_files  = [str(ml_dir / f"input_{it}.csv") for it in input_types]
    out_plot     = str(Path(args.out) / f"{drug}_data_distribution.png")
    Path(args.out).mkdir(parents=True, exist_ok=True)
    def msg(m): print(m, flush=True)

msg(f"📊 Data distribution — {drug.upper()}")

TARGET_COL = f"{drug}_resistance"
COLOR_S    = "#2ecc71"
COLOR_R    = "#e74c3c"

INPUT_LABELS = {
    "snp":          "SNP only",
    "pan":          "Pangenome only",
    "snp_pan":      "SNP + Pan",
    "card":         "CARD only",
    "snp_card":     "SNP + CARD",
    "pan_card":     "Pan + CARD",
    "snp_pan_card": "SNP + Pan + CARD",
}

# ── Compute splits for each input type ───────────────────────────────────────
results = {}

for f, it in zip(input_files, input_types):
    try:
        df = pd.read_csv(f, index_col=0, low_memory=False)
        if TARGET_COL not in df.columns:
            msg(f"   ⚠️  {it}: '{TARGET_COL}' not found — skipping")
            continue
        df = df.dropna(subset=[TARGET_COL])
        y  = df[TARGET_COL].astype(int)
        if y.nunique() < 2 or len(y) < 20:
            msg(f"   ⚠️  {it}: insufficient data (n={len(y)}) — skipping")
            continue

        # Same split logic as run_ml.py
        _, _, y_train, y_temp = train_test_split(
            pd.DataFrame(index=df.index), y,
            test_size=0.40, random_state=RANDOM_STATE, stratify=y
        )
        _, _, y_val, y_test = train_test_split(
            pd.DataFrame(index=y_temp.index), y_temp,
            test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
        )

        def row(y_arr, split):
            r = int((y_arr==1).sum()); s = int((y_arr==0).sum())
            return {"split": split, "n": len(y_arr), "R": r, "S": s,
                    "R%": round(100*r/len(y_arr),1)}

        results[it] = [row(y_train,"Train"), row(y_val,"Val"), row(y_test,"Test")]
        msg(f"   {it}: n={len(y)}  Train={len(y_train)} Val={len(y_val)} Test={len(y_test)}")

    except Exception as e:
        msg(f"   ⚠️  {it}: {e}")

if not results:
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.text(0.5, 0.5, f"No data for {drug.upper()}",
            ha="center", va="center", transform=ax.transAxes, color="grey")
    ax.axis("off")
    plt.savefig(out_plot, dpi=120, bbox_inches="tight")
    plt.close()
    sys.exit(0)

# ── Layout ────────────────────────────────────────────────────────────────────
present = [it for it in input_types if it in results]
n       = len(present)
ncols   = min(4, n)
nrows   = int(np.ceil(n / ncols))

fig, axes = plt.subplots(nrows, ncols,
                          figsize=(5 * ncols, 4.5 * nrows),
                          sharey=False)
axes = np.array(axes).flatten()

for i, it in enumerate(present):
    ax    = axes[i]
    rows  = results[it]
    df_sp = pd.DataFrame(rows)
    x     = np.arange(len(df_sp))
    w     = 0.5

    ax.bar(x, df_sp["S"], w, label="Susceptible", color=COLOR_S, alpha=0.85)
    ax.bar(x, df_sp["R"], w, bottom=df_sp["S"],
           label="Resistant", color=COLOR_R, alpha=0.85)

    for j, row in df_sp.iterrows():
        ax.text(j, row["n"] + row["n"]*0.01,
                f"n={row['n']}\nR={row['R%']}%",
                ha="center", va="bottom", fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(df_sp["split"], fontsize=10)
    ax.set_ylabel("Sample count", fontsize=9)
    ax.set_title(INPUT_LABELS.get(it, it), fontweight="bold", fontsize=11)
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_ylim(0, df_sp["n"].max() * 1.22)

for j in range(n, len(axes)):
    axes[j].set_visible(False)

fig.suptitle(
    f"Train / Val / Test Split per Input Type\n{drug.upper()} (60/20/20 stratified)",
    fontweight="bold", fontsize=13, y=1.01,
)
plt.tight_layout()
Path(out_plot).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(out_plot, dpi=180, bbox_inches="tight")
plt.close()
msg(f"✅ Data distribution → {out_plot}")
try: log.close()
except: pass