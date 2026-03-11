"""
Machine Learning script for TB drug-resistance prediction.
Handles both Random Forest (model='rf') and Logistic Regression (model='lr').

Outputs per drug per input_type:
  - ROC curve PNG (RF vs LR combined)
  - Top-10 feature importance PNG
  - Best hyperparameters CSV
  - Per-set metrics CSV (train / val / test)

Snakemake script — no subprocess calls.
"""

import gc
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import loguniform, uniform
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import MaxAbsScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")

# SNAKEMAKE BINDINGS
data_path    = snakemake.input.data
out_roc        = snakemake.output.roc
out_features   = snakemake.output.features
out_params     = snakemake.output.best_hyperparams
out_metrics    = snakemake.output.metrics
out_split_dist = snakemake.output.split_dist
out_roc_data   = snakemake.output.roc_data
drug         = snakemake.params.drug
input_type   = snakemake.params.input_type
model_type   = snakemake.params.model          # "rf" or "lr"
RANDOM_STATE = snakemake.params.random_state
N_ITER       = snakemake.params.n_iter
CV_FOLDS     = snakemake.params.cv_folds
log_file     = snakemake.log[0]

Path(log_file).parent.mkdir(parents=True, exist_ok=True)
log = open(log_file, "w")

def msg(m):
    print(m)
    log.write(m + "\n")
    log.flush()

sns.set_style("whitegrid")
plt.rcParams["font.family"] = "sans-serif"

TARGET_COL = f"{drug}_resistance"

# LOAD DATA
msg(f"📥 Loading {input_type} data for {drug} ({model_type.upper()})...")
df = pd.read_csv(data_path, index_col=0, low_memory=False)
msg(f"   Raw shape: {df.shape[0]} rows × {df.shape[1]} cols")

# CLEAN DATA
df = df.replace("", np.nan)
if TARGET_COL not in df.columns:
    msg(f"❌ Target column '{TARGET_COL}' not found in data. Skipping.")
    # Write empty placeholder outputs so Snakemake does not fail
    for out in [out_roc, out_features]:
        plt.figure(); plt.title("No data"); plt.savefig(out); plt.close()
    pd.DataFrame().to_csv(out_params)
    pd.DataFrame().to_csv(out_metrics)
    plt.figure(); plt.title("No data"); plt.savefig(out_split_dist); plt.close()
    log.close()
    sys.exit(0)

# DROP ROWS WITH MISSING TARGET
before = len(df)
df = df.dropna(subset=[TARGET_COL])
dropped = before - len(df)
if dropped > 0:
    msg(f"   Dropped {dropped} rows with missing '{TARGET_COL}' label")
msg(f"   Labelled samples remaining: {len(df)}")

# ONE-HOT ENCODING OF CATEGORICAL METADATA (e.g. country) 
if "country" in df.columns:
    df = pd.get_dummies(df, columns=["country"])

# Drop irrelevant metadata columns 
drop_cols = [
    "ena_experiment", "ena_sample",
    "isoniazid_resistance", "rifampicin_resistance",
    "ethambutol_resistance", "streptomycin_resistance",
    "pyrazinamide_resistance", "levofloxacin_resistance",
]
drop_cols = [c for c in drop_cols if c != TARGET_COL]
df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

# PREPARE FEATURE MATRIX AND TARGET VECTOR
X = df.drop(columns=[TARGET_COL]).select_dtypes(include=["number"])
y = df[TARGET_COL].astype(int)
feature_names = X.columns.tolist()

msg(f"  Features: {X.shape[1]}   Samples: {X.shape[0]}   "
    f"Resistant: {y.sum()}   Susceptible: {(y==0).sum()}")

if X.shape[0] < 20 or y.nunique() < 2:
    msg("⚠️  Insufficient samples or single class — writing empty outputs.")
    for out in [out_roc, out_features]:
        plt.figure(); plt.title("Insufficient data"); plt.savefig(out); plt.close()
    pd.DataFrame().to_csv(out_params)
    pd.DataFrame().to_csv(out_metrics)
    plt.figure(); plt.title("Insufficient data"); plt.savefig(out_split_dist); plt.close()
    log.close()
    sys.exit(0)

# TRAIN / TEST / VALIDATION SPLIT (60 / 20 / 20) 
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)
msg(f"  Train: {len(y_train)}  Val: {len(y_val)}  Test: {len(y_test)}")

# SPLIT RESISTANCE DISTRIBUTION — log + save image
def _split_dist(y, name):
    r = int((y == 1).sum()); s = int((y == 0).sum())
    msg(f"   {name:6s}  n={len(y):4d}  R={r:4d}  S={s:4d}  "
        f"R%={100*r/len(y):.1f}%")
    return {"split": name, "n": len(y), "Resistant (R)": r, "Susceptible (S)": s}

msg(f"\n📊 Resistance distribution per split ({drug.upper()}):")
split_rows = [
    _split_dist(y_train, "Train"),
    _split_dist(y_val,   "Val"),
    _split_dist(y_test,  "Test"),
]
split_df = pd.DataFrame(split_rows)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# Left: stacked bar chart
x      = range(len(split_df))
width  = 0.5
bars_s = axes[0].bar(x, split_df["Susceptible (S)"], width, label="Susceptible",
                     color="#2ecc71", alpha=0.85)
bars_r = axes[0].bar(x, split_df["Resistant (R)"],   width,
                     bottom=split_df["Susceptible (S)"], label="Resistant",
                     color="#e74c3c", alpha=0.85)
axes[0].set_xticks(list(x))
axes[0].set_xticklabels(split_df["split"])
axes[0].set_ylabel("Sample count")
axes[0].set_title(f"R/S Distribution per Split\n{drug.upper()} [{input_type}]",
                  fontweight="bold")
axes[0].legend()
axes[0].grid(True, axis="y", alpha=0.3)

# Right: table
axes[1].axis("off")
tbl = axes[1].table(
    cellText=split_df.values,
    colLabels=split_df.columns,
    cellLoc="center", loc="center",
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10)
tbl.scale(1.1, 1.8)
for j in range(len(split_df.columns)):
    tbl[0, j].set_facecolor("#2c3e50")
    tbl[0, j].set_text_props(color="white", fontweight="bold")
for i in range(1, len(split_df) + 1):
    tbl[i, 0].set_facecolor("#ecf0f1")
    tbl[i, 1].set_facecolor("#f0f0f0")
    tbl[i, 2].set_facecolor("#d5f5e3")
    tbl[i, 3].set_facecolor("#fadbd8")

plt.tight_layout()
plt.savefig(out_split_dist, dpi=180, bbox_inches="tight")
plt.close()
msg(f"✅ Split distribution plot → {out_split_dist}")

# OTHER METRICS
def calc_metrics(y_true, y_pred, y_proba):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "accuracy":    accuracy_score(y_true, y_pred),
        "precision":   precision_score(y_true, y_pred, zero_division=0),
        "recall":      recall_score(y_true, y_pred, zero_division=0),
        "f1":          f1_score(y_true, y_pred, zero_division=0),
        "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        "auc_roc":     roc_auc_score(y_true, y_proba),
        "confusion_matrix": cm,
    }

# TRAIN MODELS
if model_type == "rf":
    base_model = Pipeline([
        ("rf", RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)),
    ])
    param_dist = {
        "rf__n_estimators":      [100, 200, 300],
        "rf__max_depth":         [5, 10, 20, None],
        "rf__min_samples_split": [2, 5, 10],
        "rf__min_samples_leaf":  [1, 2, 4],
        "rf__bootstrap":         [True], # i forgot this, but by default it's true
        "rf__max_features":      ["sqrt", "log2"],
        "rf__class_weight":      ["balanced"],
    }
    X_tr, X_v, X_te = X_train, X_val, X_test

else:  # lr
    base_model = Pipeline([
        ("scaler", MaxAbsScaler()),
        ("lr",     LogisticRegression(random_state=RANDOM_STATE)),
    ])
    param_dist = {
        "lr__C":            loguniform(1e-2, 1e2),
        "lr__penalty":      ["l1", "l2", "elasticnet"],
        "lr__solver":       ["saga"],
        "lr__l1_ratio":     uniform(0, 1),
        "lr__max_iter":     [100, 500, 1000], # 500, 1000 ---girl edit this later ‼️‼️‼️
        "lr__class_weight": ["balanced"],
    }
    X_tr, X_v, X_te = X_train, X_val, X_test

msg(f"🔧 Training {model_type.upper()} with {N_ITER} iter × {CV_FOLDS}-fold CV...")
search = RandomizedSearchCV(
    base_model,
    param_dist,
    n_iter=N_ITER,
    cv=CV_FOLDS,
    scoring="roc_auc",
    n_jobs=1,
    random_state=RANDOM_STATE,
    verbose=0,
)
search.fit(X_tr, y_train)
best = search.best_estimator_
best_params = search.best_params_
msg(f"  Best CV AUC: {search.best_score_:.4f}")
msg(f"  Params: {best_params}")

# PREDICT AND CALCULATE METRICS ON ALL SETS
sets = {
    "val":   (X_v,  y_val),
    "test":  (X_te, y_test),
}
all_metrics = {}
for split_name, (Xs, ys) in sets.items():
    yp  = best.predict(Xs)
    ypr = best.predict_proba(Xs)[:, 1]
    all_metrics[split_name] = calc_metrics(ys, yp, ypr)
    msg(f"  {split_name:5s} AUC={all_metrics[split_name]['auc_roc']:.4f}  "
        f"F1={all_metrics[split_name]['f1']:.4f}")

# FEATURE IMPORTANCE
feat_names = list(feature_names)
if model_type == "rf":
    importances = best.named_steps['rf'].feature_importances_
    imp_label   = "Importance"
else:
    importances = best.named_steps['lr'].coef_.flatten()
    imp_label   = "Coefficient"

top_idx      = np.argsort(np.abs(importances))[::-1][:10]
top_features = pd.DataFrame({
    "feature":    [feat_names[i] for i in top_idx],
    "importance": importances[top_idx],
})

# SAVE BEST HYPERPARAMETERS CSV
params_row = {
    "drug":       drug,
    "input_type": input_type,
    "model":      model_type,
    "cv_auc":     search.best_score_,
}
params_row.update({str(k): str(v) for k, v in best_params.items()})
pd.DataFrame([params_row]).to_csv(out_params, index=False)

# SAVE METRICS CSV
metrics_rows = []
for split_name, m in all_metrics.items():
    metrics_rows.append({
        "drug":        drug,
        "input_type":  input_type,
        "model":       model_type,
        "split":       split_name,
        "accuracy":    m["accuracy"],
        "precision":   m["precision"],
        "recall":      m["recall"],
        "f1":          m["f1"],
        "specificity": m["specificity"],
        "auc_roc":     m["auc_roc"],
    })
pd.DataFrame(metrics_rows).to_csv(out_metrics, index=False)

# ROC CURVE
roc_rows = []
for split_name in ["val", "test"]:
    Xs, ys      = sets[split_name]
    ypr         = best.predict_proba(Xs)[:, 1]
    fpr, tpr, _ = roc_curve(ys, ypr)
    auc         = all_metrics[split_name]["auc_roc"]
    for f, t in zip(fpr, tpr):
        roc_rows.append({
            "model": model_type, "drug": drug, "input_type": input_type,
            "split": split_name, "auc": auc, "fpr": f, "tpr": t,
        })

roc_data_df = pd.DataFrame(roc_rows)
roc_data_df.to_csv(out_roc_data, index=False)
msg(f"✅ ROC data saved → {out_roc_data}")

color      = "#2ecc71" if model_type == "rf" else "#3498db"
label_name = "Random Forest" if model_type == "rf" else "Logistic Regression"

fig, ax = plt.subplots(figsize=(7, 6))
for split_name, style in [("test", "-"), ("val", "--")]:
    subset = roc_data_df[roc_data_df["split"] == split_name]
    auc    = all_metrics[split_name]["auc_roc"]
    ax.plot(subset["fpr"], subset["tpr"], lw=2, linestyle=style, color=color,
            label=f"{split_name.capitalize()} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.6, label="Random (AUC=0.500)")
ax.set_title(f"{label_name} — {drug.capitalize()} [{input_type}]", fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(out_roc, dpi=200, bbox_inches="tight")
plt.close()
msg(f"✅ ROC curve saved → {out_roc}")

# Individual model ROC plot (single model, all 3 splits)
fig, ax = plt.subplots(figsize=(7, 6))
for split_name, style in [("test", "-"), ("val", "--")]:
    subset = roc_data_df[roc_data_df["split"] == split_name]
    auc    = all_metrics[split_name]["auc_roc"]
    ax.plot(subset["fpr"], subset["tpr"], lw=2, linestyle=style, color=color,
            label=f"{split_name.capitalize()} (AUC={auc:.3f})")

ax.plot([0, 1], [0, 1], "k:", lw=1.5, alpha=0.6, label="Random (AUC=0.500)")
ax.set_title(f"{label_name} — {drug.capitalize()} [{input_type}]",
             fontweight="bold")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
ax.set_xlim([-0.02, 1.02])
ax.set_ylim([-0.02, 1.02])
plt.tight_layout()
plt.savefig(out_roc, dpi=200, bbox_inches="tight")
plt.close()
msg(f"✅ ROC curve saved → {out_roc}")

# TOP 10 FEATURE IMPORTANCE
fig, ax = plt.subplots(figsize=(8, 5))
palette = "viridis" if model_type == "rf" else None

if model_type == "lr":
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_features["importance"]]
    ax.barh(
        range(len(top_features)),
        top_features["importance"].values,
        color=colors,
        alpha=0.85,
    )
    ax.axvline(0, color="black", lw=0.8)
    legend_els = [
        mpatches.Patch(fc="#d62728", alpha=0.85, label="↑ Promotes Resistance"),
        mpatches.Patch(fc="#2ca02c", alpha=0.85, label="↓ Promotes Susceptibility"),
    ]
    ax.legend(handles=legend_els, fontsize=9)
else:
    colors = sns.color_palette("YlGn", len(top_features))[::-1]
    ax.barh(
        range(len(top_features)),
        top_features["importance"].values,
        color=colors,
        alpha=0.85,
    )

ax.set_yticks(range(len(top_features)))
ax.set_yticklabels(top_features["feature"].values, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel(imp_label)
ax.set_title(
    f"Top 10 Features — {label_name} — {drug.capitalize()} [{input_type}]",
    fontweight="bold",
)
ax.grid(True, axis="x", alpha=0.3)
plt.tight_layout()
plt.savefig(out_features, dpi=200, bbox_inches="tight")
plt.close()
msg(f"✅ Feature importance saved → {out_features}")

del search, best
gc.collect()

msg("🎉 Done")
log.close()