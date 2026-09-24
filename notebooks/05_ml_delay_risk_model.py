# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Delay-risk model: will this aircraft's next departure be 15+ min late?
# MAGIC **Use case:** when an American aircraft lands at a hub, predict whether its next departure will leave 15+ minutes late, so the hub control center can prioritise the riskiest turns (extra ground crew, early boarding, a spare aircraft).
# MAGIC
# MAGIC **Features** use only information known when the inbound flight lands: hub, inbound origin, outbound destination, weekday, month, departure hour, scheduled turn time, inbound arrival delay, turn slack. No outbound-flight information leaks in.
# MAGIC
# MAGIC | # | Model | Why it's here |
# MAGIC |---|---|---|
# MAGIC | 1 | Logistic regression | Simple, explainable **baseline** |
# MAGIC | 2 | Random forest | Classic non-linear ensemble (bagging) |
# MAGIC | 3 | Histogram gradient boosting | Modern boosted trees, fast on 1M+ rows |
# MAGIC
# MAGIC All three use the same features, the same **time-based split** (train Aug 2024 – Jan 2026, test Feb – Jul 2026) and the same metrics.
# MAGIC Every run, metric and chart is logged to MLflow; the winner is registered in Unity Catalog with the alias **`champion`**.
# MAGIC Results are exported as CSVs for Tableau.

# COMMAND ----------

import json, os, zipfile, shutil, time
import numpy as np, pandas as pd, matplotlib.pyplot as plt, mlflow
from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient
from pyspark.sql import functions as F
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.calibration import calibration_curve
from sklearn.metrics import (roc_auc_score, average_precision_score, brier_score_loss, roc_curve,
                             precision_recall_curve, f1_score, precision_score, recall_score, confusion_matrix)

CATALOG, SCHEMA = "workspace", "aa_ops"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.delay_risk_model"
SPLIT_DATE = "2026-02-01"
OUT = f"/Volumes/{CATALOG}/{SCHEMA}/raw_bts/exports/ml"
os.makedirs(OUT, exist_ok=True)
mlflow.set_registry_uri("databricks-uc")
# Use a dedicated, named experiment (not the notebook's default one, which is deleted if the notebook is deleted)
_user = spark.sql("SELECT current_user()").first()[0]
mlflow.set_experiment(f"/Users/{_user}/aa_hub_ops_delay_risk_experiment")

COLORS = {"Logistic regression": "#9AA5B1", "Random forest": "#E8833A", "Gradient boosting": "#1F5AA6"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "font.size": 10})

# COMMAND ----------

# MAGIC %md ## 1 · Data and time-based split

# COMMAND ----------

pdf = (spark.table(f"{CATALOG}.{SCHEMA}.silver_aircraft_turns")
    .select("flight_date", "hub", "in_origin", "out_dest",
            F.col("day_of_week").cast("double").alias("day_of_week"),
            F.month("flight_date").cast("double").alias("month"),
            F.col("out_dep_hour").cast("double").alias("dep_hour"),
            F.col("sched_turn_min").cast("double").alias("sched_turn_min"),
            F.col("in_arr_delay").cast("double").alias("inbound_arr_delay_min"),
            F.col("slack_min").cast("double").alias("turn_slack_min"),
            F.col("out_d15").cast("int").alias("label"))
    .dropna().toPandas())
pdf["flight_date"] = pd.to_datetime(pdf["flight_date"])

CAT = ["hub", "in_origin", "out_dest"]
NUM = ["day_of_week", "month", "dep_hour", "sched_turn_min", "inbound_arr_delay_min", "turn_slack_min"]
FEATURES = CAT + NUM
train, test = pdf[pdf.flight_date < SPLIT_DATE], pdf[pdf.flight_date >= SPLIT_DATE]
X_tr, y_tr, X_te, y_te = train[FEATURES], train.label.values, test[FEATURES], test.label.values
print(f"train {len(train):,} rows ({y_tr.mean():.1%} late)  |  test {len(test):,} rows ({y_te.mean():.1%} late)")

# COMMAND ----------

# MAGIC %md ## 2 · Define and train the three models

# COMMAND ----------

ordinal = lambda: ColumnTransformer(
    [("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1), CAT)],
    remainder="passthrough", verbose_feature_names_out=False)

models = {
    "Logistic regression": Pipeline([
        ("pre", ColumnTransformer([
            ("hub", OneHotEncoder(handle_unknown="ignore"), ["hub"]),
            ("num", StandardScaler(), NUM)], verbose_feature_names_out=False)),
        ("clf", LogisticRegression(max_iter=1000, C=1.0))]),
    "Random forest": Pipeline([
        ("pre", ordinal()),
        ("clf", RandomForestClassifier(n_estimators=150, max_depth=16, min_samples_leaf=50,
                                       max_features="sqrt", n_jobs=-1, random_state=42))]),
    "Gradient boosting": Pipeline([
        ("pre", ordinal()),
        ("clf", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06, max_leaf_nodes=63,
                                               min_samples_leaf=200, l2_regularization=1.0,
                                               categorical_features=[0], early_stopping=True,
                                               validation_fraction=0.1, random_state=42))]),
}
# Random forest is slower, so it trains on a 400k random sample of the training period
RF_SAMPLE = 400_000

def metrics(y, p):
    # best-F1 threshold is chosen per model so the comparison is fair
    prec, rec, thr = precision_recall_curve(y, p)
    f1s = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-9, None)
    t = float(thr[np.nanargmax(f1s)])
    top = p >= np.quantile(p, 0.9)
    return {"roc_auc": roc_auc_score(y, p), "pr_auc": average_precision_score(y, p),
            "brier": brier_score_loss(y, p), "best_threshold": t,
            "f1": f1_score(y, p >= t), "precision": precision_score(y, p >= t), "recall": recall_score(y, p >= t),
            "late_rate_top10pct": float(y[top].mean()), "lift_top10pct": float(y[top].mean() / y.mean())}

preds, results, runs = {}, [], {}
for name, pipe in models.items():
    Xf, yf = (X_tr, y_tr)
    if name == "Random forest" and len(X_tr) > RF_SAMPLE:
        idx = np.random.default_rng(0).choice(len(X_tr), RF_SAMPLE, replace=False)
        Xf, yf = X_tr.iloc[idx], y_tr[idx]
    with mlflow.start_run(run_name=name) as run:
        t0 = time.time(); pipe.fit(Xf, yf); fit_s = time.time() - t0
        p = pipe.predict_proba(X_te)[:, 1]
        m = metrics(y_te, p); m["train_seconds"] = fit_s
        mlflow.log_params({"model": name, "train_rows": len(Xf), "features": ",".join(FEATURES), "split_date": SPLIT_DATE})
        mlflow.log_metrics(m)
        mlflow.sklearn.log_model(pipe, name="model",
                                 signature=infer_signature(X_te.head(2000), p[:2000]), input_example=X_te.head(5))
        preds[name], runs[name] = p, run.info.run_id
        results.append({"model": name, **m})
        print(f"{name:22s} ROC AUC {m['roc_auc']:.3f} | PR AUC {m['pr_auc']:.3f} | F1 {m['f1']:.3f} | {fit_s:.0f}s")

comparison = pd.DataFrame(results).sort_values("roc_auc", ascending=False).reset_index(drop=True)
best = comparison.model.iloc[0]
display(comparison.round(3))
print(f"Best model: {best}")

# COMMAND ----------

# MAGIC %md ## 3 · Evaluation charts

# COMMAND ----------

fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
for name, p in preds.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    ax[0].plot(fpr, tpr, color=COLORS[name], lw=2, label=f"{name} (AUC {roc_auc_score(y_te, p):.3f})")
    pr, rc, _ = precision_recall_curve(y_te, p)
    ax[1].plot(rc, pr, color=COLORS[name], lw=2, label=f"{name} (AP {average_precision_score(y_te, p):.3f})")
    fr, mp = calibration_curve(y_te, p, n_bins=10, strategy="quantile")
    ax[2].plot(mp, fr, "o-", color=COLORS[name], lw=2, ms=4, label=name)
ax[0].plot([0, 1], [0, 1], "--", color="grey", lw=1); ax[0].set(title="ROC curve", xlabel="False positive rate", ylabel="True positive rate")
ax[1].axhline(y_te.mean(), ls="--", color="grey", lw=1); ax[1].set(title="Precision–recall curve", xlabel="Recall", ylabel="Precision")
ax[2].plot([0, 1], [0, 1], "--", color="grey", lw=1); ax[2].set(title="Calibration (predicted vs actual)", xlabel="Predicted probability", ylabel="Actual late rate")
for a in ax: a.legend(fontsize=8, loc="lower right" if a is not ax[1] else "upper right")
fig.suptitle(f"Test period Feb–Jul 2026 · {len(test)/1000:,.0f}K American turns", fontsize=12, x=0.01, ha="left")
plt.tight_layout(); plt.show()

# Lift by risk decile: of the turns each model ranks riskiest, how many really departed 15+ late?
dec_rows = []
for name, p in preds.items():
    d = pd.DataFrame({"p": p, "y": y_te})
    d["decile"] = pd.qcut(d.p.rank(method="first"), 10, labels=range(1, 11)).astype(int)   # 10 = riskiest
    g = d.groupby("decile").agg(turns=("y", "size"), actual_late_rate=("y", "mean"), avg_predicted=("p", "mean")).reset_index()
    g["model"] = name; dec_rows.append(g)
deciles = pd.concat(dec_rows)

fig2, ax2 = plt.subplots(1, 2, figsize=(16, 4.6))
w = 0.27
for i, name in enumerate(preds):
    g = deciles[deciles.model == name]
    ax2[0].bar(g.decile + (i - 1) * w, g.actual_late_rate, w, color=COLORS[name], label=name)
ax2[0].axhline(y_te.mean(), ls="--", color="grey", lw=1)
ax2[0].set(title="Actual late rate by predicted-risk decile (10 = riskiest)", xlabel="Risk decile", ylabel="Share departing 15+ late")
ax2[0].yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0%}")); ax2[0].legend(fontsize=8)

best_pipe = models[best]
samp = X_te.sample(50_000, random_state=1)
imp = permutation_importance(best_pipe, samp, y_te[X_te.index.get_indexer(samp.index)],
                             scoring="roc_auc", n_repeats=3, random_state=1)
importance = pd.DataFrame({"feature": FEATURES, "auc_drop": imp.importances_mean}).sort_values("auc_drop")
ax2[1].barh(importance.feature, importance.auc_drop, color=COLORS[best])
ax2[1].set(title=f"What drives the {best} model (drop in AUC when a feature is shuffled)", xlabel="AUC drop")
plt.tight_layout(); plt.show()

# Confusion matrix of the best model at its best-F1 threshold
t = float(comparison.loc[0, "best_threshold"])
cm = confusion_matrix(y_te, preds[best] >= t)
fig3, ax3 = plt.subplots(figsize=(4.8, 4))
ax3.imshow(cm, cmap="Blues"); ax3.grid(False)
for (r, c), v in np.ndenumerate(cm):
    ax3.text(c, r, f"{v:,}", ha="center", va="center", color="white" if v > cm.max() / 2 else "black", fontsize=11)
ax3.set(xticks=[0, 1], yticks=[0, 1], xticklabels=["On time", "Late 15+"], yticklabels=["On time", "Late 15+"],
        xlabel="Predicted", ylabel="Actual", title=f"{best} · threshold {t:.2f}")
plt.tight_layout(); plt.show()

# COMMAND ----------

# MAGIC %md ## 4 · Log charts, register the champion, export for Tableau

# COMMAND ----------

with mlflow.start_run(run_id=runs[best]):
    mlflow.log_figure(fig, "charts/roc_pr_calibration.png")
    mlflow.log_figure(fig2, "charts/lift_and_importance.png")
    mlflow.log_figure(fig3, "charts/confusion_matrix.png")
    mlflow.log_table(importance, "feature_importance.json")

mv = mlflow.register_model(f"runs:/{runs[best]}/model", MODEL_NAME)
MlflowClient().set_registered_model_alias(MODEL_NAME, "champion", mv.version)
print(f"Registered {MODEL_NAME} version {mv.version} as @champion ({best})")

# Scored test set from the champion, by hub and risk band (for Tableau and Genie)
scored = test[["flight_date", "hub", "in_origin", "out_dest", "dep_hour", "sched_turn_min", "inbound_arr_delay_min"]].copy()
scored["actual_d15"] = y_te
scored["delay_risk"] = preds[best]
scored["risk_band"] = pd.cut(scored.delay_risk, [0, .2, .4, .6, 1], include_lowest=True,
                             labels=["1 Low (<20%)", "2 Medium (20-40%)", "3 High (40-60%)", "4 Very high (60%+)"]).astype(str)
spark.createDataFrame(scored).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{CATALOG}.{SCHEMA}.gold_turn_risk_scores")
spark.createDataFrame(comparison).write.mode("overwrite").option("overwriteSchema", True).saveAsTable(f"{CATALOG}.{SCHEMA}.gold_model_comparison")

risk_by_hub = (scored.groupby(["hub", "risk_band"])
                     .agg(turns=("actual_d15", "size"), avg_predicted_risk=("delay_risk", "mean"),
                          actual_late_rate=("actual_d15", "mean")).reset_index())
roc_pts = []
for name, p in preds.items():
    fpr, tpr, _ = roc_curve(y_te, p)
    keep = np.unique(np.linspace(0, len(fpr) - 1, 200).astype(int))       # 200 points per curve is plenty for Tableau
    roc_pts.append(pd.DataFrame({"model": name, "false_positive_rate": fpr[keep], "true_positive_rate": tpr[keep]}))
exports = {
    "ml_model_comparison.csv": comparison,
    "ml_decile_lift.csv": deciles,
    "ml_feature_importance.csv": importance.sort_values("auc_drop", ascending=False),
    "ml_roc_curves.csv": pd.concat(roc_pts),
    "ml_risk_by_hub_band.csv": risk_by_hub,
}
with zipfile.ZipFile("/tmp/aa_hub_ops_ml.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for fname, df in exports.items():
        df.to_csv(f"/tmp/{fname}", index=False); z.write(f"/tmp/{fname}", fname)
shutil.copy("/tmp/aa_hub_ops_ml.zip", f"{OUT}/aa_hub_ops_ml.zip")
for f in ("roc_pr_calibration", "lift_and_importance", "confusion_matrix"):
    {"roc_pr_calibration": fig, "lift_and_importance": fig2, "confusion_matrix": fig3}[f].savefig(f"{OUT}/{f}.png", bbox_inches="tight", dpi=150)
print(f"Exported to {OUT}: aa_hub_ops_ml.zip + 3 chart PNGs")
display(risk_by_hub.round(3))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({"best_model": best, "comparison": comparison.round(4).to_dict("records"),
                                  "top_features": importance.sort_values("auc_drop", ascending=False).head(5).round(4).to_dict("records")},
                                 default=float))
