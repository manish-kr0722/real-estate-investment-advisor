"""Step 3 - Model training & evaluation (classification + regression) with experiment tracking.
Uses XGBoost and MLflow when installed; otherwise falls back to scikit-learn HistGradientBoosting and a JSON run log.
Usage: python src/train.py
"""
import json, time, warnings, joblib, numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix,
                             mean_squared_error, mean_absolute_error, r2_score)
from sklearn.inspection import permutation_importance
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from features import NUM, CAT, FEATURES
warnings.filterwarnings("ignore")
SEED = 42
try:
    import xgboost as xgb; HAVE_XGB = True
except ImportError: HAVE_XGB = False
try:
    import mlflow, mlflow.sklearn; HAVE_MLFLOW = True
except ImportError: HAVE_MLFLOW = False
print(f"XGBoost: {HAVE_XGB} | MLflow: {HAVE_MLFLOW}")

df = pd.read_csv("outputs/india_housing_cleaned.csv")
X = df[FEATURES]; yc = df["Good_Investment"]; yr = df["Future_Price_5Y_Lakhs"]
Xtr, Xte, yctr, ycte, yrtr, yrte = train_test_split(X, yc, yr, test_size=.2, random_state=SEED, stratify=yc)
prep = lambda: ColumnTransformer([("num", StandardScaler(), NUM), ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT)])

clf_models = {
    "LogisticRegression": LogisticRegression(max_iter=300, n_jobs=2),
    "RandomForest": RandomForestClassifier(n_estimators=60, max_depth=14, min_samples_leaf=20, n_jobs=2, random_state=SEED),
    ("XGBoost" if HAVE_XGB else "HistGradientBoosting"): (xgb.XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.1, subsample=.8, n_jobs=2, random_state=SEED, eval_metric="logloss")
        if HAVE_XGB else HistGradientBoostingClassifier(max_iter=250, learning_rate=.1, max_depth=8, random_state=SEED)),
}
reg_models = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=60, max_depth=14, min_samples_leaf=20, n_jobs=2, random_state=SEED),
    ("XGBoost" if HAVE_XGB else "HistGradientBoosting"): (xgb.XGBRegressor(n_estimators=400, max_depth=6, learning_rate=.1, subsample=.8, n_jobs=2, random_state=SEED)
        if HAVE_XGB else HistGradientBoostingRegressor(max_iter=400, learning_rate=.1, max_depth=8, random_state=SEED)),
}

runs = []
def log_run(task, name, model, params, metrics, secs):
    runs.append(dict(task=task, model=name, params=params, metrics=metrics, train_seconds=round(secs, 1)))
    if HAVE_MLFLOW:
        mlflow.set_experiment(f"RealEstate_{task}")
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params); mlflow.log_metrics(metrics); mlflow.sklearn.log_model(model, "model")

best = {}
# ---------------- classification ----------------
for name, m in clf_models.items():
    t = time.time(); pipe = Pipeline([("prep", prep()), ("m", m)]).fit(Xtr, yctr); secs = time.time() - t
    p = pipe.predict(Xte); pr = pipe.predict_proba(Xte)[:, 1]
    met = dict(accuracy=accuracy_score(ycte, p), precision=precision_score(ycte, p), recall=recall_score(ycte, p), f1=f1_score(ycte, p), roc_auc=roc_auc_score(ycte, pr))
    met = {k: round(float(v), 4) for k, v in met.items()}
    log_run("classification", name, pipe, {"model": name, "n_features": len(FEATURES), "test_size": .2}, met, secs)
    print("CLF", name, met)
    if name not in best.get("clf", {}) and met["f1"] >= best.get("clf_f1", -1): best["clf"] = (name, pipe); best["clf_f1"] = met["f1"]; best["clf_pred"] = p
# ---------------- regression ----------------
for name, m in reg_models.items():
    t = time.time(); pipe = Pipeline([("prep", prep()), ("m", m)]).fit(Xtr, yrtr); secs = time.time() - t
    p = pipe.predict(Xte)
    met = dict(rmse=mean_squared_error(yrte, p) ** .5, mae=mean_absolute_error(yrte, p), r2=r2_score(yrte, p), mape_pct=float(np.mean(np.abs((yrte - p) / yrte)) * 100))
    met = {k: round(float(v), 4) for k, v in met.items()}
    log_run("regression", name, pipe, {"model": name, "n_features": len(FEATURES), "test_size": .2}, met, secs)
    print("REG", name, met)
    if met["rmse"] <= best.get("reg_rmse", 1e9): best["reg"] = (name, pipe); best["reg_rmse"] = met["rmse"]; best["reg_pred"] = p

# ---------------- save best + diagnostics ----------------
cn, cp = best["clf"]; rn, rp = best["reg"]
joblib.dump(cp, "models/best_classifier.joblib"); joblib.dump(rp, "models/best_regressor.joblib")
cm = confusion_matrix(ycte, best["clf_pred"]).tolist()
sub = Xte.sample(8000, random_state=1).index
imp = {}
for tag, name, pipe, y in [("clf", cn, cp, yc), ("reg", rn, rp, yr)]:
    r = permutation_importance(pipe, Xte.loc[sub], y.loc[sub], n_repeats=3, random_state=SEED, n_jobs=1,
                               scoring="f1" if tag == "clf" else "neg_root_mean_squared_error")
    s = pd.Series(r.importances_mean, index=FEATURES).sort_values(ascending=False); imp[tag] = s.head(12).round(5).to_dict()
    plt.figure(figsize=(7, 5)); s.head(12)[::-1].plot.barh(color="#4C72B0"); plt.title(f"Permutation importance - {tag} ({name})"); plt.tight_layout(); plt.savefig(f"outputs/figures/model_importance_{tag}.png", dpi=130); plt.close()
fig, ax = plt.subplots(1, 2, figsize=(11, 4.3))
ax[0].imshow(cm, cmap="Blues"); [ax[0].text(j, i, f"{v:,}", ha="center", va="center") for i, row in enumerate(cm) for j, v in enumerate(row)]
ax[0].set(title=f"Confusion matrix - {cn}", xticks=[0, 1], yticks=[0, 1], xlabel="Predicted", ylabel="Actual")
ax[1].scatter(yrte.sample(5000, random_state=1), pd.Series(best["reg_pred"], index=yrte.index).loc[yrte.sample(5000, random_state=1).index], s=3, alpha=.3)
ax[1].plot([0, 900], [0, 900], "r--"); ax[1].set(title=f"Actual vs predicted 5-yr price - {rn}", xlabel="Actual (Lakhs)", ylabel="Predicted"); plt.tight_layout(); plt.savefig("outputs/figures/model_diagnostics.png", dpi=130); plt.close()
# leakage-free baseline: how well does the classifier do with ONLY the rule-free 'raw listing' view? (documented in report)
base_acc = float(max(yc.mean(), 1 - yc.mean()))
json.dump(dict(xgboost=HAVE_XGB, mlflow=HAVE_MLFLOW, runs=runs, best_classifier=cn, best_regressor=rn, confusion_matrix=cm, importance=imp,
               majority_baseline_accuracy=round(base_acc, 4), mean_future_price=float(yr.mean()), n_train=len(Xtr), n_test=len(Xte)),
          open("outputs/model_results.json", "w"), indent=2)
print("best:", cn, rn)
