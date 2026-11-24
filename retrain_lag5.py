"""
retrain_lag5.py
Step 1 experiment: make training match live use.

The problem:
    In training, visibility_prev_reading was the visibility 1 hour before.
    In live use, run_forecast.py gives the CURRENT visibility,
    which is 5 hours before the target hour, and capped at 10 km
    because OpenWeatherMap never reports more than 10 km.

The fix:
    Retrain the same XGBoost model with the same 31 features and settings.
    Only visibility_prev_reading changes: it becomes the visibility
    5 hours before, capped at 10 km. This is exactly what live use gives.

Nothing old is changed:
    The old model stays as it is.
    The new model is saved as model/xgb_model_lag5.pkl.

What it prints (all on the same test rows):
    A. Old model, old style input  (the 77.8% from the notebook)
    B. Old model, live style input (how the old model really behaves live)
    C. New model, live style input (the experiment)

How to run (Anaconda Prompt):
    cd /d "C:\\Users\\Admin\\fyp project"
    python retrain_lag5.py
"""

import os

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(HERE, "data", "master_dataset_with_pollution.csv")
OLD_MODEL_PATH = os.path.join(HERE, "model", "xgb_model_with_pollution_final.pkl")
NEW_MODEL_PATH = os.path.join(HERE, "model", "xgb_model_lag5.pkl")
RESULT_PATH = os.path.join(HERE, "retrain_lag5_results.txt")

LEAD_HOURS = 5      # live prediction is 5 hours ahead
LIVE_CAP_KM = 10.0  # OpenWeatherMap never reports more than 10 km
LEVELS = ["CRITICAL", "HIGH", "MODERATE", "LOW", "SAFE"]
DANGER = {"CRITICAL", "HIGH", "MODERATE"}


def classify_risk(v):
    if v < 0.5: return "CRITICAL"
    elif v < 2.0: return "HIGH"
    elif v < 5.0: return "MODERATE"
    elif v < 10.0: return "LOW"
    else: return "SAFE"


def scores(name, y_true, pred):
    actual = y_true.apply(classify_risk)
    predicted = pd.Series(pred, index=y_true.index).apply(classify_risk)

    accuracy = (actual == predicted).mean()
    crit = actual == "CRITICAL"
    crit_recall = (predicted[crit] == "CRITICAL").mean() if crit.any() else float("nan")

    real_danger = actual.isin(DANGER)
    pred_danger = predicted.isin(DANGER)
    caught = (real_danger & pred_danger).sum()
    false_alarms = (~real_danger & pred_danger).sum()

    lines = [
        f"=== {name} ===",
        f"R2: {r2_score(y_true, pred):.4f} | MAE: {mean_absolute_error(y_true, pred):.3f} km"
        f" | RMSE: {np.sqrt(mean_squared_error(y_true, pred)):.3f} km",
        f"Risk level correct: {accuracy * 100:.1f}%",
        f"CRITICAL recall: {crit_recall * 100:.1f}%",
        f"Danger hours caught (MODERATE or worse): {caught} of {real_danger.sum()}"
        f" ({100 * caught / max(real_danger.sum(), 1):.1f}%)",
        f"False alarms: {false_alarms}",
        "",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------
# 1. Load the same dataset the notebook used
# ------------------------------------------------------------
df = pd.read_csv(DATA_PATH)
df["datetime"] = pd.to_datetime(df["datetime"])
df = df.sort_values("datetime").reset_index(drop=True)
print(f"Rows loaded: {len(df)}")

exclude_cols = ["visibility", "datetime", "time_gap_hours", "city_name"]
feature_cols = [c for c in df.columns if c not in exclude_cols]
print(f"Features ({len(feature_cols)}): same as the final model")

# ------------------------------------------------------------
# 2. Live style lag: visibility 5 hours before, capped at 10 km
# ------------------------------------------------------------
lookup = df.set_index(["city_name", "datetime"])["visibility"]
lookup = lookup[~lookup.index.duplicated(keep="first")]

earlier = pd.MultiIndex.from_arrays(
    [df["city_name"], df["datetime"] - pd.Timedelta(hours=LEAD_HOURS)]
)
df["lag5_live"] = lookup.reindex(earlier).values.clip(max=LIVE_CAP_KM)
print(f"Rows with a reading 5 hours before: {df['lag5_live'].notna().sum()}")

# ------------------------------------------------------------
# 3. Same time based 80/20 split as the notebook
# ------------------------------------------------------------
split_index = int(len(df) * 0.8)
train = df.iloc[:split_index]
test = df.iloc[split_index:]

# Only rows that have a 5 hour earlier reading are used
train = train[train["lag5_live"].notna()]
test = test[test["lag5_live"].notna()]
print(f"Training rows: {len(train)} | Testing rows: {len(test)}\n")

y_train = train["visibility"]
y_test = test["visibility"]

# Old style inputs (as in the notebook) and live style inputs
X_test_old_style = test[feature_cols]
X_train_live = train[feature_cols].copy()
X_train_live["visibility_prev_reading"] = train["lag5_live"]
X_test_live = test[feature_cols].copy()
X_test_live["visibility_prev_reading"] = test["lag5_live"]

# ------------------------------------------------------------
# 4. Old model on both kinds of input
# ------------------------------------------------------------
old_model = joblib.load(OLD_MODEL_PATH)
report = []
report.append(scores("A. Old model, old style input (1 hour before)", y_test,
                     old_model.predict(X_test_old_style)))
report.append(scores("B. Old model, live style input (5 hours before, max 10 km)", y_test,
                     old_model.predict(X_test_live)))

# ------------------------------------------------------------
# 5. New model: same settings, live style input
# ------------------------------------------------------------
sample_weights = np.where(y_train < 0.5, 5, 1)
new_model = xgb.XGBRegressor(
    n_estimators=100, max_depth=6, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42, n_jobs=-1,
)
new_model.fit(X_train_live, y_train, sample_weight=sample_weights)
report.append(scores("C. New model, live style input (5 hours before, max 10 km)", y_test,
                     new_model.predict(X_test_live)))

joblib.dump(new_model, NEW_MODEL_PATH)

text = "\n".join(report)
print(text)
with open(RESULT_PATH, "w", encoding="utf-8") as f:
    f.write(text)
print(f"New model saved: {NEW_MODEL_PATH}")
print(f"Results saved:   {RESULT_PATH}")
print("The old model was not changed.")
