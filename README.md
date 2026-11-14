# Smog Prediction Backend

Backend of the Final Year Project **AI-Based Smog Risk Prediction for Punjab Motorways**.
It predicts visibility about 5 hours ahead for Lahore, Islamabad, Faisalabad and Multan.
Each prediction is placed in one of five risk levels from SAFE to CRITICAL.
The Flutter app reads these predictions through a small live server.

Department of Computer Science, NUML Rawalpindi.
Supervisor: Ma'am Bushra Shabbir.

## Folder contents

| Item | What it is |
|---|---|
| `FYP_Week2_Preprocessing.ipynb` | Cleaning the hourly weather data |
| `FYP_Week3_EDA.ipynb` | Exploring the data with charts |
| `FYP_Week4_FeatureEngineering.ipynb` | Creating the model features |
| `FYP_Week5_ModelTraining.ipynb` | Training and comparing XGBoost, Random Forest and SVM |
| `FYP_Week5b_ForecastFix.ipynb` | Time based split fix to avoid data leakage |
| `FYP_Week7_RiskClassification.ipynb` | Turning visibility into five risk levels |
| `FYP_Week8_FoliumMap.ipynb` | Interactive motorway risk map |
| `FYP_Week9_ForecastAPI_Test.ipynb` | Testing the OpenWeatherMap forecast APIs |
| `FYP_Pollution_Merge.ipynb` | Merging pollution data with weather data |
| `FYP_Week10_PollutionMerge_Retrain.ipynb` | Final model with pollution features |
| `run_forecast.py` | Live prediction for the four cities |
| `verify.py` | Compares past predictions with the real visibility |
| `auto_run.py` | Runs the forecast once every hour |
| `prediction_log.csv` | Every live prediction with its real value |
| `verify_summary.txt` and `verify_chart.png` | Latest live test results |
| `model/` | The trained XGBoost model |
| `data/` | The final merged dataset |
| `results/` | Charts and the risk map |

## Model

* Algorithm: XGBoost regression
* Target: visibility in km
* Data: winter months of 2021 to 2025 for four cities
* Final dataset: 36,700 hourly rows with 31 features
* Split: time based 80/20 so the model never sees the future
* Rare low visibility cases (below 0.5 km) are given 5 times more weight

## Risk levels

| Visibility | Risk level |
|---|---|
| below 0.5 km | CRITICAL |
| 0.5 to 2 km | HIGH |
| 2 to 5 km | MODERATE |
| 5 to 10 km | LOW |
| 10 km and above | SAFE |

## How to run

1. Install the packages:
   ```
   pip install pandas numpy xgboost joblib requests matplotlib flask flask-cors
   ```
2. Create a file named `api_key.txt` in this folder.
   Put your own OpenWeatherMap API key inside it.
   This file is never uploaded to GitHub.
3. Set `MODEL_PATH` at the top of `run_forecast.py` to the model file in `model/`.
4. Run one forecast:
   ```
   python run_forecast.py
   ```
5. Run a forecast every hour:
   ```
   python auto_run.py
   ```
6. Check past predictions against real visibility:
   ```
   python verify.py
   ```

The forecast writes `latest_risk.json` into the Flutter app.
The app reads it through `live_server.py`, which is in the app repository.

## Note

The API key is kept out of this repository on purpose.
Each user must use their own key.
