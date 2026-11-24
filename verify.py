"""
Checks old predictions against what really happened.

How it works:
    1. Reads prediction_log.csv (made by run_forecast.py).
    2. For every prediction whose time has passed, it asks OpenWeatherMap
       for the real visibility at that hour.
    3. Writes the real value next to the prediction in the same file.
    4. Prints a summary: how often the risk level was right,
       average error in km, missed danger and false alarms.
    5. Saves verify_chart.png (predicted vs actual) for the report.

Run it any time (Anaconda Prompt):
    cd "C:\\Users\\Admin\\fyp project"
    python verify.py

run_forecast.py also runs it automatically after every forecast.

Duplicates: if the forecast was run twice for the same city and the same
forecast hour, only the latest of those predictions is counted in the
summary and the chart. The log file itself keeps every row.

Two models:
    Until 23 Nov 2026 about 22:45 the old model made the predictions.
    After that the new model (lag5) makes them. From 24 Nov the old model
    also predicts the same hours in the background. The summary shows
    the old model, the new model, and both side by side on the same hours.

Note: OpenWeatherMap reports at most 10 km visibility. So before comparing,
predictions above 10 km are also counted as 10 km. This is fair because
10 km and above are all SAFE anyway.
"""

import csv
import os
from datetime import datetime, timedelta

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(SCRIPT_DIR, "prediction_log.csv")
CHART_PATH = os.path.join(SCRIPT_DIR, "verify_chart.png")
SUMMARY_PATH = os.path.join(SCRIPT_DIR, "verify_summary.txt")
KEY_FILE = os.path.join(SCRIPT_DIR, "api_key.txt")

LOG_COLUMNS = ["run_time", "city", "target_time", "predicted_km", "predicted_risk",
               "weather_source", "actual_km", "actual_risk", "level_match", "checked_at",
               "model", "old_predicted_km", "old_predicted_risk"]

# Predictions made from this time on came from the new model
SWITCH_TIME = "2026-11-23 22:45:00"

CITY_COORDS = {
    'Lahore':     [31.5204, 74.3587],
    'Islamabad':  [33.6844, 73.0479],
    'Faisalabad': [31.4504, 73.1350],
    'Multan':     [30.1575, 71.5249]
}

DANGER = {"MODERATE", "HIGH", "CRITICAL"}
WAIT_HOURS = 1        # check a prediction only 1 hour after its time has passed
OWM_MAX_KM = 10.0     # OpenWeatherMap never reports more than 10 km


def classify_risk(v):
    if v < 0.5: return "CRITICAL"
    elif v < 2.0: return "HIGH"
    elif v < 5.0: return "MODERATE"
    elif v < 10.0: return "LOW"
    else: return "SAFE"


def load_key():
    key = os.environ.get("OWM_API_KEY", "").strip()
    if key:
        return key
    with open(KEY_FILE, "r", encoding="utf-8") as f:
        return f.read().strip()


def fetch_hours(api_key, city, start_dt):
    """Get ~20 hourly records for a city starting at start_dt.
    Returns {local hour (datetime): visibility km}."""
    lat, lon = CITY_COORDS[city]
    start_unix = int(start_dt.timestamp())
    url = (f"https://api.openweathermap.org/data/4.0/onecall/timeline/1h"
           f"?lat={lat}&lon={lon}&start={start_unix}&units=metric&appid={api_key}")
    try:
        r = requests.get(url, timeout=15)
        r.raise_for_status()
        data = r.json().get("data", [])
    except Exception as e:
        print(f"  Could not get real visibility for {city}: {str(e).replace(api_key, '***KEY***')}")
        return {}

    hours = {}
    for entry in data:
        if "visibility" not in entry:
            continue
        hour = datetime.fromtimestamp(entry["dt"]).replace(minute=0, second=0, microsecond=0)
        hours[hour] = entry["visibility"] / 1000.0  # metres -> km
    return hours


def read_log():
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_log(rows):
    with open(LOG_PATH, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in LOG_COLUMNS})


def fill_actuals(rows, api_key):
    """Add the real visibility to every prediction whose time has passed."""
    now = datetime.now()
    cache = {}  # (city, hour) -> km
    filled = 0

    for row in rows:
        if row.get("actual_km"):
            continue
        target = datetime.strptime(row["target_time"], "%Y-%m-%d %H:%M:%S")
        if target > now - timedelta(hours=WAIT_HOURS):
            continue  # too early to check

        hour = target.replace(minute=0, second=0, microsecond=0)
        key = (row["city"], hour)
        if key not in cache:
            for h, km in fetch_hours(api_key, row["city"], hour).items():
                cache[(row["city"], h)] = km
        if key not in cache:
            continue

        actual = cache[key]
        row["actual_km"] = f"{actual:.2f}"
        row["actual_risk"] = classify_risk(actual)
        predicted_capped = min(float(row["predicted_km"]), OWM_MAX_KM)
        row["level_match"] = "YES" if classify_risk(predicted_capped) == row["actual_risk"] else "NO"
        row["checked_at"] = now.strftime("%Y-%m-%d %H:%M:%S")
        filled += 1

    return filled


def unique_rows(rows):
    """Keep one prediction per city and forecast hour (the latest one)."""
    latest = {}
    for r in rows:
        key = (r["city"], r["target_time"])
        if key not in latest or r["run_time"] >= latest[key]["run_time"]:
            latest[key] = r
    return list(latest.values())


def model_of(row):
    if row.get("model"):
        return row["model"]
    return "lag5" if row["run_time"] >= SWITCH_TIME else "old"


def stats_lines(rows, key, title):
    """Summary lines for one model. key is the column with its prediction."""
    rows = [r for r in rows if r.get("actual_km") and r.get(key)]
    lines = [title]
    if not rows:
        lines.append("  Nothing checked yet.")
        return lines

    def pred(r):
        return min(float(r[key]), OWM_MAX_KM)

    def pred_risk(r):
        return classify_risk(pred(r))

    matches = sum(1 for r in rows if pred_risk(r) == r["actual_risk"])
    errors = [abs(pred(r) - float(r["actual_km"])) for r in rows]
    real_danger = [r for r in rows if r["actual_risk"] in DANGER]
    caught = sum(1 for r in real_danger if pred_risk(r) in DANGER)
    false_alarm = sum(1 for r in rows if pred_risk(r) in DANGER and r["actual_risk"] not in DANGER)

    lines.append(f"  Checked predictions:  {len(rows)}")
    lines.append(f"  Risk level correct:   {matches} of {len(rows)} ({100 * matches / len(rows):.1f}%)")
    lines.append(f"  Average error:        {sum(errors) / len(errors):.2f} km")
    lines.append(f"  Real danger hours (MODERATE or worse): {len(real_danger)}")
    lines.append(f"    caught:             {caught}")
    lines.append(f"    missed:             {len(real_danger) - caught}")
    lines.append(f"  False alarms:         {false_alarm}")
    lines.append("  By city:")
    for city in CITY_COORDS:
        c = [r for r in rows if r["city"] == city]
        if c:
            m = sum(1 for r in c if pred_risk(r) == r["actual_risk"])
            e = sum(abs(pred(r) - float(r["actual_km"])) for r in c) / len(c)
            lines.append(f"    {city:<11} {m}/{len(c)} correct, average error {e:.2f} km")
    return lines


def summarize(rows):
    total_rows = len(rows)
    unique = unique_rows(rows)
    done = [r for r in unique if r.get("actual_km")]

    old_rows = unique_rows([r for r in rows if model_of(r) == "old"])
    new_rows = unique_rows([r for r in rows if model_of(r) == "lag5"])
    paired = [r for r in new_rows if r.get("old_predicted_km")]

    lines = []
    lines.append(f"Rows in log file:     {total_rows}")
    lines.append(f"Duplicates ignored:   {total_rows - len(unique)}")
    lines.append(f"Unique predictions:   {len(unique)}")
    lines.append(f"Predictions checked:  {len(done)}")
    lines.append(f"Waiting to be checked: {len(unique) - len(done)}")
    lines.append("")
    lines += stats_lines(old_rows, "predicted_km", "OLD MODEL (until 23 Nov 22:45)")
    lines.append("")
    lines += stats_lines(new_rows, "predicted_km", "NEW MODEL lag5 (from 23 Nov 22:45)")
    lines.append("")
    lines.append("SIDE BY SIDE on exactly the same hours:")
    lines += stats_lines(paired, "predicted_km", " New model:")
    lines += stats_lines(paired, "old_predicted_km", " Old model:")
    return "\n".join(lines)


def make_chart(rows):
    done = [r for r in unique_rows(rows) if r.get("actual_km")]
    if not done:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available, chart skipped.")
        return

    fig, ax = plt.subplots(figsize=(7, 6))
    for city in CITY_COORDS:
        c = [r for r in done if r["city"] == city]
        if c:
            ax.scatter([float(r["actual_km"]) for r in c],
                       [min(float(r["predicted_km"]), OWM_MAX_KM) for r in c],
                       label=city, alpha=0.75, s=40)
    ax.plot([0, OWM_MAX_KM], [0, OWM_MAX_KM], color="gray", linestyle="--", linewidth=1, label="perfect prediction")
    for t in (0.5, 2, 5):
        ax.axvline(t, color="lightgray", linewidth=0.8)
        ax.axhline(t, color="lightgray", linewidth=0.8)
    ax.set_xlim(0, OWM_MAX_KM + 0.5)
    ax.set_ylim(0, OWM_MAX_KM + 0.5)
    ax.set_xlabel("Actual visibility (km)")
    ax.set_ylabel("Predicted visibility 5 hours ahead (km, capped at 10)")
    ax.set_title(f"Predicted vs actual visibility ({len(done)} checked predictions)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(CHART_PATH, dpi=150)
    plt.close(fig)
    print(f"Chart saved: {CHART_PATH}")


def run(with_chart=True):
    rows = read_log()
    if not rows:
        print("prediction_log.csv is empty. Run run_forecast.py first.")
        return

    filled = fill_actuals(rows, load_key())
    if filled:
        write_log(rows)
    print(f"\nChecked {filled} new predictions against real visibility.")

    summary = summarize(rows)
    print("\n" + summary)
    with open(SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write(f"Updated: {datetime.now():%Y-%m-%d %H:%M}\n\n{summary}\n")

    if with_chart:
        make_chart(rows)


if __name__ == "__main__":
    run(with_chart=True)
