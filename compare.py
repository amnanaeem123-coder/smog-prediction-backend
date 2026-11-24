"""
compare.py
Shows, hour by hour, what the NEW model and the OLD model predicted
and what the real visibility was. Only hours where both models predicted.

How to run (Anaconda Prompt):
    cd /d "C:\\Users\\Admin\\fyp project"
    python compare.py
"""

import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_PATH = os.path.join(HERE, "prediction_log.csv")
CAP = 10.0  # OpenWeatherMap never reports more than 10 km


def risk(v):
    if v < 0.5: return "CRITICAL"
    elif v < 2.0: return "HIGH"
    elif v < 5.0: return "MODERATE"
    elif v < 10.0: return "LOW"
    else: return "SAFE"


with open(LOG_PATH, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

# One row per city and hour (the latest prediction), both models present
latest = {}
for r in rows:
    if r.get("old_predicted_km"):
        key = (r["city"], r["target_time"])
        if key not in latest or r["run_time"] >= latest[key]["run_time"]:
            latest[key] = r
pairs = sorted(latest.values(), key=lambda r: (r["target_time"], r["city"]))

print(f"{'Hour':<17}{'City':<12}{'REAL':>14}{'NEW model':>20}{'OLD model':>20}   Closer")
print("-" * 92)

new_wins = old_wins = ties = new_right = old_right = checked = 0
for r in pairs:
    new = min(float(r["predicted_km"]), CAP)
    old = min(float(r["old_predicted_km"]), CAP)
    hour = r["target_time"][:16]
    if not r.get("actual_km"):
        print(f"{hour:<17}{r['city']:<12}{'waiting':>14}"
              f"{new:>8.1f} {risk(new):<11}{old:>8.1f} {risk(old):<11}   -")
        continue

    real = float(r["actual_km"])
    checked += 1
    new_err, old_err = abs(new - real), abs(old - real)
    if abs(new_err - old_err) < 0.05:
        closer = "same"
        ties += 1
    elif new_err < old_err:
        closer = "NEW"
        new_wins += 1
    else:
        closer = "OLD"
        old_wins += 1
    new_right += risk(new) == risk(real)
    old_right += risk(old) == risk(real)
    print(f"{hour:<17}{r['city']:<12}{real:>5.1f} {risk(real):<8}"
          f"{new:>8.1f} {risk(new):<11}{old:>8.1f} {risk(old):<11}   {closer}")

print("-" * 92)
if checked:
    print(f"Checked hours: {checked}")
    print(f"Closer to real visibility:  NEW {new_wins} | OLD {old_wins} | same {ties}")
    print(f"Correct risk level:         NEW {new_right} of {checked} | OLD {old_right} of {checked}")
else:
    print("No hours checked yet. A prediction is checked 1 hour after its time.")

# Also show the latest full summary (made by verify.py after every forecast)
summary_path = os.path.join(HERE, "verify_summary.txt")
if os.path.exists(summary_path):
    print("\n" + "=" * 92)
    with open(summary_path, encoding="utf-8") as f:
        print(f.read())
