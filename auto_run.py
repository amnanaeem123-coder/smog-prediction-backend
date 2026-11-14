"""
auto_run.py
Runs run_forecast.py once every hour.
Keep this window open. Press Ctrl+C to stop.
"""
import datetime
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))

while True:
    now = datetime.datetime.now().strftime("%d %b %H:%M")
    print("=" * 50)
    print(f"Running forecast at {now}")
    print("=" * 50)

    # Runs the same forecast script you run by hand.
    subprocess.run([sys.executable, "run_forecast.py"], cwd=HERE)

    print("\nDone. Next run in 1 hour.")
    print("Keep this window open. Press Ctrl+C to stop.\n")
    time.sleep(3600)
