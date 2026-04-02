"""
run_all.py  —  One-shot pipeline runner.
Run:  python run_all.py
Generates data → ETL → models → confirms dashboard is ready.
"""
import subprocess, sys, os
from pathlib import Path

ROOT = Path(__file__).parent
steps = [
    ("Generate synthetic data", [sys.executable, str(ROOT/"src/generate_data.py")]),
    ("Run ETL pipeline",        [sys.executable, str(ROOT/"src/etl.py")]),
    ("Run forecast model",      [sys.executable, str(ROOT/"src/models/forecast.py")]),
    ("Run churn model",         [sys.executable, str(ROOT/"src/models/churn.py")]),
    ("Run profitability model", [sys.executable, str(ROOT/"src/models/profitability.py")]),
]

def run_step(label, cmd):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    result = subprocess.run(cmd, cwd=str(ROOT))
    if result.returncode != 0:
        print(f"\n❌  Failed at: {label}")
        sys.exit(1)
    print(f"✓  {label} complete")

if __name__ == "__main__":
    print("\n🚀  Olist Analytics — Full Pipeline")
    for label, cmd in steps:
        run_step(label, cmd)
    print("\n" + "="*55)
    print("  ✅  All steps complete!")
    print("="*55)
    print("\nLaunch dashboard:")
    print("  streamlit run app/streamlit_app.py\n")
