"""
This script is the head-master script to execute the entire workflow. 
It executes the two master-sub-scripts in order to simplify the workflow:
    1. survey_data_master.py
    2. download_master.py
"""

import sys
import subprocess
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def run_subscript(script_name):
    script_path = os.path.join(BASE_DIR, script_name)
    print(f"\n--- Running {script_name} ---")
    result = subprocess.run([sys.executable, script_path])
    if result.returncode != 0:
        print(f"Error while running {script_name}")
    else:
        print(f"{script_name} completed successfully.")

if __name__ == "__main__":
    run_subscript("survey_data_master.py")
    run_subscript("download_master.py")
    print("\nAll workflow steps completed.")