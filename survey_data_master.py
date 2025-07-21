"""
Master script to execute the entire survey data processing workflow.
Executes the following steps in order:
1. fill_missing_coordinates
2. survey_summary
3. survey_data_processor
4. convert_to_bw
5. image_collector
6. selective_training_dataset_generator

This script does not yet include the code to analyze the forms-survey results, which is done in forms_survey_summary.py.
"""
import subprocess
import sys

def run_script(script_name):
    print(f"\n--- Running {script_name} ---")
    result = subprocess.run([sys.executable, script_name])
    if result.returncode != 0:
        print(f"Error while running {script_name}")
    else:
        print(f"{script_name} completed successfully.")

if __name__ == "__main__":
    scripts = [
        "fill_missing_coordinates.py",
        "survey_summary.py",
        "survey_data_processor.py",
        "convert_to_bw.py",
        "image_collector.py",
        "selective_training_dataset_generator.py"
    ]
    for script in scripts:
        run_script(script)
    print("\nAll processing steps completed.")