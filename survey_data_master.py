"""
Master script to execute the entire survey data processing workflow.
Executes the following steps in order:
1. fill_missing_coordinates
2. survey_summary
3. survey_data_processor
4. convert_to_bw
5. image_collector
6. selective_training_dataset_generator

- To ensure the input of selective_training_dataset_generator.py, is executed interactively pty.spawn is used.
  The pty library might cause some malfunctions in e. g. windows, so it is recommended to run this script in a Unix-like environment (Linux or macOS).

- This script does not yet include the code to analyze the forms-survey results, which is done in forms_survey_summary.py.
"""
import sys
import pty
import subprocess

def run_script(script_name, interactive=False):
    """Runs a script by its name, optionally in interactive mode. If interactive is True, it uses pty.spawn to run the script.
    Otherwise, it uses subprocess.run to execute the script.
    this is useful for scripts that require user input or interaction, such as the selective_training_dataset_generator.py script."""

    print(f"\n--- Running {script_name} ---")
    if interactive:
        pty.spawn([sys.executable, script_name])
    else:
        result = subprocess.run([sys.executable, script_name])
        if result.returncode != 0:
            print(f"Error while running {script_name}")
        else:
            print(f"{script_name} completed successfully.")

if __name__ == "__main__":
    scripts = [
        #("fill_missing_coordinates.py", False),
        #("survey_summary.py", False),
        #("forms_survey_summary.py", False)
        #("survey_data_processor.py", False),
        #("convert_to_bw.py", False),
        #("image_collector.py", False),
        ("selective_training_dataset_generator.py", True) #can be activated if needed
    ]
    for script, interactive in scripts:
        run_script(script, interactive)
    print("\nAll processing steps completed.")