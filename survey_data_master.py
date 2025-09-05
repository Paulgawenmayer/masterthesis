"""
Master script to execute the entire survey data processing workflow.
Executes the following steps in order:
1. fill_missing_coordinates
2. survey_summary
3. forms_survey_summary
4. survey_data_processor
5. GML_slicer.py
6. gml_distributor.py
7. data_augmentator.py
8. convert_to_bw.py
9. image_collector.py
10. selective_training_dataset_generator.py

- To ensure the input of selective_training_dataset_generator.py, is executed interactively pty.spawn is used.
  The pty library might cause some malfunctions in e. g. windows, so it is recommended to run this script in a Unix-like environment (Linux or macOS).

- This script does not yet include the code to analyze the forms-survey results, which is done in forms_survey_summary.py.
"""
import sys
import pty
import subprocess
import os

def run_script(script_name, interactive=False, args=None):
    """Runs a script by its name, optionally in interactive mode. If interactive is True, it uses pty.spawn to run the script.
    Otherwise, it uses subprocess.run to execute the script.
    this is useful for scripts that require user input or interaction, such as the selective_training_dataset_generator.py script."""

    print(f"\n--- Running {script_name} ---")
    
    # Build command with arguments if provided
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    if interactive:
        pty.spawn(cmd)
    else:
        result = subprocess.run(cmd)
        if result.returncode != 0:
            print(f"Error while running {script_name}")
        else:
            print(f"{script_name} completed successfully.")

if __name__ == "__main__":
    # Get the absolute path to the Downloads directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    downloads_dir = os.path.join(script_dir, "training_datasets/colored")
    
    scripts = [
        #("fill_missing_coordinates.py", False), # fill missing coordinates in survey CSV files
        #("survey_summary.py", False), # create summary-charts and tables for the field survey results
        #("forms_survey_summary.py", False), # create summary-charts and tables for the forms survey results
        #("survey_data_processor.py", False), # process survey data (e. g. create training_datasets)
        #("GML_slicer.py", False), # Slice .gml-data for every single training_dataset # external script-execution not yet validated!
        #("gml_distributor.py", False), # Distribute .gml-data to training_datasets # external script-execution not yet validated!
        ("data_augmentator.py", False, ["--dir", downloads_dir]),  # Pass directory to be augmented directly
        #("convert_to_bw.py", False), # create another training_dataset with bw-images
        #("image_collector.py", False), # extract GSV-images from survey data to create selective training datasets
        #("selective_training_dataset_generator.py", True) #can be activated if needed to create selective training_datasets
    ]
    
    for item in scripts:
        if len(item) == 3:
            script, interactive, args = item
            run_script(script, interactive, args)
        else:
            script, interactive = item
            run_script(script, interactive)
            
    print("\nAll processing steps completed.")