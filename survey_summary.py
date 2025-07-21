"""
This script processes field survey_results CSV files to create a summary table of survey results.
It counts the occurrences of specific checked columns and glazing types across all survey files,
and saves the summary as a CSV file in the 'field_survey' directory.
It also generates a shapefile containing survey points with their coordinates and attributes.
Furthermore, it collects all addresses from the survey results into a separate CSV file.
If used, it is designed to be run in a directory structure where survey results are stored in subdirectories
under a base directory named 'field_survey'.
"""
import os
import pandas as pd
import shapefile  # pyshp

# this function creates a summary table of field-survey results
def create_summary_table(base_dir="field_survey", target_file="survey_results.csv"):
    # Columns to count 'checked'
    checked_columns = [
        "Aufsparrendämmung",
        "Dach saniert?",
        "Fassadendämmung",
        "Sockeldämmung",
        "Fenster fassadenbündig"
    ]
    # Column for glazing types
    glazing_column = "Verglasungstyp"
    glazing_types = ["Einfachverglasung", "Zweifachverglasung", "Dreifachverglasung"]

    checked_counts = {col: 0 for col in checked_columns}
    glazing_counts = {typ: 0 for typ in glazing_types}
    total_rows = 0

    for root, dirs, files in os.walk(base_dir):
        if target_file in files:
            file_path = os.path.join(root, target_file)
            try:
                df = pd.read_csv(file_path)
                total_rows += len(df)
                for col in checked_columns:
                    if col in df.columns:
                        checked_counts[col] += (df[col].astype(str).str.strip().str.lower() == "checked").sum()
                if glazing_column in df.columns:
                    for typ in glazing_types:
                        glazing_counts[typ] += (df[glazing_column].astype(str).str.strip() == typ).sum()
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    # Prepare summary as DataFrame
    summary_data = {"Category": [], "Count": []}
    summary_data["Category"].append("Total rows (all surveys)")
    summary_data["Count"].append(total_rows)
    for col in checked_columns:
        summary_data["Category"].append(col)
        summary_data["Count"].append(checked_counts[col])
    for typ in glazing_types:
        summary_data["Category"].append(typ)
        summary_data["Count"].append(glazing_counts[typ])
    summary_df = pd.DataFrame(summary_data)

    # Save as CSV in field_survey/survey_summary
    summary_dir = os.path.join(base_dir, "survey_summary")
    os.makedirs(summary_dir, exist_ok=True)
    summary_path = os.path.join(summary_dir, "survey_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    print(f"Summary saved to {summary_path}")


# this function generates a shapefile from field-survey results
def shapefile_output(base_dir="field_survey", target_file="survey_results.csv", shapefile_name="houses_examined"):
    # Prepare shapefile writer
    summary_dir = os.path.join(base_dir, "survey_summary")
    os.makedirs(summary_dir, exist_ok=True)
    shp_path = os.path.join(summary_dir, shapefile_name)
    w = shapefile.Writer(shp_path, shapeType=shapefile.POINT)
    # Add all required fields in English
    w.field('ID', 'N')
    w.field('Coordinates', 'C')
    w.field('Address', 'C')
    w.field('Rafter_insulation', 'C')
    w.field('Roof_renovated', 'C')
    w.field('Facade_insulation', 'C')
    w.field('Base_insulation', 'C')
    w.field('Flush_windows', 'C')
    w.field('Glazing_type', 'C')
    w.field('Comment', 'C')
    idx = 0
    for root, dirs, files in os.walk(base_dir):
        if target_file in files:
            file_path = os.path.join(root, target_file)
            try:
                df = pd.read_csv(file_path)
                if 'Koordinaten' in df.columns:
                    for _, row in df.iterrows():
                        coords = row['Koordinaten']
                        if isinstance(coords, str) and ',' in coords:
                            latlon = coords.split(',')
                            try:
                                lat = float(latlon[0].strip())
                                lon = float(latlon[1].strip())
                                w.point(lon, lat)  # shapefile expects (x, y) = (lon, lat)
                                w.record(
                                    idx,
                                    coords,
                                    row.get('Adresse', ''),
                                    row.get('Aufsparrendämmung', ''),
                                    row.get('Dach saniert?', ''),
                                    row.get('Fassadendämmung', ''),
                                    row.get('Sockeldämmung', ''),
                                    row.get('Fenster fassadenbündig', ''),
                                    row.get('Verglasungstyp', ''),
                                    row.get('Kommentar', '')
                                )
                                idx += 1
                            except Exception as e:
                                print(f"Invalid coordinates in {file_path}: {coords} ({e})")
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    w.close()
    # Write .prj file for WGS84 (EPSG:4326)
    prj_path = shp_path + ".prj"
    with open(prj_path, 'w') as prj:
        prj.write('GEOGCS["WGS 84",DATUM["WGS_1984",SPHEROID["WGS 84",6378137,298.257223563]],PRIMEM["Greenwich",0],UNIT["degree",0.0174532925199433]]')
    print(f"Shapefile saved to {shp_path}.shp (WGS84/EPSG:4326)")


# this function collects all addresses from survey results
def create_address_list(base_dir="field_survey", target_file="survey_results.csv"):
    summary_dir = os.path.join(base_dir, "survey_summary")
    os.makedirs(summary_dir, exist_ok=True)
    address_list = []
    for root, dirs, files in os.walk(base_dir):
        if target_file in files:
            file_path = os.path.join(root, target_file)
            try:
                df = pd.read_csv(file_path)
                if 'Adresse' in df.columns:
                    address_list.extend(df['Adresse'].dropna().astype(str).tolist())
            except Exception as e:
                print(f"Error processing {file_path}: {e}")
    # Write address list to CSV
    address_list_path = os.path.join(summary_dir, "address_list.csv")
    pd.DataFrame({'Address': address_list}).to_csv(address_list_path, index=False)
    print(f"Address list saved to {address_list_path}")


if __name__ == "__main__":
    create_summary_table()
    shapefile_output()
    create_address_list()
