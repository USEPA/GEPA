"""
Name:                   task_livestock_enteric_fermentation.py
Date Last Modified:     2025-01-30
Authors Name:           Andrew Burnette (RTI International)
Purpose:                Mapping of Livestock Enteric Fermentation emissions to Year,
                        Month, State, County, emissions format
gch4i_name:             3A_enteric_fermentation, 3B_manure_management
Input Files:            - {ghgi_data_dir_path}/3A_enteric_fermentation/
                            Gridded Methane - Enteric total emissions by
                            County_v2_17Sept2024.xlsx
                        - {ghgi_data_dir_path}/3B_manure_management/
                            Gridded Methane - Manure emissions by
                            County_v1_17Sept2024.xlsx
Output Files:           - {emi_data_dir_path}/
                            enteric_fermentation_beef_emi.csv
                            enteric_fermentation_bison_emi.csv
                            enteric_fermentation_cattle_emi.csv
                            enteric_fermentation_dairy_emi.csv
                            enteric_fermentation_goats_emi.csv
                            enteric_fermentation_horses_emi.csv
                            enteric_fermentation_mules_emi.csv
                            enteric_fermentation_onfeed_emi.csv
                            enteric_fermentation_sheep_emi.csv
                            enteric_fermentation_swine_emi.csv
                            manure_management_beef_emi.csv
                            manure_management_bison_emi.csv
                            manure_management_broilers_emi.csv
                            manure_management_chickens_emi.csv
                            manure_management_dairy_emi.csv
                            manure_management_goats_emi.csv
                            manure_management_horses_emi.csv
                            manure_management_layers_emi.csv
                            manure_management_mules_emi.csv
                            manure_management_pullets_emi.csv
                            manure_management_sheep_emi.csv
                            manure_management_swine_emi.csv
                            manure_management_turkeys_emi.csv
                            manure_management_cattle_emi.csv
                            manure_management_onfeed_emi.csv
NOTES:                  - NFK 2025.05.30: I have reworked this file as the original
                        approach was not appropriately pickup up all the GHGI animal
                        names to aggregate the emissions. This new code will now throw
                        an error if the name is not appropriately crosswalked. There are
                        reference crosswalk dictionaries for both the EF and MM sources
                        for reference, but the code still uses the data found in the
                        data guide excel file.
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
import ast
import re
from pathlib import Path
from typing import Annotated

import pandas as pd
import pytask
from pytask import Product

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year,
)
from gch4i.utils import tg_to_kt

# %% Step 1. Create Function
ef_crosswalk_dict = {
    "beef_NOF_bull": "beef",
    "beef_NOF_cow": "beef",
    "beef_NOF_heifers": "cattle",
    "beef_NOF_steer": "cattle",
    "beef_OF_heifers": "onfeed",
    "beef_OF_steer": "onfeed",
    "calf_NOF_beef": "beef",
    "calf_NOF_dairy": "dairy",
    "dairy_cow": "dairy",
    "dairy_heifers": "dairy",
    "Bison": "bison",
    "Goats": "goats",
    "Horses": "horses",
    "Mules": "mules",
    "Sheep": "sheep",
    "swine_180": "swine",
    "swine_50_119": "swine",
    "swine_50": "swine",
    "swine_breeding": "swine",
    "swine_120_179": "swine",
}

mm_crosswalk_dict = {
    "beef_NOF_bull": "Beef",
    "beef_NOF_cow": "Beef",
    "beef_NOF_heifers": "Cattle",
    "beef_NOF_steer": "Cattle",
    "beef_OF_heifers": "OnFeed",
    "beef_OF_steer": "OnFeed",
    "bison": "Bison",
    "calf_NOF_beef": "Beef",
    "calf_NOF_dairy": "Dairy",
    "dairy_cow": "Dairy",
    "dairy_heifers": "Dairy",
    "goats": "Goats",
    "horses": "Horses",
    "mules": "Mules",
    "poultry_broilers": "Broilers",
    "poultry_chickens": "Chickens",
    "poultry_layers": "Layers",
    "poultry_pullets": "Pullets",
    "poultry_turkeys": "Turkeys",
    "sheep": "Sheep",
    "swine_120_179": "Swine",
    "swine_180": "Swine",
    "swine_50": "Swine",
    "swine_50_119": "Swine",
    "swine_breeding": "Swine",
}

# %% STEP 2. Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name in gch4i_data_guide_v3.xlsx, emi_proxy_mapping sheet
source_name_ef = "3A_enteric_fermentation"
source_name_mm = "3B_manure_management"
# Data Guide Directory
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
# Read and query for the source name (ghch4i_name)
proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name_ef}' | gch4i_name == '{source_name_mm}'"
)

# Initialize the emi_parameters_dict
emi_parameters_dict = {}
# Loop through the proxy data and store the parameters in the emi_parameters_dict
for group_name, group_data in proxy_data.groupby("gch4i_name"):
    file_name = group_data.file_name.iloc[0]
    file_path = ghgi_data_dir_path / group_name / file_name
    short_group_name = group_name.split("_", maxsplit=1)[-1]
    sheet_params = ast.literal_eval(group_data.add_params.iloc[0])["arguments"]
    emi_dict = {}
    output_paths = []
    for emi_row in group_data.itertuples():
        proxy_name = emi_row.Subcategory2.strip().casefold()
        # Create output path
        output_path = emi_data_dir_path / f"{short_group_name}_{proxy_name}_emi.csv"
        output_paths.append(output_path)
        emi_name_list = ast.literal_eval(emi_row.add_params)["substrings"]
        for emi_name in emi_name_list:
            emi_dict[emi_name] = proxy_name
    emi_dict

    emi_parameters_dict[group_name] = dict(
        input_path=file_path,
        sheet_params=sheet_params,
        crosswalk_dict=emi_dict,
        output_paths=output_paths,
    )
emi_parameters_dict

# %% STEP 3. Create Pytask Function and Loop


def task_livestock_emi(
    input_path: Path,
    sheet_params: list[str],
    crosswalk_dict: dict,
    output_paths: Annotated[list[Path], Product],
):
    print("reading input file:", input_path.name)
    # Read the input file
    in_df = pd.read_excel(
        input_path,
        sheet_name=sheet_params[0],  # Sheet name
        skiprows=sheet_params[1],  # Skip Rows
    )
    print("done reading input file:", input_path.name)

    # we now process the file at once instead of doing for each animal type
    print("processing data...")
    emi_df = in_df.copy()
    emi_df = emi_df.iloc[:, 1:]
    emi_df.columns.values[5:] = list(range(min_year, max_year + 1))
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .rename(columns={"state": "state_code"})
        # Filter for specific animal category
        .dropna(subset=["state_code", "county", "fips", "month", "animal"], how="all")
        .set_index(["state_code", "county", "fips", "month", "animal"])
        # Convert NA to 0 & Drop states with no data
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        # Melt the data: unique state/county/fips/month
        .assign(
            animal=lambda df: df["animal"].replace(crosswalk_dict)
        )  # Crosswalk animal names
        .melt(
            id_vars=["state_code", "county", "fips", "month", "animal"],
            var_name="year",
            value_name="ch4_tg",
        )
        # Convert tg to kt
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # Ensure only years between min_year and max_year are included
        .query("year.between(@min_year, @max_year)")
        # Ensure state/county/fips/year/month grouping is unique
        # Fips kept in due to different counties having the same name
        .groupby(["state_code", "county", "fips", "year", "month", "animal"])[
            "ghgi_ch4_kt"
        ]
        .sum()
        .reset_index()
        .sort_values(by=["fips", "year", "month"])
    )

    print("writing output files...")
    for animal, data in emi_df.groupby("animal"):
        animal = animal.lower()
        print(f"Processing {animal}")
        # this sort of backs into making sure that the animal name in the dataframe as
        # assigned from the crosswalk_dict matches what we expect for the output file.
        # If the crosswalk did not work or an animal name is not accounted for this will
        # throw an error.
        try:
            output_path = [x for x in output_paths if animal in x.name][0]
        except IndexError:
            raise ValueError(
                f"Animal '{animal}' not found in crosswalk_dict. "
                "Please check the crosswalk dictionary."
            )
        data.drop(columns="animal").to_csv(output_path, index=False)
        print(f"Saved to {output_path.name}\n")


# %%
sesh = pytask.build(
    tasks=[task_livestock_emi(**kwargs) for kwargs in emi_parameters_dict.values()],
    marker_expression="mark.persist",
    dry_run=True,
)
sesh
# %%
