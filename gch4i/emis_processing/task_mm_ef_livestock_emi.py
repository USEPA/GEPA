"""
Name:                   task_livestock_enteric_fermentation.py
Date Last Modified:     2025-01-30
Authors Name:           Andrew Burnette (RTI International)
Purpose:                Mapping of Livestock Enteric Fermentation emissions to Year,
                            Month, State, County, emissions format
gch4i_name:             3A_enteric_fermentation
Input Files:            - {ghgi_data_dir_path}/3A_enteric_fermentation/
                            Gridded Methane - Enteric total emissions by
                            County_v2_17Sept2024.xlsx
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
"""

import ast
import re

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
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

param_dict = {
    "3A_enteric_fermentation": {
        "input_path": ghgi_data_dir_path
        / "3A_enteric_fermentation"
        / "Gridded Methane - Enteric total emissions by County_v2_17Sept2024.xlsx",
        "sheet_params": [
            "Gridded_Methane___Enteric_total",
            2,
        ],
        "crosswalk_dict": ef_crosswalk_dict,
        "output_paths": [
            emi_data_dir_path / f"enteric_fermentation_{animal}_emi.csv"
            for animal in ef_crosswalk_dict.values()
        ],
    },
    "3B_manure_management": {
        "input_path": ghgi_data_dir_path
        / "3B_manure_management"
        / "Gridded Methane - Manure emissions by County_v1_17Sept2024.xlsx",
        "sheet_params": [
            "Gridded_Methane___Manure_total",
            2,
        ],
        "crosswalk_dict": mm_crosswalk_dict,
        "output_paths": [
            emi_data_dir_path / f"manure_management_{animal}_emi.csv"
            for animal in mm_crosswalk_dict.values()
        ],
    },
}


# %% STEP 3. Create Pytask Function and Loop


def task_livestock_emi(
    input_path: Path,
    sheet_params: list[str],
    crosswalk_dict: dict,
    output_paths: Annotated[list[Path], Product],
):

    in_df = pd.read_excel(
        input_path,
        sheet_name=sheet_params[0],  # Sheet name
        skiprows=sheet_params[1],  # Skip Rows
    )

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

    for animal, data in emi_df.groupby("animal"):
        animal = animal.lower()
        output_path = [x for x in output_paths if animal in x.name][0]
        print(f"Processing {animal}")
        # output_path = emi_data_dir_path / f"enteric_fermentation_{animal}_emi.csv"
        data.drop(columns=["animal"]).to_csv(output_path, index=False)
        print(f"Saved to {output_path.name}\n")


# %%
sesh = pytask.build(
    tasks=[task_livestock_emi(**kwargs) for kwargs in param_dict.values()],
    marker_expression="persist",
    dry_run=True,
)
sesh
# %%
ef_results = []
for out_path in output_paths:
    ef_results.append(pd.read_csv(out_path).assign(animal=out_path.name.split("_")[-2]))
# %%
ef_results_df = pd.concat(ef_results)
ef_results_df
# %%
ef_results_by_year = (
    ef_results_df.groupby(["year", "animal"])["ghgi_ch4_kt"]
    .sum()
    .reset_index()
    .sort_values(by=["animal", "year"])
)
ef_results_by_year
# %%

emi_total_df = (
    emi_df.groupby(["animal", "year"])["ghgi_ch4_kt"]
    .sum()
    .reset_index()
    .sort_values(by=["animal", "year"])
)
emi_total_df
# %%
animal
# %%
data
# %%
bison_df = pd.read_csv(
    Path(
        "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/emis/enteric_fermentation_bison_emi.csv"
    )
)
# %%
bison_df
# %%
emi_df.query("animal == 'bison' & county == 'ALEUTIAN ISLANDS'").sort_values(
    by=["year", "month"]
)
# %%
emi_df
# %%
in_df.query("animal == 'Bison'")["2012County"].sum() * tg_to_kt
# %%
bison_df.groupby("year")["ghgi_ch4_kt"].sum()
# %%
emi_df.query("animal == 'bison' & year == 2012")["ghgi_ch4_kt"].sum()
# %%
for group_name, group_data in proxy_data.groupby("gch4i_name"):
    input_path = Path(group_data["file_name"].values[0])
    sheet_params = ast.literal_eval(group_data["add_params"].values[0])
# %%
