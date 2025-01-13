"""
Name:                   task_abandoned_og_wells_emi.py
Date Last Modified:     2024-09-16
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of wells emissions to State, Year, emissions format
Input Files:            - {V3_DATA_PATH}/ghgi/1B2ab_abandoned_og_wells/Abandoned_Wells_90-22_FR.xlsx
                        - gch4i_data_guide_v3.xlsx
Output Files:           - {emi_data_dir_path}/aog_gas_wells_emi.csv, {emi_data_dir_path}/aog_oil_wells_emi.csv
"""
# %% STEP 0. Load required packages and configuration settings ------------------------
from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import ast

# Import configuration variables for data paths and year ranges
from gch4i.config import (
    V3_DATA_PATH,
    tmp_data_dir_path,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year
)
# Import unit conversion factor from Tg to kT
from gch4i.utils import tg_to_kt

# %% STEP 1. Define data processing function ----------------------------------------

def get_abandoned_og_wells_inv_data(in_path, src, params):
    """
    Read and process abandoned oil and gas wells inventory data.
    
    Args:
        in_path: Path to input Excel file
        src: Source category string to filter data
        params: Dictionary containing processing parameters
            - arguments[0]: Sheet name
            - arguments[1]: Number of rows to skip
            - arguments[2]: Number of rows to read
    
    Returns:
        DataFrame with columns: [state_code, year, ghgi_ch4_kt]
    """
    # Read raw emissions data from Excel file
    emi_df = pd.read_excel(
        in_path,
        sheet_name=params["arguments"][0],  # Sheet name from InvDB
        skiprows=params["arguments"][1],  # Skip header rows (15)
        nrows=params["arguments"][2],  # Number of data rows to read (514)
        )
    
    # Generate list of years for filtering columns
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

    # Process emissions data through a series of transformations
    emi_df = (
        # Standardize column names to lowercase
        emi_df.rename(columns=lambda x: str(x).lower())
        # Create standardized source category column
        .assign(
            ghgi_source=lambda df: df["subcategory2"]
            .astype(str)
            .str.strip()
            .str.casefold()
        )
        # Rename geography column to standard name
        .rename(columns={"georef": "state_code"})
        # Filter to CH4 emissions for specified source
        .query(f"(ghg == 'CH4') & (ghgi_source == '{src}')")
        # Keep only state and year columns
        .filter(items=["state_code"] + year_list, axis=1)
        # Set state as index for pivoting
        .set_index("state_code")
        # Convert zeros to NA for proper handling
        .replace(0, pd.NA)
        # Convert all values to numeric
        .apply(pd.to_numeric, errors="coerce")
        # Remove rows with all NA values
        .dropna(how="all")
        # Replace remaining NA with zeros
        .fillna(0)
        # Reset index to make state_code a column again
        .reset_index()
        # Reshape from wide to long format
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        # Convert emissions from Tg to kT
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        # Remove intermediate Tg column
        .drop(columns=["ch4_tg"])
        # Convert year to int and emissions to float
        .astype({"year": int, "ghgi_ch4_kt": float})
        # Fill any remaining NA values with zero
        .fillna({"ghgi_ch4_kt": 0})
        # Filter to configured year range
        .query("year.between(@min_year, @max_year)")
        # Sum emissions by state and year
        .groupby(["state_code", "year"])["ghgi_ch4_kt"]
        .sum()
        .reset_index()
        )
    return emi_df


# %% STEP 2. Set up processing parameters from gch4i_data_guide_v3.xlsx -----------
# Load ghgi_data_guide and retrieve relevant cells
source_name = "1B2ab_abandoned_og_wells"
source_path = "1B2ab_abandoned_og_wells"
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)

# Create dictionary of processing parameters for each emission type
emi_parameters_dict = {}
for emi_name, data in proxy_data.groupby("emi_id"):
    emi_parameters_dict[emi_name] = {
        "input_paths": [ghgi_data_dir_path / source_path / x for x in data.file_name],
        "source_list": [x.strip().casefold() for x in data.Subcategory2.to_list()],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
    }


# %% STEP 3. Define and configure pytask processing function ---------------------

# Create task for each emission type using parameters dictionary
for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_abandoned_og_wells_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:
        """Process abandoned oil and gas wells emissions data and save to CSV."""
        # Process each input file and store results
        emi_df_list = []
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_abandoned_og_wells_inv_data(input_path,
                                                                ghgi_group,
                                                                parameters)
            emi_df_list.append(individual_emi_df)

        # Combine and aggregate all emissions by state and year
        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        emission_group_df.head()
        # Save final emissions data to CSV
        emission_group_df.to_csv(output_path)
