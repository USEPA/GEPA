"""
Name:                   task_mobile_comb_emi.py
Date Last Modified:     2025-01-30
Authors Name:           Andrew Burnette (RTI International)
Purpose:                Mapping of mobile combustion emissions
gch4i_name:             1A_mobile_combustion
Input Files:            - {ghgi_data_dir_path}/1A_mobile_combustion/
                            Mobile non-CO2 InvDB State Breakout_2022.xlsx
                            Mobile Dataframe 11.24.2023_proxied_DC fixed.xlsx
Output Files:           - {emi_data_dir_path}/
                            emi_passenger_cars.csv
                            emi_light.csv
                            emi_heavy.csv
                            emi_all_roads.csv
                            emi_waterways.csv
                            emi_railroads.csv
                            emi_aircraft.csv
                            emi_farm.csv
                            emi_equip.csv
                            emi_other.csv
Notes:                  - Relative proportions come from "Mobile Dataframe" data.
                        - Emissions numbers from come from "Mobile non-CO2 InvDB" data.
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------
from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import ast

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year
)

#from gch4i.utils import tg_to_kt
tg_to_kt = 1000


# Create function to grab the parameters from the excel file
def read_excel_params(file_path, subsector, emission, sheet='emi_proxy_mapping'):
    """
    Reads add_param column from gch4i_data_guide_v3.xlsx and returns a dictionary.
    """
    # Read in Excel File
    df = (pd.read_excel(file_path, sheet_name=sheet)
            .assign(
                ghgi_group=lambda x: x['Subcategory2'].str.strip().str.casefold()
            ))
    # Edit emission
    # Filter for Emissions Dictionary
    df = df.loc[df['gch4i_name'] == subsector]
    df = df.loc[df['ghgi_group'] == emission, 'add_params']
    # Convert to dictionary
    result = ast.literal_eval(df.iloc[0])
    return result

# %% STEP 1. Create Emi Mapping Functions


def get_comb_mobile_inv_data(in_path, src, params):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - Emi_Passenger
    - Emi_Light
    - Emi_Heavy
    - Emi_AllRoads
    - Emi_Waterways
    - Emi_Railroads
    - Emi_Aircraft
    - Emi_Farm
    - Emi_Equip
    - Emi_Other
    ----------

    This function takes different cleaning paths, depending on the emission source.
    Some emis group based on multiple sources, while others are based on a
    single source. The function reads in the data and returns the emissions in kt.

    ----------
    Parameters
    ----------
    in_path : str
        path to the input file
    src : str
        subcategory of interest
    params : dict
        additional parameters
    """

    ####################################################################################

    # Overwrite parameters if Emi group is made up of mutliple srcs
    # If source is in this list, then params must be overwritten, as it contributes
    # to a combined emission group.
    if src in (['motorcycles', 'passenger cars',
                'heavy-duty vehicles', 'diesel highway']):
        # Directly overwrite the params dictionary
        params = read_excel_params(proxy_file_path, source_name, src,
                                   sheet='emi_proxy_mapping')
    else:
        params = params

    # Read in the first file - InvDB data
    emi_df = (
        pd.read_excel(
            in_path[0],
            sheet_name=params["arguments"][0],  # Sheet name
            skiprows=params["arguments"][1],    # Skip rows
            index_col=None
        )
    )
    # Specify years to keep
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

    # Clean and format the data
    emi_df = (
        # Rename columns
        emi_df.rename(columns=lambda x: str(x).lower())
        .rename(columns={"georef": "state_code"})
        # Query for CH4 emissions
        .query('ghg == "CH4"')
        # Query for whether subcategory1 contains any of the sources
        .query(f'subcategory1.str.contains("{params["substrings"][0]}", regex=True)',
               engine='python')
        # Filter state code and years
        .filter(items=["state_code"] + year_list, axis=1)
        .set_index("state_code")
        # Replace NA values with 0
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        # Melt the data: unique state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        # Convert tg to kt
        .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # Ensure state/year grouping is unique
        .groupby(["state_code", "year"]).sum().reset_index()
        # Ensure only years between min_year and max_year are included
        .query("year.between(@min_year, @max_year)")
        .sort_values(["state_code", "year"])
    )
    ####################################################################################
    # End the function here if the source is in the list
    if src in (["alternative fuel highway", "farm equipment",
                "construction equipment", "other", "diesel highway"]):

        emi_df = emi_df.rename(columns={"ch4_kt": "ghgi_ch4_kt"})

        # Quick Dirty Fix: Remove HI and AK state_codes
        emi_df = emi_df.query("state_code != 'HI' and state_code != 'AK'")

        return emi_df

    ####################################################################################

    # Read in the second file - Mobile Dataframe
    emi_df2 = (
        pd.read_excel(
            in_path[1],
            sheet_name=params["arguments"][2],  # Sheet name
            index_col=None
        )
    )

    # Clean and format the data
    emi_df2 = (
        # Rename columns
        emi_df2.rename(columns=lambda x: str(x).lower())
        .drop(columns=["state"])
        .rename(columns={'state code': 'state_code'})
        # Remove CO2 from sector and get emissions for specific subcategory
        .query(f'sector.str.contains("CH4") and sector.str.contains("{params["substrings"][1]}", regex=True)', engine='python')
        .query(f'sector.str.contains("{params["substrings"][2]}", regex=True) or sector.str.endswith("{params["substrings"][1]}")')
        # Melt the data: unique state/sector
        .melt(id_vars=["state_code", "sector"],
              var_name="year",
              value_name="ch4_metric")
        .astype({"year": int})
        # Ensure only years between min_year and max_year are included
        .query("year.between(@min_year, @max_year)")
        # Pivot the data: unique state/year
        .pivot_table(index=["state_code", "year"],
                     columns="sector",
                     values="ch4_metric")
    )

    # Calculate the relative proportion of emissions
    emi_df2 = (
        emi_df2.div(emi_df2.iloc[:, 0], axis=0)
        .drop(columns=emi_df2.columns[0])
        .assign(proportion=lambda df: df.sum(axis=1))
        .iloc[:, -1]
        .reset_index()
    )

    ####################################################################################

    # Merge the two dataframes
    emi_df3 = (
        pd.merge(emi_df, emi_df2, on=["state_code", "year"], how="left")
        .assign(ghgi_ch4_kt=lambda df: df.iloc[:, 2] * df["proportion"])
    )

    # Drop unnecessary columns
    emi_df3 = emi_df3.drop(columns=["proportion", emi_df3.columns[2]])

    # Quick Dirty Fix: Remove HI and AK state_codes
    emi_df3 = emi_df3.query("state_code != 'HI' and state_code != 'AK'")

    return emi_df3


########################################################################################
########################################################################################

# %% STEP 2. Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name in gch4i_data_guide_v3.xlsx, emi_proxy_mapping sheet
source_name = "1A_mobile_combustion"
# Directory name for GHGI data
source_path = "1A_mobile_combustion"

# Data Guide Directory
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
# Read and query for the source name (ghch4i_name)
proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)

# Initialize the emi_parameters_dict
# This process is different from other emi_functions, as it reads in multiple files
emi_parameters_dict = {}
# Loop through the proxy data and store the parameters in the emi_parameters_dict
for emi_name, data in proxy_data.groupby("emi_id"):
    filenames = data.file_name.iloc[0].split(",")
    emi_parameters_dict[emi_name] = {
        "input_paths": [ghgi_data_dir_path / source_path / x for x in filenames],
        "source_list": [x.strip().casefold() for x in data.Subcategory2.to_list()],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
    }

emi_parameters_dict


# %% STEP 3. Create Pytask Function and Loop

for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_mobile_comb_emi(
        input_paths: list[Path],
        source_list: list[str],
        parameters: dict,
        output_path: Annotated[Path, Product],
    ) -> None:

        # Initialize the emi_df_list
        emi_df_list = []
        # Loop through the input paths and source list to get the emissions data
        for ghgi_group in source_list:
            individual_emi_df = get_comb_mobile_inv_data(input_paths,
                                                         ghgi_group,
                                                         parameters)
            emi_df_list.append(individual_emi_df)

        # Concatenate the emissions data and group by state and year
        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        # Save the emissions data to the output path
        emission_group_df.to_csv(output_path)


# %% TESTING
# emi_aircraft = pd.read_csv(tmp_data_dir_path / "emi_aircraft.csv")
# emi_allroads = pd.read_csv(tmp_data_dir_path / "emi_allroads.csv")
# emi_equip = pd.read_csv(tmp_data_dir_path / "emi_equip.csv")
# emi_farm = pd.read_csv(tmp_data_dir_path / "emi_farm.csv")
# emi_heavy = pd.read_csv(tmp_data_dir_path / "emi_heavy.csv")
# emi_light = pd.read_csv(tmp_data_dir_path / "emi_light.csv")
# emi_other = pd.read_csv(tmp_data_dir_path / "emi_other.csv")
# emi_passenger = pd.read_csv(tmp_data_dir_path / "emi_passenger.csv")
# emi_railroads = pd.read_csv(tmp_data_dir_path / "emi_railroads.csv")
# emi_waterways = pd.read_csv(tmp_data_dir_path / "emi_waterways.csv")

# test_aircraft = pd.read_csv(emi_data_dir_path / "emi_aircraft.csv")
# test_allroads = pd.read_csv(emi_data_dir_path / "emi_allroads.csv")
# test_equip = pd.read_csv(emi_data_dir_path / "emi_equip.csv")
# test_farm = pd.read_csv(emi_data_dir_path / "emi_farm.csv")
# test_heavy = pd.read_csv(emi_data_dir_path / "emi_heavy.csv")
# test_light = pd.read_csv(emi_data_dir_path / "emi_light.csv")
# test_other = pd.read_csv(emi_data_dir_path / "emi_other.csv")
# test_passenger = pd.read_csv(emi_data_dir_path / "emi_passenger.csv")
# test_railroads = pd.read_csv(emi_data_dir_path / "emi_railroads.csv")
# test_waterways = pd.read_csv(emi_data_dir_path / "emi_waterways.csv")


# # %% TESTING
# params = emi_parameters_dict["emi_passenger"]
# in_path = emi_parameters_dict["emi_passenger"]["input_paths"]
# src = emi_parameters_dict["emi_passenger"]["source_list"]
# params = emi_parameters_dict["emi_passenger"]["parameters"]

# # %%

# %%
