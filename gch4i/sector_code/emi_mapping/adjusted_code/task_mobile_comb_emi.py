"""
Name:                   task_mobile_comb_emi.py
Date Last Modified:     2024-08-27
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of mobile combustion emissions
Input Files:            - Mobile non-CO2 InvDB State Breakout_2022.xlsx
                        - SIT Mobile Dataframe 5.24.2023.xlsx
Output Files:           - Emissions by State, Year for each subcategory
Notes:                  - Relative proportions from "SIT Mobile",
                        - Emissions numbers from "Mobile non-CO2 InvDB".
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

from gch4i.utils import tg_to_kt
from gch4i.sector_code.emi_mapping.a_excel_dict import read_excel_params2

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
    """

################################################################################

    # Overwrite parameters if Emi group is made up of mutliple srcs
    if src in (['motorcycles', 'passenger cars',
                'heavy-duty vehicles', 'diesel highway']):
        params = read_excel_params2(proxy_file_path, source_name, src,
                                    sheet='emi_proxy_mapping')
    else:
        params = params

    # Read in input_data[0]

    emi_df = (
        # read in the data
        pd.read_excel(
            in_path[0],
            sheet_name=params["arguments"][0],  # param
            skiprows=params["arguments"][1],    # param
            index_col=None
        )
    )
    # Initialize years
    year_list = [str(x) for x in list(range(min_year, max_year + 1))]

    emi_df = (
        emi_df.rename(columns=lambda x: str(x).lower())
        .rename(columns={"georef": "state_code"})
        .query('ghg == "CH4"')
        .query(f'subcategory1.str.contains("{params["substrings"][0]}", regex=True)',
               engine='python')  # param
        .filter(items=["state_code"] + year_list, axis=1)
        .set_index("state_code")
        .replace(0, pd.NA)
        .apply(pd.to_numeric, errors="coerce")
        .dropna(how="all")
        .fillna(0)
        .reset_index()
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        .groupby(["state_code", "year"]).sum().reset_index()
        .query("year.between(@min_year, @max_year-1)")  # Missing 2022
        .sort_values(["state_code", "year"])
    )
    ################################################################################
    # End Here if subcategory is in list
    if src in (["alternative fuel highway", "farm equipment",
                "construction equipment", "other", "diesel highway"]):

        emi_df = emi_df.rename(columns={"ch4_kt": "ghgi_ch4_kt"})

        return emi_df

    ################################################################################

    # Read in input_data[1]
    emi_df2 = (
        # read in the data
        pd.read_excel(
            in_path[1],
            sheet_name=params["arguments"][2],  # param
            index_col=None
        )
    )

    emi_df2 = (
        emi_df2.rename(columns=lambda x: str(x).lower())
        .drop(columns=["state"])
        .rename(columns={'unnamed: 0': 'state_code'})
        # Remove CO2 from sector and get emissions for specific subcategory
        .query(f'sector.str.contains("CH4") and sector.str.contains("{params["substrings"][1]}", regex=True)', engine='python')  # param
        .query(f'sector.str.contains("{params["substrings"][2]}", regex=True) or sector.str.endswith("{params["substrings"][1]}")') # param
        .melt(id_vars=["state_code", "sector"],
              var_name="year",
              value_name="ch4_metric")
        .astype({"year": int})
        .query("year.between(@min_year, @max_year)")
        .pivot_table(index=["state_code", "year"],
                     columns="sector",
                     values="ch4_metric")
    )

    emi_df2 = (
        emi_df2.div(emi_df2.iloc[:, 0], axis=0)
        .drop(columns=emi_df2.columns[0])
        .assign(proportion=lambda df: df.sum(axis=1))
        .iloc[:, -1]
        .reset_index()
    )

    ################################################################################

    emi_df3 = (
        pd.merge(emi_df, emi_df2, on=["state_code", "year"], how="left")
        .assign(ghgi_ch4_kt=lambda df: df.iloc[:, 2] * df["proportion"])
    )

    emi_df3 = emi_df3.drop(columns=["proportion", emi_df3.columns[2]])

    return emi_df3


################################################################################
################################################################################

# %% STEP 2. Initialize Parameters

source_name = "1A_mobile_combustion"
source_path = "combustion_mobile"

proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)

# Edited for multiple filenames
emi_parameters_dict = {}
for emi_name, data in proxy_data.groupby("emi_id"):
    filenames = data.file_name.iloc[0].split(",")
    emi_parameters_dict[emi_name] = {
        "input_paths": [ghgi_data_dir_path / source_path / x for x in filenames],
        "source_list": [x.strip().casefold() for x in data.Subcategory2.to_list()],
        "parameters": ast.literal_eval(data.add_params.iloc[0]),
        "output_path": emi_data_dir_path / f"{emi_name}.csv"
    }

emi_parameters_dict
# if src in cars, motorcycles then.
# if src in diesel, trucks/buses then.
# else...

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

        emi_df_list = []
        for ghgi_group in source_list:
            individual_emi_df = get_comb_mobile_inv_data(input_paths,
                                                         ghgi_group,
                                                         parameters)
            emi_df_list.append(individual_emi_df)

        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        emission_group_df.head()
        emission_group_df.to_csv(output_path)
