"""
Name:                   coal_emi_adj.py
Date Last Modified:     2024-08-13
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of coal emissions to State, Year, emissions format
Input Files:            - Coal_90-22_FRv1-InvDBcorrection.xlsx
Output Files:           - coal_post_surf_emi.csv, coal_post_under_emi.csv,
                        coal_surf_emi.csv, coal_under_emi.csv
Notes:                  - This version of emi mapping is draft for mapping .py files
"""
# %% STEP 0. Load packages, configuration files, and local parameters ------------------
from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import ast
from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.utils import tg_to_kt

from gch4i.sector_code.emi_mapping.a_excel_dict import read_excel_params

# %% Step 1. Create Function


def get_coal_inv_data(in_path, src):
    """read in the ch4_kt values for each state
    User is required to specify the subcategory of interest:
    - coal_post_surf
    - coal_post_under
    - coal_surf
    - coal_under
    """

    subcategory_strings = read_excel_params(file_path=proxy_file_path, emission=src, sheet="testing")

    subcategory_string = subcategory_strings.get(src)
    if subcategory_string is None:
        raise ValueError("""Invalid subcategory. Please choose from ghgi_group.""")
    emi_df = pd.read_excel(
        in_path,
        sheet_name=subcategory_strings["arguments"][0],  # InvDB
        skiprows=subcategory_strings["arguments"][1],  # 15
        nrows=subcategory_strings["arguments"][2],  # 514
        )
    columns_to_keep = [col for col in emi_df.columns if not str(col).isdigit() or int(col) <= max_year]
    emi_df = emi_df[columns_to_keep]

    emi_df = (emi_df.rename(columns=lambda x: str(x).lower())
              .query("(ghg == 'CH4') & (subcategory1.str.contains(@subcategory_string[0], regex=False))")
              .filter(regex='georef|19|20')
              .rename(columns={"georef": "state_code"})
              .set_index("state_code")
              .replace(0, pd.NA)
              .apply(pd.to_numeric, errors="coerce")
              .dropna(how="all")
              .fillna(0)
              .reset_index()
              .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
              .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
              .drop(columns=["ch4_tg"])
              .astype({"year": int, "ghgi_ch4_kt": float})
              .fillna({"ghgi_ch4_kt": 0})
              .query("year.between(@min_year, @max_year)")
              .groupby(["state_code", "year"])["ghgi_ch4_kt"]
              .sum()
              .reset_index()
              )
    return emi_df


# %% STEP 2. Initialize Parameters
source_name = "coal"

proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

proxy_data = pd.read_excel(proxy_file_path, sheet_name="testing").query(
    f"gch4i_name == '{source_name}'"
)

emi_parameters_dict = {}
for emi_name, data in proxy_data.groupby("emi"):
    emi_parameters_dict[emi_name] = {
        "input_paths": [ghgi_data_dir_path / source_name / x for x in data.file_name],
        "source_list": data.ghgi_group.to_list(),
        "output_path": tmp_data_dir_path / f"{emi_name}.csv"
    }

emi_parameters_dict

# input_paths, source_list, output_path = emi_parameters_dict["coal_under_emi"].values()
# display(input_paths, source_list, output_path)


# %% STEP 3. Create Pytask Function and Loop

for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_coal_emi(
        input_paths: list[Path],
        source_list: list[str],
        output_path: Annotated[Path, Product],
    ) -> None:

        # source_list = [x.strip().casefold() for x in source_list]

        emi_df_list = []
        for input_path, ghgi_group in zip(input_paths, source_list):
            individual_emi_df = get_coal_inv_data(input_path, ghgi_group)
            emi_df_list.append(individual_emi_df)

        emission_group_df = (
            pd.concat(emi_df_list)
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        emission_group_df.head()
        emission_group_df.to_csv(output_path)

# %% TESTING RUN

# testing = get_coal_inv_data(emi_parameters_dict["coal_post_surf_emi"]["input_paths"][0], emi_parameters_dict["coal_post_surf_emi"]["source_list"][0])

# print(emi_parameters_dict["coal_post_surf_emi"]["source_list"][0])




        # %% STEP 4. Testing

        # proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
        # source_name = "coal"
        # in_path = ghgi_data_dir_path / source_name / "Coal_90-22_FRv1-InvDBcorrection.xlsx"
        # src = "Post-Mining (Surface)"

        # in_path = ghgi_data_dir_path / source_name / "Coal_90-22_FRv1-InvDBcorrection.xlsx"


        # def get_coal_inv_data2(in_path, src):
        #     """read in the ch4_kt values for each state
        #     User is required to specify the subcategory of interest:
        #     - coal_post_surf
        #     - coal_post_under
        #     - coal_surf
        #     - coal_under
        #     """

        #     subcategory_strings = read_excel_params(file_path=proxy_file_path, emission=src, sheet="testing")

        #     subcategory_string = subcategory_strings.get(src)
        #     if subcategory_string is None:
        #         raise ValueError("""Invalid subcategory. Please choose from ghgi_group.""")
        #     emi_df = (
        #         # read in the data
        #         pd.read_excel(
        #             io=inventory_workbook_path,
        #             sheet_name=subcategory_strings["arguments"][0],  # InvDB
        #             skiprows=subcategory_strings["arguments"][1],  # 15
        #             nrows=subcategory_strings["arguments"][2],  # 514
        #         )
        #     )
        #     columns_to_keep = [col for col in emi_df.columns if not str(col).isdigit() or int(col) <= max_year]
        #     emi_df = emi_df[columns_to_keep]

        #     emi_df = (emi_df.rename(columns=lambda x: str(x).lower())
        #               .query("(ghg == 'CH4') & (subcategory1.str.contains(@subcategory_string[0], regex=False))")
        #               .filter(regex='georef|19|20')
        #               .rename(columns={"georef": "state_code"})
        #               .set_index("state_code")
        #               .replace(0, pd.NA)
        #               .apply(pd.to_numeric, errors="coerce")
        #               .dropna(how="all")
        #               .fillna(0)
        #               .reset_index()
        #               .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        #               .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        #               .drop(columns=["ch4_tg"])
        #               .astype({"year": int, "ch4_kt": float})
        #               .fillna({"ch4_kt": 0})
        #               .query("year.between(@min_year, @max_year)")
        #               )
        #     return emi_df

        # %% Test


        # testing = get_coal_inv_data2(proxy_file_path, src)

        # %%

        # subcategory_strings = read_excel_params(file_path=proxy_file_path, emission=src, sheet="testing")

        # emi_df = (
        #     pd.read_excel(
        #         io=in_path,
        #         sheet_name=subcategory_strings["arguments"][0],
        #         skiprows=15,
        #         nrows=514,
        #     )
        # )

        # columns_to_keep = [col for col in emi_df.columns if not str(col).isdigit() or int(col) <= max_year]
        # emi_df = emi_df[columns_to_keep]
