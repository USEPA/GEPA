"""
Name:                  3F4 Field Burning Emissions
Date Last Modified:    2025-01-21
Authors Name:          Nick Kruskamp (RTI International)
Purpose:               Clean and standardized field burning emissions data
Input Files:           - gch4i_data_guide_v3.xlsx
                       - {V3_DATA_PATH}/ghgi/3F4_fbar/FBAR_90-22_State.xlsx.
Output Files:          - {emi_data_dir_path}/barley_emi.csv
                                            /chickpeas_emi.csv
                                            /cotton_emi.csv
                                            /drybeans_emi.csv
                                            /grasshay_emi.csv
                                            /leghay_emi.csv
                                            /lentils_emi.csv
                                            /maize_emi.csv
                                            /oats_emi.csv
                                            /other_grains_emi.csv
                                            /peanuts_emi.csv
                                            /peas_emi.csv
                                            /potatoes_emi.csv
                                            /rice_emi.csv
                                            /sorghum_emi.csv
                                            /soybeans_emi.csv
                                            /sugarbeets_emi.csv
                                            /sugarcane_emi.csv
                                            /sunflower_emi.csv
                                            /tobacco_emi.csv
                                            /vegetables_emi.csv
                                            /wheat_emi.csv
"""

# %% Import Libraries
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from pytask import Product, mark, task

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year
)
from gch4i.utils import tg_to_kt


# %% Initialize Parameters
"""
This section initializes the parameters for the task and stores them in the
emi_parameters_dict.

The parameters are read from the emi_proxy_mapping sheet of the gch4i_data_guide_v3.xlsx
file. The parameters are used to create the pytask task for the emi.
"""
# gch4i_name to filter the data guide
source_name = "3F4_fbar"
# Data Guide Dictionary
proxy_file_path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"
# Read and query for the source name (gch4i_name)
proxy_data = pd.read_excel(proxy_file_path, sheet_name="emi_proxy_mapping").query(
    f"gch4i_name == '{source_name}'"
)

# Initialize the emi_parameters_dict
emi_parameters_dict = {}
# Loop through the proxy data and store the parameters in the emi_parameters_dict
for emi_name, data in proxy_data.groupby("emi_id"):
    emi_parameters_dict[emi_name] = {
        "input_path": ghgi_data_dir_path / source_name / data.file_name.iloc[0],
        "source_list": data.gch4i_source.to_list(),
        "output_path": emi_data_dir_path / f"{emi_name}.csv",
    }


# %% Create Pytask Function and Loop
for _id, _kwargs in emi_parameters_dict.items():

    @mark.persist
    @task(id=_id, kwargs=_kwargs)
    def task_field_burning_emi(
        input_path: Path,
        source_list: str,
        output_path: Annotated[Path, Product],
    ) -> None:
        """Process field burning emissions data for specific crops.

        Args:
            input_path (Path): Path to the input data file.
            source_list (str): List of source names to extract from the data.
            output_path (Annotated[Path, Product]): Path to the output data file.

        Output file contains:
            - state_code (str): State Abbreviation
            - year (int): Year of the emissions data
            - ghgi_ch4_kt (float): CH4 emissions in kilotonnes
        """

        # Clean the source list
        source_list = [x.strip().casefold() for x in source_list]
        # Define years of interest
        year_list = [str(x) for x in list(range(min_year, max_year + 1))]

        emi_df = (
            # read in the data
            pd.read_excel(
                input_path,
                sheet_name="InvDB",
                skiprows=15,
                # nrows=115,
                # usecols="A:BA",
            )
            # name column names lower
            .rename(columns=lambda x: str(x).lower())
            # format the names of emission source group strings
            .assign(
                ghgi_source=lambda df: df["subcategory3"]
                .replace("", np.nan)
                .fillna(df["subcategory2"])
                .replace("", np.nan)
                .fillna(df["subcategory1"])
                .str.replace(" ", "_")
                .str.lower()
            )
            .dropna(subset="ghgi_source")
            # rename the location column to what we need
            .rename(columns={"georef": "state_code"})
            # get just CH4 emissions, get only the emissions of our ghgi group
            .query("(ghg == 'CH4') & (ghgi_source.isin(@source_list))")
            # get just the columns we need
            .filter(
                items=["state_code"] + year_list,
                axis=1,
            )
            .rename(columns={"georef": "state_code"})
            .set_index("state_code")
            # Replace NA values with 0
            .replace(0, pd.NA)
            .apply(pd.to_numeric, errors="coerce")
            .dropna(how="all")
            .fillna(0)
            .reset_index()
            # make the table long by state/year
            .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
            # Convert units to kt
            .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
            # Drop tg units column
            .drop(columns=["ch4_tg"])
            # Ensure correct data types
            .astype({"year": int, "ghgi_ch4_kt": float})
            .fillna({"ghgi_ch4_kt": 0})
            # get only the years we need
            .query("year.between(@min_year, @max_year)")
            # calculate a single value for each state/year
            # NOTE: applies when more than 1 source are being combined together.
            # otherwise has no effect.
            .groupby(["state_code", "year"])["ghgi_ch4_kt"]
            .sum()
            .reset_index()
        )
        # write out data to output_path
        emi_df.to_csv(output_path, index=False)
