"""
Name:                   task_iron_steel_emi.py
Date Last Modified:     2024-12-12
Authors Name:           C. COxen
Purpose:                Mapping of iron and steel emissions to State, Year, emissions
                        format
gch4i_name:             2C1_iron_and_steel
Input Files:            - 2C1_iron_and_steel/State_Iron-Steel_1990-2022.xlsx
Output Files:           - iron_steel_emi.csv
Notes:
"""

from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, mark, task

from gch4i.config import (  # noqa
    emi_data_dir_path,
    ghgi_data_dir_path,
)
from gch4i.utils import tg_to_kt


@mark.persist
@task(id="iron_steel_emi")
def task_get_iron_and_steel_inv_data(
    input_path: Path = (
        ghgi_data_dir_path / "2C1_iron_and_steel/State_Iron-Steel_1990-2022.xlsx"
    ),
    output_path: Annotated[Path, Product] = emi_data_dir_path / "iron_steel_emi.csv",
) -> None:
    """
    Read in the iron and steel data from the GHGI
    """
    emi_df = (
        # read in the data
        pd.read_excel(
            input_path,
            sheet_name="InvDB",
            skiprows=15,
            nrows=457,
            usecols="A:BA",
        )
        # name column names lower
        # drop columns we don't need
        .drop(
            columns=[
                "Data Type",
                "Sector",
                "Subsector",
                "Category",
                # "Subcategory1",
                "Subcategory2",
                "Subcategory3",
                "Subcategory4",
                "Subcategory5",
                "Carbon Pool",
                "Fuel1",
                "Fuel2",
                # "GeoRef",
                "Exclude",
                "CRT Code",
                "ID",
                "Sensitive (Y or N)",
                "Units",
                # "GHG",
                "GWP",
            ]
        )
        # filter on Sinter because it's the only emission type with CH4 emission values
        .query("Subcategory1 == 'Sinter Production'")
        .drop(columns="Subcategory1")
        .rename(columns=lambda x: str(x).lower())
        # get just methane emissions
        .query("ghg == 'CH4'")
        # remove that column
        .drop(columns="ghg")
        # set the index to state
        .rename(columns={"georef": "state_code"})
        .query("state_code != 'National'")
        .set_index("state_code")
        # covert "NO" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        # drop states that have all nan values
        .dropna(how="all")
        # reset the index state back to a column
        .reset_index()
        # make the table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ghgi_ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ghgi_ch4_kt": float})
        .fillna({"ghgi_ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
    )
    emi_df.to_csv(output_path, index=False)
