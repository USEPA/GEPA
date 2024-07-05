# %%
from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task, mark

from gch4i.config import (emi_data_dir_path, ghgi_data_dir_path, max_year,
                          min_year)
from gch4i.utils import tg_to_kt


@mark.persist
@task(id="msw_landfills_emi")
def task_get_msw_landfills_inv_data(
    input_path: Path = ghgi_data_dir_path
    / "landfills/State_MSW_LF_1990-2022_LA.xlsx",
    reporting_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "msw_landfills_reporting_emi.csv",
    nonreporting_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "msw_landfills_nonreporting_emi.csv",
) -> None:
    """read in the ghgi_ch4_kt values for each state"""
    nonreporting_emi_df = pd.DataFrame()
    emi_df = (
        # read in the data
        pd.read_excel(
            input_path,
            sheet_name="InvDB",
            skiprows=15,
            nrows=58,
            usecols="A:BA",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # drop columns we don't need
        # # get just methane emissions
        .query("(ghg == 'CH4')")
        .drop(
            columns=[
                "sector",
                "category",
                "subcategory1",
                "subcategory2",
                "subcategory3",
                "subcategory4",
                "subcategory5",
                "carbon pool",
                "fuel1",
                "fuel2",
                "exclude",
                "id",
                "sensitive (y or n)",
                "data type",
                "subsector",
                "crt code",
                "units",
                "ghg",
                "gwp",
            ]
        )
        # set the index to state
        .rename(columns={"georef": "state_code"})
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
    emi_df.to_csv(reporting_emi_output_path, index=False)

    # Get non-reporting emissions by scaling reporting emissions.
    # Assume emissions are 9% of reporting emissions for 2016 and earlier.
    # Assume emissions are 11% of reporting emissions for 2017 and later.
    emi_09 = emi_df.query("year <= 2016").assign(ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt"] * 0.09)
    emi_11 = emi_df.query("year >= 2017").assign(ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt"] * 0.11)
    nonreporting_emi_df = pd.concat([nonreporting_emi_df, emi_09, emi_11], axis=0)

    nonreporting_emi_df.to_csv(nonreporting_emi_output_path, index=False)
