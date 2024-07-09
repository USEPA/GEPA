from pathlib import Path
from typing import Annotated

import pandas as pd
from pytask import Product, task, mark

from gch4i.config import (emi_data_dir_path, ghgi_data_dir_path, max_year,
                          min_year)
from gch4i.utils import tg_to_kt


@mark.persist
@task(id="petrochemicals_emi")
def task_get_petrochemicals_inv_data(
    input_path: Path = ghgi_data_dir_path
    / "industry/State_Petrochemicals_1990-2022.xlsx",
    output_path: Annotated[Path, Product] = emi_data_dir_path / "petrochemicals_emi.csv",
) -> None:
    """read in the ghgi_ch4_kt values for each state"""

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
        .rename(columns=lambda x: str(x).lower())
        # drop columns we don't need
        # # get just methane emissions
        .query("(ghg == 'CH4') & (subcategory1.str.contains('Acrylonitrile'))")
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
    emi_df.to_csv(output_path, index=False)
