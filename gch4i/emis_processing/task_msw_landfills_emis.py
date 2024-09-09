# %%
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile
import calendar
import datetime

from pyarrow import parquet
import pandas as pd
import osgeo
import geopandas as gpd
import numpy as np
import seaborn as sns
from pytask import Product, task, mark
import geopy
from geopy.geocoders import Nominatim
import duckdb

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    ghgi_data_dir_path,
    emi_data_dir_path,
    max_year,
    min_year,
    years,
)

from gch4i.utils import name_formatter

t_to_kt = 0.001  # conversion factor, metric tonnes to kilotonnes
mmt_to_kt = 1000  # conversion factor, million metric tonnes to kilotonnes
year_range = [*range(min_year, max_year+1,1)] #List of emission years
year_range_str=[str(i) for i in year_range]
num_years = len(year_range)


@mark.persist
@task(id="msw_landfills_emi")
def task_get_msw_landfills_inv_data(
    input_path: Path = ghgi_data_dir_path
    / "landfills/State_MSW_LF_1990-2022_LA.xlsx",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    reporting_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "msw_landfills_r_emi.csv",
    nonreporting_emi_output_path: Annotated[Path, Product] = emi_data_dir_path / "msw_landfills_nr_emi.csv",
) -> None:
    
    # Get state vectors and state_code for use with inventory and proxy data
    state_gdf = (
    gpd.read_file(state_path)
    .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
    .rename(columns=str.lower)
    .rename(columns={"stusps": "state_code", "name": "state_name"})
    .astype({"statefp": int})
    # get only lower 48 + DC
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
    .to_crs(4326)
    )

    """read in the ghgi_ch4_kt values for each state"""
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
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
    )

    reporting_emi_df = emi_df.copy()
    nonreporting_emi_df = pd.DataFrame()

    # Get non-reporting emissions by scaling reporting emissions.
    # Assume emissions are 9% of reporting emissions for 2016 and earlier.
    # Assume emissions are 11% of reporting emissions for 2017 and later.
    emi_09 = emi_df.query("year <= 2016").assign(ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt"] * 0.09)
    emi_11 = emi_df.query("year >= 2017").assign(ghgi_ch4_kt=lambda df: df["ghgi_ch4_kt"] * 0.11)
    nonreporting_emi_df = pd.concat([nonreporting_emi_df, emi_09, emi_11], axis=0)

    nonreporting_emi_df.to_csv(nonreporting_emi_output_path, index=False)

    reporting_emi_df["ghgi_ch4_kt"] = emi_df["ghgi_ch4_kt"] - nonreporting_emi_df["ghgi_ch4_kt"]
    reporting_emi_df.to_csv(reporting_emi_output_path, index=False)

# %%
