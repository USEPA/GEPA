"""
Name:                   task_comb_mob_mines.py
Date Last Modified:     2025-01-23
Authors Name:           Nick Kruskamp (RTI International)
Purpose:                This script creates a proxy for mobile combustion mines. This
                        proxy is created from the MSHA dataset and the US state
                        shapefile. All active mines are used here, relative emissions
                        are calculated by state, and the data is normalized. The
                        emissions are equally allocated to all mines by state.
Input Files:            - Mines.zip
                        - tl_2020_us_state.zip
Output Files:           - mines_proxy.parquet
"""

# %%

from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import osgeo  # noqa
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
from pytask import Product, mark, task

from gch4i.config import (
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
)
from gch4i.utils import normalize


@mark.persist
@task(id="mines_proxy")
def task_mob_comb_mines(
    msha_path: Path = sector_data_dir_path / "abandoned_mines/Mines.zip",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path / "mines_proxy.parquet",
):

    with ZipFile(msha_path) as z:
        with z.open("Mines.txt") as f:
            msha_df = (
                pd.read_table(
                    f,
                    sep="|",
                    encoding="ISO-8859-1",
                    usecols=["MINE_ID", "LATITUDE", "LONGITUDE", "CURRENT_MINE_STATUS"],
                )
                .query("CURRENT_MINE_STATUS == 'Active'")
                .dropna(subset=["LATITUDE", "LONGITUDE"])
                .set_index("MINE_ID")
            )

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

    msha_gdf = (
        gpd.GeoDataFrame(
            msha_df.drop(columns=["LATITUDE", "LONGITUDE"]),
            geometry=gpd.points_from_xy(msha_df["LONGITUDE"], msha_df["LATITUDE"]),
            crs=4326,
        )
        .sjoin(state_gdf, how="inner")
        .assign(emis_kt=1)
    )
    msha_gdf["rel_emi"] = msha_gdf.groupby("state_code")["emis_kt"].transform(normalize)
    print("Active mines with location: ", len(msha_gdf))

    all_eq_df = (
        msha_gdf.groupby("state_code")["rel_emi"]
        .sum()
        .rename("sum_check")
        .to_frame()
        .assign(is_close=lambda df: (np.isclose(df["sum_check"], 1)))
    )
    all_eq_df

    if not all_eq_df["is_close"].all():
        raise ValueError("not all values are normed correctly!")
    _, ax = plt.subplots(figsize=(20, 20), dpi=150)
    state_gdf.boundary.plot(color="xkcd:slate", ax=ax)
    msha_gdf.plot(ax=ax, color="xkcd:goldenrod", markersize=1)

    msha_gdf.to_parquet(output_path)
