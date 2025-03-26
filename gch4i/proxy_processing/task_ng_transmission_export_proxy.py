"""
Name:                  task_ng_transmission_export_proxy.py
Date Last Modified:    2025-03-24
Authors Name:          John Bollenbacher  (RTI International)
Purpose:               Creates proxy data for LNG export terminals based on terminal
                        locations and volumes.
Input Files:           - {sector_data_dir_path}/lng/lng_importexport_terminals_v3.csv
Output Files:          - {proxy_data_dir_path}/export_terminals_proxy.parquet
"""

# %% Import Libraries
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import numpy as np
import pandas as pd
from pytask import Product, mark, task

from gch4i.config import (
    max_year,
    min_year,
    proxy_data_dir_path,
    sector_data_dir_path,
    global_data_dir_path,
)

from gch4i.utils import normalize


# %% Pytask Function
@mark.persist
@task(id="export_terminals_proxy")
def get_ng_export_proxy_data(
    lng_terminals_path: Path = sector_data_dir_path
    / "lng/lng_importexport_terminals_v3.csv",
    state_geo_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "export_terminals_proxy.parquet"
    ),
):
    """
    Creates proxy data for LNG export terminals based on terminal locations and volumes.

    Args:
        lng_terminals_path (Path): Path to the LNG terminal data.

    Returns:
        None. Proxy data is saved to a parquet file.
    """

    # Load LNG terminal data
    lng_terminals = pd.read_csv(lng_terminals_path)

    state_gdf = (
        gpd.read_file(state_geo_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .rename(columns={"statefp": "fips"})
        .to_crs(4326)
    )

    # Filter for export terminals and relevant years
    export_terminals = lng_terminals[
        (lng_terminals["import_or_export"] == "export")
        & (lng_terminals["year"] >= min_year)
        & (lng_terminals["year"] <= max_year)
    ]

    # Create a GeoDataFrame from the export terminals data
    gdf_terminals = gpd.GeoDataFrame(
        export_terminals,
        geometry=gpd.points_from_xy(
            export_terminals["Terminal Longitude"],
            export_terminals["Terminal Latitude"],
        ),
        crs="EPSG:4326",
    )
    gdf_terminals.normalize()

    ax = gdf_terminals.plot(markersize=10, label="Terminals")
    state_gdf.boundary.plot(color="black", linewidth=1, ax=ax)
    ax.set_title("LNG Export Terminals by State RAW")

    gdf_terminals = gdf_terminals.sjoin(state_gdf, how="inner")
    ax = gdf_terminals.plot(
        "state_code",
        cmap="Set1",
        legend=True,
        figsize=(10, 10),
        alpha=0.5,
        missing_kwds={"color": "gray", "label": "No State"},
    )
    state_gdf.boundary.plot(color="black", linewidth=1, ax=ax)
    ax.set_title("LNG Export Terminals by State FILTERED")

    # Rename columns to match the expected output format
    gdf_terminals = gdf_terminals.rename(
        columns={
            "terminal_name": "terminal_name",
            "year": "year",
            "volume": "rel_emi",
        }
    )

    proxy_gdf = (
        gdf_terminals.groupby(["terminal_name", "year"])
        .agg({"rel_emi": "sum", "geometry": "first", "state_code": "first"})
        .reset_index()
        .sort_values(by=["year", "state_code", "terminal_name"])
    )
    proxy_gdf = gpd.GeoDataFrame(proxy_gdf, geometry="geometry", crs="EPSG:4326")
    ax = proxy_gdf.plot(
        "state_code", cmap="Set1", legend=True, figsize=(10, 10), alpha=0.5
    )
    state_gdf.boundary.plot(color="black", linewidth=1, ax=ax)

    # Normalize relative emissions to sum to 1 for each year and state
    # drop state-years with 0 total volume
    proxy_gdf = proxy_gdf.groupby(["year"]).filter(
        lambda x: x["rel_emi"].sum() > 0
    )
    # normalize to sum to 1
    proxy_gdf["rel_emi"] = proxy_gdf.groupby(["year"])[
        "rel_emi"
    ].transform(normalize)
    # get sums to check normalization
    sums = proxy_gdf.groupby(["year"])["rel_emi"].sum()
    # assert that the sums are close to 1
    assert np.isclose(
        sums, 1.0, atol=1e-8
    ).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"

    # Save to parquet
    proxy_gdf.to_parquet(output_path)

    return None


get_ng_export_proxy_data()
# %%
