"""
Name:                  task_ng_transmission_import_proxy.py
Date Last Modified:    2025-01-27
Authors Name:          John Bollenbacher  (RTI International)
Purpose:               Creates proxy data for LNG import terminals.
Input Files:           - {sector_data_dir_path}/lng/lng_importexport_terminals_v3.csv
Output Files:          - {proxy_data_dir_path}/import_terminals_proxy.parquet
"""

# %% Import Libraries
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import pandas as pd
import numpy as np
from pytask import Product, mark, task

from gch4i.config import (
    max_year,
    min_year,
    proxy_data_dir_path,
    sector_data_dir_path,
)


# %% Pytask Function
@mark.persist
@task(id="import_terminals_proxy")
def get_ng_import_proxy_data(
    lng_terminals_path: Path = sector_data_dir_path / "lng/lng_importexport_terminals_v3.csv",
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "import_terminals_proxy.parquet"
    ),
):
    """
    Create proxy data for LNG import terminals.

    Args:
        lng_terminals_path (Path): Path to the LNG terminal data.
        output_path (Path): Path to save the output proxy data.

    Returns:
        None. Proxy data is saved to a parquet file.
    """

    # Load LNG terminal data
    lng_terminals = pd.read_csv(lng_terminals_path)

    # Filter for import terminals and relevant years
    import_terminals = lng_terminals[
        (lng_terminals['import_or_export'] == 'import') &
        (lng_terminals['year'] >= min_year) &
        (lng_terminals['year'] <= max_year)
    ]

    # Create a GeoDataFrame from the import terminals data
    gdf_terminals = gpd.GeoDataFrame(
        import_terminals,
        geometry=gpd.points_from_xy(
            import_terminals['Terminal Longitude'],
            import_terminals['Terminal Latitude']),
        crs="EPSG:4326"
    )

    # Rename columns to match the expected output format
    gdf_terminals = gdf_terminals.rename(columns={
        'terminal_name': 'terminal_name',
        'state': 'state_code',
        'year': 'year',
        'volume': 'rel_emi'
    })

    # aggregate volumes by terminal-year
    terminal_year_volumes = gpd.GeoDataFrame(
        gdf_terminals.groupby(['terminal_name', 'year'])['rel_emi'].sum().reset_index()
        )

    proxy_gdf = (gpd.GeoDataFrame(
        terminal_year_volumes.merge(
            gdf_terminals[['terminal_name',
                           'year',
                           'state_code',
                           'geometry']].drop_duplicates(),
            on=['terminal_name', 'year'],
            how='left')).sort_values(by=['year', 'state_code', 'terminal_name'])
        .reset_index(drop=True)[['year',
                                 'state_code',
                                 'rel_emi',
                                 'geometry',
                                 'terminal_name']])

    # Normalize relative emissions to sum to 1 for each year and state
    # drop state-years with 0 total volume
    proxy_gdf = (
        proxy_gdf.groupby(['year'])
        .filter(lambda x: x['rel_emi'].sum() > 0)
        )
    # normalize to sum to 1
    proxy_gdf['rel_emi'] = (
        proxy_gdf.groupby(['year'])['rel_emi']
        .transform(lambda x: x / x.sum() if x.sum() > 0 else 0))
    # get sums to check normalization
    sums = proxy_gdf.groupby(["year"])["rel_emi"].sum()
    # assert that the sums are close to 1
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"

    # Drop state_code column
    proxy_gdf = proxy_gdf.drop(columns=['state_code'])

    # Save to parquet
    proxy_gdf.to_parquet(output_path)

    return None


get_ng_import_proxy_data()
# %%
