# %%
from pathlib import Path
from typing import Annotated

from pyarrow import parquet
import pandas as pd
import osgeo
import geopandas as gpd
import numpy as np
import seaborn as sns
from pytask import Product, task, mark

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
)

# %%
@mark.persist
@task(id="oil_well_avg_proxy")
def task_get_oil_well_avg_proxy_data(
    oil_all_well_prod_proxy_path: Annotated[Path, Product] = proxy_data_dir_path / "oil_all_well_prod_proxy.parquet",
    oil_well_avg_output_path: Annotated[Path, Product] = proxy_data_dir_path / "oil_well_avg_proxy.parquet",
    ):
    """
    This proxy is the weighted average of the well count proxy and the oil production 
    proxy for all wells. 50% of the relative emission is assigned based on the 
    individual well's well count, and 50% of the relative emission is assigned based 
    on the individual well's oil production.

    This file takes the relative emissions based on oil production from the
    oil_all_well_prod_proxy, adds in a new relative emission column for well count with
    the assumption that each location has WELL_COUNT = 1, and takes the weighted average
    of the two relative emission types to create a new average proxy.

    """

    # Read in the oil production proxy and assign well count
    well_avg_gdf = (gpd.read_parquet(oil_all_well_prod_proxy_path)
                    .rename(columns={'rel_emi': 'prod_rel_emi'})
                    .assign(well_count = 1.0)
                    )
    
    # Convert well count into a relative emission where each state-year combination sums to 1
    well_avg_gdf['count_rel_emi'] = well_avg_gdf.groupby(['state_code', 'year'])['well_count'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    well_avg_gdf = well_avg_gdf.drop(columns='well_count')

    # Check that relative emissions sum to 1.0 each state/year combination
    prod_sums = well_avg_gdf.groupby(["state_code", "year"])["prod_rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(prod_sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {prod_sums}"  # assert that the sums are close to 1

    count_sums = well_avg_gdf.groupby(["state_code", "year"])["count_rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(count_sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {count_sums}"  # assert that the sums are close to 1

    # Create the average relative emission with 50% weights
    well_avg_gdf = (well_avg_gdf
                    .assign(rel_emi=lambda df: 0.5 * (df['prod_rel_emi'] + df['count_rel_emi']))
                    .drop(columns={'prod_rel_emi', 'count_rel_emi'})
                    )
    
    # Check that relative emissions sum to 1.0 each state/year combination
    avg_sums = well_avg_gdf.groupby(["state_code", "year"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(avg_sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {avg_sums}"  # assert that the sums are close to 1

    # Output Proxy Parquet Files
    well_avg_gdf = well_avg_gdf.astype({'year': str})
    well_avg_gdf.to_parquet(oil_well_avg_output_path)

    return None
