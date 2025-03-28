"""
Name:                   task_peatlands_proxy.py
Date Last Modified:     2024-01-22
Authors Name:           C. Coxen (RTI International)
Purpose:                Generate proxy data for peatlands emissions
Input Files:            - {emi_data_dir_path}/peatlands_emi.csv
                        - {sector_data_dir_path}/peatlands/peat_producers_2013.csv
                          {sector_data_dir_path}/peatlands/
                            peat_producers_2013_geocoded.csv
Output Files:           - peatlands_proxy.parquet
Notes:                  - This script assigns GHGI emissions from peatland production
                            based on the location of peat producers sourced from the
                            2013 USGS peatland producers dataset
                        - 2013 Peat Producers data: https://d9-wret.s3.us-west-2.
                            amazonaws.com/assets/palladium/production/mineral-pubs/
                            peat/dir-2013-peat.pdf
                        - Additional lat/long data was sourced from Google Maps for
                            facilities that could not be referenced by Nomatim
                        - Some facilities were not able to be found in Google Maps and
                            were dropped from the dataset
"""
# %% Import Libraries
from pathlib import Path
from typing import Annotated
import numpy as np
import geopandas as gpd
import pandas as pd

from pytask import Product, mark, task

from gch4i.config import (
    sector_data_dir_path,
    proxy_data_dir_path
)
from gch4i.utils import (
    geocode_address,
    normalize
)

# %% Input data paths
peatland_producers_path = sector_data_dir_path / "peatlands" / "peat_producers_2013.csv"
geocoded_peatland_producers_path = sector_data_dir_path / "peatlands" / "peat_producers_2013_geocoded.csv"
proxy_output_path = proxy_data_dir_path / "peatlands_proxy.parquet"

# %% Pytask function
@mark.persist
@task(id="peatlands_proxy")
def task_peatlands_proxy(
    peatland_producers_path: Path = peatland_producers_path,
    geocoded_peatland_producers_path: Path = geocoded_peatland_producers_path,
    proxy_output_path: Annotated[Path, Product] = proxy_output_path
) -> None:
    """
    Create proxy data for peatlands emissions.

    Args:
        peatland_producers_path (Path): Path to peatland producers data (if needed)
        geocoded_peatland_producers_path (Path): Path to geocoded peatland producers
        data

    Returns:
        None. Proxy data is saved to output_path.
    """

    # Check if geocoded peatland producers data exists
    if geocoded_peatland_producers_path.exists():
        peatland_producers = pd.read_csv(geocoded_peatland_producers_path)

    # Geocode peatland producers to have latitute and longitude information
    else:
        peatland_producers = pd.read_csv(peatland_producers_path)
        peatland_producers = geocode_address(peatland_producers, "Address")

        # drop rows with missing lat/long - we are unable to assign emissions to these
        # producers because we can't locate them
        peatland_producers.dropna(subset=["latitude", "longitude"], inplace=True)

        # save geocoded peatland producers
        peatland_producers.to_csv(geocoded_peatland_producers_path, index=False)

    # Only keep relevant columns
    peatlands_proxy = peatland_producers[['state_code', 'latitude', 'longitude']].copy()

    # Initialize relative emissions to 1.0
    peatlands_proxy['rel_emi'] = 1.0 

    # Normalize relative emissions for each state
    peatlands_proxy["rel_emi"] = peatlands_proxy.groupby("state_code")["rel_emi"].transform(normalize)

    # Check that the relative emissions sum to 1 for each state
    sums = peatlands_proxy.groupby(["state_code"])["rel_emi"].sum()
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}" # assert that the sums are close to 1
    # Create a GeoDataFrame and generate geometry from longitude and latitude
    peatlands_proxy = gpd.GeoDataFrame(
        peatlands_proxy,
        geometry=gpd.points_from_xy(peatlands_proxy['longitude'], peatlands_proxy['latitude'], crs='EPSG:4326')
    )

    peatlands_proxy = peatlands_proxy[['state_code', 'rel_emi', 'geometry']]

    # Save proxy data
    peatlands_proxy.to_parquet(proxy_output_path, index=False)

# %%
