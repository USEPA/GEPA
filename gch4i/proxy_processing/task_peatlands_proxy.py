"""
Name:                   task_peatlands_proxy.py
Date Last Modified:     2024-01-22
Authors Name:           C. Coxen (RTI International)
Purpose:                Assign GHGI emissions to peatlands 
Input Files:            - peatlands_emi.csv, peat_producers_2013_geocoded.csv
Output Files:           - peatlands_proxy.parquet
Notes:                  - This script assigns GHGI emissions from peatland production based on the location of peat producers sourced from the 2013 USGS peatland producers dataset
                        - 2013 Peat Producers data: https://d9-wret.s3.us-west-2.amazonaws.com/assets/palladium/production/mineral-pubs/peat/dir-2013-peat.pdf
                        - Additional lat/long data was sourced from Google Maps for facilities that could not be referenced by Nomatim
                        - Some facilities were not able to be found in Google Maps and were dropped from the dataset
"""

# %%
from pathlib import Path
from typing import Annotated
import pandas as pd

from pytask import Product, mark

from gch4i.config import (
    emi_data_dir_path,
    sector_data_dir_path,
    proxy_data_dir_path,
    years,
)
from gch4i.utils import (
    geocode_address,
    create_final_proxy_df
)


# %% Input data paths
emi_path = emi_data_dir_path / "peatlands_emi.csv"
peatland_producers_path = sector_data_dir_path / "peatlands" / "peat_producers_2013.csv"
geocoded_peatland_producers_path = sector_data_dir_path / "peatlands" / "peat_producers_2013_geocoded.csv"
proxy_output_path = proxy_data_dir_path / "peatlands_proxy.parquet"
# %% py task

@mark.persist  
@task
def task_peatlands_proxy(
    emi_path: Path = emi_path,
    peatland_producers_path: Path = peatland_producers_path,
    geocoded_peatland_producers_path: Path = geocoded_peatland_producers_path,
    output_path: Annotated[Path, Product] = proxy_output_path
) -> None:

    # %%
    # Read in data
    peatland_emissions = pd.read_csv(emi_path)

    if geocoded_peatland_producers_path.exists():
        peatland_producers = pd.read_csv(geocoded_peatland_producers_path)

    else: # Geocode peatland producers to have latitute and longitude information
        peatland_producers = pd.read_csv(peatland_producers_path)
        peatland_producers = geocode_address(peatland_producers, "Address")

        # drop rows with missing lat/long - we are unable to assign emissions to these producers because we can't locate them
        peatland_producers.dropna(subset=["latitude", "longitude"], inplace=True)

        # save geocoded peatland producers
        peatland_producers.to_csv(geocoded_peatland_producers_path, index=False)

    # %% Functions
    def calculate_emissions_proxy(emi_df, proxy_df, year_range):
        """
        Function to calculate emissions proxy by joining emissions and facility data by state,
        then equally dividing emissions across facilities within each state.

        Parameters:
        - emi_df: DataFrame containing GHGI emissions data
        - proxy_df: DataFrame containing facility location data
        - year_range: List or range of years to process

        Returns:
        - DataFrame with emissions allocated to each facility
        """
        # Create a cross join between proxy locations and years. Expand the proxy data to include an entry for each year
        proxy_years = proxy_df.assign(key=1).merge(
            pd.DataFrame({'year': year_range, 'key': 1}),
            on='key'
        ).drop('key', axis=1)

        # Merge emissions data with proxy locations
        final_proxy_df = proxy_years.merge(
            emi_df[['state_code', 'year', 'ghgi_ch4_kt']], 
            on=['state_code', 'year'],
            how='inner'
        )

        # Calculate facility count per state-year
        state_facility_counts = final_proxy_df.groupby(['state_code', 'year']).size().reset_index(name='facility_count')
        
        # Merge facility counts and calculate relative emissions
        final_proxy_df = final_proxy_df.merge(
            state_facility_counts,
            on=['state_code', 'year']
        )
        final_proxy_df['emis_kt'] = final_proxy_df['ghgi_ch4_kt'] / final_proxy_df['facility_count']

        return final_proxy_df
    # %%

    final_proxy_df = calculate_emissions_proxy(peatland_emissions, peatland_producers, years)

    peatlands_proxy = create_final_proxy_df(final_proxy_df)

    peatlands_proxy.to_parquet(proxy_output_path)

# %%
