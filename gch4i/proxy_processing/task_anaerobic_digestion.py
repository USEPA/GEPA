"""
Name:                   task_anaerobic_digestion.py
Date Last Modified:     2024-01-27
Authors Name:           C. Coxen (RTI International)
Purpose:                Mapping of stationary combustion proxy emissions
Input Files:            - anaeorbic_digestion_emi.csv
                        - AnaerobicDigestionFacilities.xlsx"
Output Files:           - anaerobic_digestion_proxy.parquet
Notes:                  - The anaerobic digestion facilities are from the Excess Food Opportunities Map Data (EPA)
"""
# %%
from pathlib import Path
from typing import Annotated
from pyarrow import parquet
from io import StringIO
import pandas as pd

import geopandas as gpd
import numpy as np
from pytask import Product, task, mark

from gch4i.config import (
    proxy_data_dir_path,
    sector_data_dir_path
)

from gch4i.utils import (
    normalize
)

anaerobic_digestion_proxy_path = sector_data_dir_path / "anaerobic_digestion/AnaerobicDigestionFacilities.xlsx"
anaerobic_digestion_proxy_ouput_path = proxy_data_dir_path / "anaerobic_digestion_proxy.parquet"

# %% pytask
@mark.persist
@task

# %% pytask function
@mark.persist
@task(id='anaerobic_digestion_proxy')
def task_anaerobic_digestion_proxy_data(
    proxy_path: Path = anaerobic_digestion_proxy_path,
    output_path: Annotated[Path, Product] = anaerobic_digestion_proxy_ouput_path
    ) -> None:
    """
    This function processes the anaerobic digestion proxy data and calculates the proxy emissions.

    Args:
    proxy_path: Path to the anaerobic digestion proxy data.
    output_path: Path to save the final proxy data.

    Returns:
    None. Proxy data is saved to a parquet file at the output_path.
    """

    # %% Read in the proxy data
    # The anaerobic digestion facilities are from the Excess Food Opportunities Map Data (EPA)
    # EPA info page: https://www.epa.gov/sustainable-management-food/excess-food-opportunities-map

    # download link: https://epa.maps.arcgis.com/home/item.html?id=d37cd4368d4d4ab8806546dd158c1d44
    # Within the subfolder 'ExcelTables', there is a file named 'AnaerobicDigestionFacilities.xlsx'

    anaerobic_digestion_proxy = pd.read_excel(proxy_path, sheet_name='Data')

    # drop rows that are missing latitudes and longitudes - note that missing values appear to all be farm digesters
    anaerobic_digestion_proxy = anaerobic_digestion_proxy.dropna(subset=['Latitude', 'Longitude'])

    # %% Biogas production ratio to assign proportional emissions to WRRF and stand-alone digesters
    # provided in the EPA document "Anaerobic Digestion Facilities Processing Food Waste in the United States (2019)"
    # https://www.epa.gov/system/files/documents/2023-04/Anaerobic_Digestion_Facilities_Processing_Food_Waste_in_the_United_States_2019_20230404_508.pdf

    # From Table 24 on page 27
    # The total biogas production in 2019 is 29,877 SCFM (Standard Cubic Feet per Minute)


    total_SCFM = 29877 - 1465 # 1,465 SCFM is from the farm digesters (all have missing lat/long data), which we need to substract from the total in order to get accurate proportions for the WRRF and stand-alone digesters
    WRRF = 23587 / total_SCFM # 23,587 SCFM is from WRRF
    stand_alone = 4825 / total_SCFM # 4,825 SCFM is from stand-alone digesters

    # Assign biogas production ratios to each facility type
    anaerobic_digestion_proxy.loc[anaerobic_digestion_proxy['Facility Type'] == 'Stand-Alone', 'biogas'] = stand_alone
    anaerobic_digestion_proxy.loc[anaerobic_digestion_proxy['Facility Type'] == 'Water Resource Recovery Facility', 'biogas'] = WRRF

    # Rename columns
    anaerobic_digestion_proxy = anaerobic_digestion_proxy.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude', 'State': 'state_code'})

    # Create relative emissions for each state
    anaerobic_digestion_proxy["rel_emi"] = anaerobic_digestion_proxy.groupby("state_code")["biogas"].transform(normalize)

    # Check that the relative emissions sum to 1 for each state
    sums = anaerobic_digestion_proxy.groupby(["state_code"])["rel_emi"].sum() #get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}" # assert that the sums are close to 1

    # Create a GeoDataFrame and generate geometry from longitude and latitude
    anaerobic_digestion_proxy = gpd.GeoDataFrame(
        anaerobic_digestion_proxy,
        geometry=gpd.points_from_xy(anaerobic_digestion_proxy['longitude'], anaerobic_digestion_proxy['latitude'], crs='EPSG:4326')
    )

    # Only keep the columns we need
    anaerobic_digestion_proxy = anaerobic_digestion_proxy[['state_code', 'rel_emi', 'geometry']]

    # %% Save the final proxy dataframes to parquet files
    anaerobic_digestion_proxy.to_parquet(output_path, index=False)

# %%
