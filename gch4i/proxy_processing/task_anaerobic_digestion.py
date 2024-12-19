# %%
from pathlib import Path
from typing import Annotated
from pyarrow import parquet
from io import StringIO
import pandas as pd

import geopandas as gpd
import numpy as np
import seaborn as sns
from pytask import Product, task, mark

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    emi_data_dir_path,
    global_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year,
    years
)

# %% Read in the data

# Read in emi data
anaerobic_digestion_emi = pd.read_csv(emi_data_dir_path / "anaerobic_digestion_emi.csv")

# %% Read in the proxy data
# The anaerobic digestion facilities are from the Excess Food Opportunities Map Data (EPA)
# EPA info page: https://www.epa.gov/sustainable-management-food/excess-food-opportunities-map

# download link: https://epa.maps.arcgis.com/home/item.html?id=d37cd4368d4d4ab8806546dd158c1d44
# Within the subfolder 'ExcelTables', there is a file named 'AnaerobicDigestionFacilities.xlsx'

anaerobic_digestion_proxy = pd.read_excel(proxy_data_dir_path / "anaerobic_digestion/AnaerobicDigestionFacilities.xlsx", sheet_name='Data')

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


# %% Calculate proxy emissions
final_proxy = pd.DataFrame()
for state in anaerobic_digestion_emi['state_code'].unique():
    state_emi = anaerobic_digestion_emi[anaerobic_digestion_emi['state_code'] == state].copy()
    state_proxy = anaerobic_digestion_proxy[anaerobic_digestion_proxy['State'] == state].copy()

    for year in years:
        state_emi_year = state_emi[state_emi['year'] == year]
        state_proxy['year'] = year

        WRRF_count = len(state_proxy[state_proxy['Facility Type'] == 'Water Resource Recovery Facility'])
        stand_alone_count = len(state_proxy[state_proxy['Facility Type'] == 'Stand-Alone'])

        # Calculate proportional emissions if both facility types exist in the state
        if stand_alone_count > 0 and WRRF_count > 0:
            WRRF_emi = state_emi_year['ghgi_ch4_kt'].values[0] * WRRF
            stand_alone_emi = state_emi_year['ghgi_ch4_kt'].values[0] * stand_alone
            state_proxy.loc[state_proxy['Facility Type'] == 'Stand-Alone', 'emis_kt'] = stand_alone_emi / stand_alone_count
            state_proxy.loc[state_proxy['Facility Type'] == 'Water Resource Recovery Facility', 'emis_kt'] = WRRF_emi / WRRF_count
        
        # Assign all emissions to WRRF if only WRRF facilities exist in the state
        elif stand_alone_count == 0 and WRRF_count > 0:
            state_proxy.loc[state_proxy['Facility Type'] == 'Water Resource Recovery Facility', 'emis_kt'] = state_emi_year['ghgi_ch4_kt'].values[0] / WRRF_count
        
        # Assign all emissions to stand-alone if only stand-alone facilities exist in the state (this never happens in the data but is included for completeness)
        elif stand_alone_count > 0 and WRRF_count == 0:
             state_proxy.loc[state_proxy['Facility Type'] == 'Stand-Alone', 'emis_kt'] = state_emi_year['ghgi_ch4_kt'].values[0] / stand_alone_count

        final_proxy = pd.concat([final_proxy, state_proxy], ignore_index=True)

# %% Create the final proxy df

def create_final_proxy_df(proxy_df):   
    """
    Function to create the final proxy df that is ready for gridding

    Parameters:
    - proxy_df: DataFrame containing proxy data.

    Returns:
    - final_proxy_df: DataFrame containing the processed emissions data for each state.
    """
    
    # Create a GeoDataFrame and generate geometry from longitude and latitude
    proxy_gdf = gpd.GeoDataFrame(
        proxy_df,
        geometry=gpd.points_from_xy(proxy_df['longitude'], proxy_df['latitude'], crs='EPSG:4326')
    )

    # subset to only include the columns we want to keep
    proxy_gdf = proxy_gdf[['state_code', 'year', 'latitude', 'longitude', 'emis_kt', 'geometry']]
    
    # Normalize relative emissions to sum to 1 for each year and state
    proxy_gdf = proxy_gdf.groupby(['state_code', 'year']).filter(lambda x: x['emis_kt'].sum() > 0) #drop state-years with 0 total volume
    proxy_gdf['emis_kt'] = proxy_gdf.groupby(['year', 'state_code'])['emis_kt'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0) #normalize to sum to 1
    sums = proxy_gdf.groupby(["state_code", "year"])["emis_kt"].sum() #get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}" # assert that the sums are close to 1
    
    return proxy_gdf

final_proxy = final_proxy.rename(columns={'Latitude': 'latitude', 'Longitude': 'longitude', 'State': 'state_code'})

final_proxy = create_final_proxy_df(final_proxy)

# %% Save the final proxy dataframes to parquet files
final_proxy.to_parquet(proxy_data_dir_path / "anaerobic_digestion_proxy.parquet", index=False)
