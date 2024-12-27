# %%
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile
import calendar
import datetime

from pyarrow import parquet
from io import StringIO
import pandas as pd
import duckdb
import osgeo
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
grassland_emi = pd.read_csv(emi_data_dir_path / "grassland_emi.csv")

# Read in proxy data
grassland_proxy = pd.read_csv(proxy_data_dir_path / "fire/MTBS_byEventFuelFuelbed_09Sep2024.csv")

# Read in MTBS proxy lat long data
mtbs_lat_long = pd.read_csv(proxy_data_dir_path / "fire/mtbs_lat_longs.csv")

# Read in fuelbed crosswalk data
fccs_fuelbed = pd.read_csv(proxy_data_dir_path / "fire/fccs_fuelbed_Aug2023_jesModified.csv")
nawfd_fuelbed = pd.read_csv(proxy_data_dir_path / "fire/nawfd_fuelbed_Aug2023_jesModified.csv")

# %% Functions

def get_habitat_types(dataset, fuelbed_column, habitat_types):
    """
    Get the specific habitat types of interest from the MTBS data. 
    
    Parameters:
    dataset (pd.DataFrame): The dataset to search for the habitat types of interest.
    fuelbed_column (str): The column name of the fuelbed to search for in the dataset.
    habitat_types (list): The list of habitat types to search for in the dataset.
    
    Returns:
    habitat_df (pd.Dataframe): A subset DataFrame of dataset that contains only the habitat types of interest.
    """

    habitat_df = dataset[dataset[fuelbed_column].str.lower().str.contains(habitat_types, na=False)]  

    return habitat_df


def convert_FIPS_to_two_letter_code(df, state_column):
    """
    Convert numeric FIPS state values in a DataFrame column to two-letter state codes.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the state names.
    state_column (str): The name of the column containing the FIPS numeric names.

    Returns:
    pd.DataFrame: DataFrame with the state column changed to two-letter state codes.
    """
    
    # Dictionary mapping full state names to their two-letter codes
    fips_state_abbr = {
    "1": "AL", "2": "AK", "4": "AZ", "5": "AR", "6": "CA",
    "8": "CO", "9": "CT", "10": "DE", "11": "DC", "12": "FL", 
    "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
    "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", 
    "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
    "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
    "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
    "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", 
    "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
    "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
    "56": "WY"
    }

    # Map the full state names to their two-letter codes using the dictionary
    df[state_column] = df[state_column].astype(int).astype(str).map(fips_state_abbr)
    
    return df


def calculate_state_emissions(emi_df, proxy_df, year_range):
    """
    Function to calculate emissions proxy for states based on the GHGI emissions data.

    Parameters:
    - emi_df: DataFrame containing GHGI emissions data.
    - proxy_df: DataFrame containing proxy data.
    - year_range: List or range of years to process.

    Returns:
    - final_proxy_df: DataFrame containing the processed emissions data for each state.
    """

    # Create the final dataframe that we will populate below
    final_proxy_df = pd.DataFrame()

    # Get the unique states in the emi data
    unique_states = emi_df['state_code'].unique()

    for year in year_range:
        
        # Filter total GHGI emissions for the current year
        year_emi = emi_df[emi_df['year'] == year].copy()  

        # Filter echo_df for the current year
        year_proxy = proxy_df[proxy_df['year'] == year].copy() 

        # Process emissions for each state
        for state in unique_states:

            year_state_proxy = year_proxy[year_proxy['originstatecd'] == state].copy()  

            # Group by eventID and sum the ch4_mg and burnBndAc columns
            year_state_proxy = year_state_proxy.groupby(['eventID'], as_index=False).agg({
            'eventID': 'first',
            'ch4_mg': 'sum',
            'burnBndAc': 'sum',
            'originstatecd': 'first',
            'year': 'first'
            })
            
            # Calculate the proportion of the total MTBS proxy emissions for the year-state
            year_state_proxy.loc[:, 'ch4_proportion'] = (
            year_state_proxy['ch4_mg'] / 
            year_state_proxy['ch4_mg'].sum()
            )       

            # Calculate the emissions for each facility
            state_ghgi_emis_value = year_emi[year_emi['state_code'] == state]['ghgi_ch4_kt'].values[0]

            # Round to 6 sig figs to match the GHGI data
            year_state_proxy.loc[:, 'emis_kt'] = np.round(
                year_state_proxy['ch4_proportion'] * state_ghgi_emis_value, 6
            )

            # Drop the flow proportion column
            year_state_proxy = year_state_proxy.drop(columns=['ch4_proportion'])

            # Concatenate the GHGRP matches and non-matches for our final dataframe
            final_proxy_df = pd.concat([final_proxy_df, year_state_proxy], ignore_index=True)

    return final_proxy_df

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
        geometry=gpd.points_from_xy(proxy_df['longitude'], proxy_df['latitude'])
    )

    # subset to only include the columns we want to keep
    proxy_gdf = proxy_gdf[['eventID', 'state_code', 'year', 'latitude', 'longitude', 'emis_kt', 'geometry']]
    
    # Normalize relative emissions to sum to 1 for each year and state
    proxy_gdf = proxy_gdf.groupby(['state_code', 'year']).filter(lambda x: x['emis_kt'].sum() > 0) #drop state-years with 0 total volume
    proxy_gdf['emis_kt'] = proxy_gdf.groupby(['year', 'state_code'])['emis_kt'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0) #normalize to sum to 1
    sums = proxy_gdf.groupby(["state_code", "year"])["emis_kt"].sum() #get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}" # assert that the sums are close to 1
    
    return proxy_gdf

# %% Step 1 - Data wrangling

# Edit the fuelbed_aggregate column to remove the 'evg' string and convert to an integer
grassland_proxy['fuelbed_aggregate'] = grassland_proxy['fuelbed_aggregate'].str.replace('evg', '').astype(float).astype(int)

# Convert the FIPS state codes to two-letter state codes
grassland_proxy = convert_FIPS_to_two_letter_code(grassland_proxy, 'originstatecd')

# Get the grassland habitat types from the FCCS data
fccs_grass = get_habitat_types(fccs_fuelbed, 'FUELBED', 'grassland')

nawfd_grass = get_habitat_types(nawfd_fuelbed, 'name', 'grassland')

# Filter the MTBS data to only include grassland habitat types in the FCCS and NAWFD data

grassland_proxy = grassland_proxy[
    grassland_proxy['fuelbed_aggregate'].isin(fccs_grass['FCCS']) |
    grassland_proxy['fuelbed_aggregate'].isin(nawfd_grass['nawfd_id'])
]


# %% Step 2 - Calculate proxy emissions for each state
grassland_proxy_df = calculate_state_emissions(grassland_emi, grassland_proxy, years)

# %% Join the MTBS lat long data to the proxy data

mtbs_lat_long = mtbs_lat_long[['Event_ID', 'BurnBndLat', 'BurnBndLon']]
mtbs_lat_long.rename(columns={'Event_ID': 'eventID', 'BurnBndLat': 'latitude', 'BurnBndLon': 'longitude'}, inplace=True)
grassland_proxy_df = grassland_proxy_df.merge(mtbs_lat_long, on='eventID', how='left')
grassland_proxy_df.rename(columns={'originstatecd': 'state_code'}, inplace=True)

# %% Step 3 Create the final proxy dataframe

final_grassland_proxy_df = create_final_proxy_df(grassland_proxy_df)

final_grassland_proxy_df.to_parquet(proxy_data_dir_path / "grassland_proxy.parquet", index=False)

# %%
