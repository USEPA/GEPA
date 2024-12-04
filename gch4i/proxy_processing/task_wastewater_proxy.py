# %%
from pathlib import Path
from typing import Annotated, List, Dict
from zipfile import ZipFile
import calendar
import datetime

import requests
import time
from pyarrow import parquet
from io import StringIO
import pandas as pd
import duckdb
import osgeo
import geopandas as gpd
import numpy as np
import seaborn as sns
from pytask import Product, task, mark
from geopy.distance import geodesic
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderServiceError

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

from gch4i.utils import name_formatter

# TO DO:
# 1. Add this facility to the FRS data so we have any data for fruit and veg https://abundantmontana.com/amt-lister/mission-mountain-food-enterprise-center/
# 2. Write a separate function for brewery processing since we also need to incorporate the brewery db data. 
# add the brewery and FRS data together, then add to the ECHO data. Allocate 25% of the emis equally to the FRS and brewery db data. If no data, all to brewery db
# 3. Write some code that will add lat long geocodes to the brewery db data. see below.

# Change the logic so that we exclude FRS facilities when checking matches with the GHGRP data
# Only match GHGRP to ECHO and deal with frs and brewerydb data separately.



# from geopy.geocoders import Nominatim
# from geopy.exc import GeocoderTimedOut, GeocoderServiceError
# import time
# import pandas as pd





# %%

# Read in emi data
ww_dom_nonseptic_emi = pd.read_csv(emi_data_dir_path / "ww_dom_nonseptic_emi.csv")
ww_sep_emi = pd.read_csv(emi_data_dir_path / "ww_sep_emi.csv")
ww_brew_emi = pd.read_csv(emi_data_dir_path / "ww_brew_emi.csv")
ww_ethanol_emi = pd.read_csv(emi_data_dir_path / "ww_ethanol_emi.csv")
ww_fv_emi = pd.read_csv(emi_data_dir_path / "ww_fv_emi.csv")
ww_mp_emi = pd.read_csv(emi_data_dir_path / "ww_mp_emi.csv")
ww_petrref_emi = pd.read_csv(emi_data_dir_path / "ww_petrref_emi.csv")
ww_pp_emi = pd.read_csv(emi_data_dir_path / "ww_pp_emi.csv")

# Concatenate all industrial emissions dataframes for later use
industrial_emi_dfs = [ww_brew_emi, ww_ethanol_emi, ww_fv_emi, ww_mp_emi, ww_petrref_emi, ww_pp_emi]
industrial_emis = pd.concat(industrial_emi_dfs, axis=0, ignore_index=True)

# GHGRP Data
ghgrp_emi_ii_inputfile = proxy_data_dir_path / "wastewater/ghgrp_subpart_ii.csv"

ghgrp_facility_ii_inputfile = proxy_data_dir_path / "wastewater/SubpartII_Facilities.csv"

# %% Functions
def read_combined_file(file_path):
    """Reads the combined ECHO data from the given file path."""
    print('Combined ECHO data already created and file has been read in.')
    return pd.read_csv(file_path, low_memory=False)

def read_and_combine_csv_files(directory):
    """Reads and combines all ECHO .csv files in the given directory, excluding the combined file if it exists."""
    dataframes = []
    for file in os.listdir(directory):
        if file.endswith('.csv') and not file.startswith("combined_echo_data"):
            file_path = os.path.join(directory, file)
            df = pd.read_csv(file_path, skiprows=3, low_memory=False)
            dataframes.append(df)
    combined_df = pd.concat(dataframes, ignore_index=True)
    # Only keep the necessary columns and fill missing values with 0
    combined_df = combined_df[[
        "Year", "NPDES Permit Number", "FRS ID", "CWNS ID(s)", "Facility Type Indicator", 
        "SIC Code", "NAICS Code", "City", "State", "County", "Facility Latitude", 
        "Facility Longitude", "Wastewater Flow (MGal/yr)", "Average Daily Flow (MGD)"
    ]].fillna(0)
    # Write the combined data to a CSV file
    combined_df.to_csv(combined_echo_file_path, index=False)
    return combined_df


def extract_industry_facilities(echo_nonpotw, industry_name, naics_codes, sic_codes):
    """
    Extract industry-specific facilities from the non-POTW list.
    
    Parameters:
    echo_nonpotw (pd.DataFrame): The input DataFrame containing non-POTW facilities.
    industry_name (str): The name of the industry (used for the output variable name).
    naics_codes (str or list): Single NAICS code/prefix or list of NAICS codes.
    sic_codes (list): A list of SIC codes for the industry.
    
    Returns:
    pd.DataFrame: A DataFrame containing the extracted industry-specific facilities.
    """
    # Convert single NAICS code to list for consistency
    if isinstance(naics_codes, str):
        naics_codes = [naics_codes]
    
    # Create NAICS filter condition
    naics_condition = echo_nonpotw['NAICS Code'].str.startswith(tuple(naics_codes))
    
    # Create SIC filter condition
    sic_condition = echo_nonpotw['SIC Code'].str.contains('|'.join(sic_codes))
    
    # Apply both conditions
    industry_df = echo_nonpotw[naics_condition | sic_condition].copy()
    
    industry_df.reset_index(inplace=True, drop=True)
    
    return industry_df


def convert_state_names_to_codes(df, state_column):
    """
    Convert full state names in a DataFrame column to two-letter state codes.
    If column already contains valid two-letter codes, return DataFrame unchanged.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the state names.
    state_column (str): The name of the column containing the full state names.

    Returns:
    pd.DataFrame: DataFrame with the state column changed to two-letter state codes.
    """
    
    # Dictionary mapping full state names to their two-letter codes
    state_name_to_code = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'District of Columbia': 'DC', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
        'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
        'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    
    # Get set of valid state codes
    valid_codes = set(state_name_to_code.values())
    
    # Check if values are already state codes
    current_values = set(df[state_column].unique())
    if current_values.issubset(valid_codes):
        return df
    
    # Convert all caps to title case
    df[state_column] = df[state_column].str.title()

    # Map the full state names to their two-letter codes using the dictionary
    df[state_column] = df[state_column].map(state_name_to_code)
    
    return df


def calculate_potw_emissions_proxy(emi_df, echo_df, year_range):
    """
    Function to calculate emissions proxy for states based on GHGI and GHGRP data.

    Parameters:
    - emi_df: DataFrame containing GHGI emissions data.
    - echo_df: DataFrame containing ECHO facility data.
    - year_range: List or range of years to process.

    Returns:
    - final_proxy_df: DataFrame containing the processed emissions data for each state.
    """

    # Create the final dataframe that we will populate below
    final_proxy_df = pd.DataFrame()

    # Get the unique states in the ECHO data
    unique_states = emi_df['state_code'].unique()

    for year in year_range:
        
        # Filter total GHGI emissions for the current year
        year_emi = ww_dom_nonseptic_emi[ww_dom_nonseptic_emi['year'] == year].copy()  

        # Filter echo_df for the current year
        year_echo = echo_potw[echo_potw['Year'] == year].copy() 

        # Process emissions for each state
        for state in unique_states:

            year_state_echo = year_echo[year_echo['State'] == state].copy()  

            # Calculate the proportional flow at each facility
            year_state_echo.loc[:, 'flow_proportion'] = (
                year_state_echo['Wastewater Flow (MGal/yr)'] / 
                year_state_echo['Wastewater Flow (MGal/yr)'].sum()
            )

            # Calculate the emissions for each facility
            state_ghgi_emis_value = year_emi[year_emi['state_code'] == state]['ghgi_ch4_kt'].values[0]

            year_state_echo.loc[:, 'emis_kt'] = (
                year_state_echo['flow_proportion'] * state_ghgi_emis_value
            )

            # Drop the flow proportion column
            year_state_echo = year_state_echo.drop(columns=['flow_proportion'])

            # Concatenate the GHGRP matches and non-matches for our final dataframe
            final_proxy_df = pd.concat([final_proxy_df, year_state_echo], ignore_index=True)

    return final_proxy_df


def check_facility_distance_then_add(echo_df, frs_df, filter_distance=1.0):
    """
    Vectorized function to check facility distances and add FRS facilities 
    that are at least 1 km away from any ECHO facility.
    
    Parameters:
    - echo_df: DataFrame with ECHO facility data
    - frs_df: DataFrame with FRS facility data
    - filter_distance: Distance in kilometers (default 1.0 km)
    
    Returns:
    - DataFrame with ECHO data plus filtered FRS data
    """
    # Add data source columns
    echo_df['data_source'] = 'echo'

    echo_df = echo_df[['Year', 'State', 'Facility Latitude', 'Facility Longitude', 'Wastewater Flow (MGal/yr)', 'data_source']]
  
    echo_df = echo_df.rename(columns={
        'Facility Latitude': 'latitude',
        'Facility Longitude': 'longitude',
        'State': 'state_code',
        'Year': 'year'
        })

    # Extract coordinates
    echo_coords = np.array([(lat, lon) for lat, lon in 
                           zip(echo_df['latitude'], 
                               echo_df['longitude'])])
    
    frs_coords = np.array([(lat, lon) for lat, lon in 
                          zip(frs_df['latitude'], 
                              frs_df['longitude'])])
    
    # Calculate distances using broadcasting
    lat1 = echo_coords[:, 0][:, np.newaxis]
    lon1 = echo_coords[:, 1][:, np.newaxis]
    lat2 = frs_coords[:, 0]
    lon2 = frs_coords[:, 1]
    
    # Haversine formula components
    R = 6371  # Earth's radius in kilometers
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)
    a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
    distances = 2 * R * np.arcsin(np.sqrt(a))
    
    # Create mask where facility is far enough from all ECHO facilities
    mask = ~(distances <= filter_distance).any(axis=0)
    
    # Filter FRS facilities and combine with ECHO data
    new_rows = frs_df[mask]

    # duplicate filtered facilities to create a full FRS time series instead of a single entry
    new_rows = expand_years(new_rows)
    new_rows = new_rows[new_rows['year'] >= min_year]

    result_df = pd.concat([echo_df, new_rows], ignore_index=True)
    
    return result_df


def calculate_emissions_proxy(emi_df, ghgrp_df, echo_df, year_range):
    """
    Function to calculate emissions proxy for states based on GHGI and GHGRP data.

    Parameters:
    - emi_df: DataFrame containing GHGI emissions data.
    - ghgrp_df: DataFrame containing GHGRP emissions data.
    - echo_df: DataFrame containing ECHO facility data.
    - year_range: List or range of years to process.

    Returns:
    - final_proxy_df: DataFrame containing the processed emissions data for each state.
    """

    # Create the final dataframe that we will populate below
    final_proxy_df = pd.DataFrame()

    # States that occur in the ghgi emi data
    ghgi_states = set(emi_df['state_code'].unique())

    for year in year_range:
        
        # Filter total GHGI emissions for the current year
        year_emi = emi_df[emi_df['year'] == year].copy() 

        # Filter echo_df for the current year
        year_echo = echo_df[echo_df['year'] == year].copy() 
        
        # Some states have ECHO facilities that blink in and out of existence. We need to generate our echo state list from the echo data for each year.
        # States that occur in the echo data
        echo_states = set(year_echo[(year_echo['data_source'] == 'echo') | (year_echo['data_source'] == 'ghgrp')]['state_code'].unique())

        # ghgi states that have echo data
        echo_in_ghgi = ghgi_states.intersection(echo_states)

        # ghgi states that are missing from the echo data
        ghgi_without_echo = ghgi_states - echo_states

        # Filter total GHGRP emissions for the current year
        year_total_ghgrp_emis = ghgrp_df[ghgrp_df['year'] == year]['emis_kt'].sum()

        # Calculate the difference between national GHGI and GHGRP emissions
        year_national_emis_available = year_emi['ghgi_ch4_kt'].sum() - year_total_ghgrp_emis

        # Calculate the proportion of each state's GHGI methane emissions for the current year
        year_emi.loc[:, 'state_proportion'] = year_emi['ghgi_ch4_kt'] / year_emi['ghgi_ch4_kt'].sum()

        # Allocate emissions to each state based on their proportion of total emissions
        year_emi.loc[:, 'state_ghgi_emis'] = year_emi['state_proportion'] * year_national_emis_available

        # Give 75% of the emis to ECHO and 25% to FRS
        year_emi.loc[:, 'echo_emis'] = year_emi.loc[:, 'state_ghgi_emis'] * 0.75
        year_emi.loc[:, 'frs_emis'] = year_emi.loc[:, 'state_ghgi_emis'] * 0.25
        
        # Process emissions for each state
        for state in echo_in_ghgi:
        
            year_state_echo = year_echo[year_echo['state_code'] == state].copy()
            
            # Filter out ECHO-GHGRP spatial matches or GHGRP that was added to ECHO. These already have emissions from GHGRP data. We'll add them at the end. 
            year_state_echo_ghgrp = year_state_echo[(year_state_echo['ghgrp_match'] == 1) | (year_state_echo['ghgrp_match'] == 2)].copy() 

            # ECHO with no GHGRP matches and no FRS data
            year_state_echo_no_ghgrp = year_state_echo[(year_state_echo['ghgrp_match'] == 0) 
                & (year_state_echo['data_source'] != 'frs') & (year_state_echo['data_source'] != 'brewerydb')].copy()  
            
            # Calculate the proportional flow at each facility
            year_state_echo_no_ghgrp.loc[:, 'flow_proportion'] = (
                year_state_echo_no_ghgrp['Wastewater Flow (MGal/yr)'] / 
                year_state_echo_no_ghgrp['Wastewater Flow (MGal/yr)'].sum()
            )

            # FRS data calculation
            year_state_frs = year_state_echo[(year_state_echo['data_source'] != 'echo') & (year_state_echo['data_source'] != 'ghgrp')].copy() #only return frs and brewerydb data
            
            # If there is GHGRP data but no ECHO data, allocate all FRS emissions using the state's GHGI emissions instead of the ECHO-FRS ratio emis
            if len(year_state_echo_no_ghgrp) == 0:
                frs_emis_to_allocate = year_emi[year_emi['state_code'] == state]['state_ghgi_emis'].values[0]

            else: # ECHO data exist, use the ECHO-FRS ratio emis
                frs_emis_to_allocate = year_emi[year_emi['state_code'] == state]['frs_emis'].values[0]
            
            # Allocate FRS emissions to each facility. If there are no FRS facilities, calculate ECHO without the ECHO-FRS ratio emis. 
            # Else, calculate with the ECHO-FRS ratio emis.
            try:
                with np.errstate(divide='raise'):
                    # Apply the FRS emissions equally across all facilities in the state
                    year_state_frs.loc[:, 'emis_kt'] = (
                        frs_emis_to_allocate / len(year_state_frs)
                    )
            except FloatingPointError:
                print(f"No FRS facilities for {state} in {year} in state with ECHO data")
                # Calculate the emissions for each ECHO facility using the state's GHGI emissions instead of the ECHO-FRS ratio emis
                state_ghgi_emis_value = year_emi[year_emi['state_code'] == state]['state_ghgi_emis'].values[0]

                year_state_echo_no_ghgrp.loc[:, 'emis_kt'] = (
                year_state_echo_no_ghgrp['flow_proportion'] * state_ghgi_emis_value
                )
                # Drop the flow proportion column
                year_state_echo_no_ghgrp = year_state_echo_no_ghgrp.drop(columns=['flow_proportion'])

                # Concatenate the GHGRP matches and non-matches for our final dataframe
                final_proxy_df = pd.concat([final_proxy_df, year_state_echo_ghgrp, year_state_echo_no_ghgrp, year_state_frs], ignore_index=True)
            else:
                # Calculate the emissions for each facility using the ECHO-FRS ratio emis
                state_ghgi_emis_value = year_emi[year_emi['state_code'] == state]['echo_emis'].values[0]
                
                year_state_echo_no_ghgrp.loc[:, 'emis_kt'] = (
                    year_state_echo_no_ghgrp['flow_proportion'] * state_ghgi_emis_value
                )

                # Drop the flow proportion column
                year_state_echo_no_ghgrp = year_state_echo_no_ghgrp.drop(columns=['flow_proportion'])

                # Concatenate the GHGRP matches and non-matches for our final dataframe
                final_proxy_df = pd.concat([final_proxy_df, year_state_echo_ghgrp, year_state_echo_no_ghgrp, year_state_frs], ignore_index=True)
        
        # Dealing with states without any ECHO or GHGRP data
        for state in ghgi_without_echo: 
        
            year_state_frs = year_echo[year_echo['state_code'] == state].copy()

            state_ghgi_emis_value = year_emi[year_emi['state_code'] == state]['state_ghgi_emis'].values[0]

            try:
                with np.errstate(divide='raise'):
                    # Distribute emissions equally among facilities
                    year_state_frs.loc[:, 'emis_kt'] = state_ghgi_emis_value / len(year_state_frs)
            except FloatingPointError:
                print(f"Error allocating emissions for {state} in {year} in state without ECHO data")
                continue
            
            final_proxy_df = pd.concat([final_proxy_df, year_state_frs], ignore_index=True)

    return final_proxy_df


def process_facilities_and_emissions(echo_df, ghgrp_df, emi_df, year_range, industry='Pulp and Paper'):
    """
    Process facilities data, match with GHGRP data, and distribute emissions.
    
    Parameters:
    echo_df (pd.DataFrame): ECHO dataset (this will contain ECHO, FRS, and brewerydb data, depending on the industry)
    ghgrp_df (pd.DataFrame): GHGRP dataset
    emi_df (pd.DataFrame): EPA GHGI emissions data
    year_range (list): List of years to process
    industry (str): Industry name for EPA emissions filtering
    
    Returns:
    pd.DataFrame: Processed ECHO dataset with matched and distributed emissions
    """

    # Step 1: Match facilities to GHGRP based on location
    echo_df['ghgrp_match'] = 0
    echo_df['emis_kt'] = 0
    ghgrp_df['found'] = 0

    ghgrp_df['data_source'] = 'ghgrp'
    
    # Step 1 Data wrangling
    # convert int to float to avoid potential math / dtype errors
    echo_df['emis_kt'] = echo_df['emis_kt'].astype(float)
    ghgrp_df['emis_kt_tot'] = ghgrp_df['emis_kt_tot'].astype(float)

    ghgrp_df = convert_state_names_to_codes(ghgrp_df, 'State')
    
    # Rename columns to match the desired output
    ghgrp_df = ghgrp_df.rename(columns={
            'Year': 'year',
            'State': 'state_code',
            'emis_kt_tot': 'emis_kt'
        })

    ghgrp_df = ghgrp_df[['year', 'state_code', 'latitude', 'longitude', 'data_source', 'emis_kt', 'found']]

    # Get a list of states from emi_df
    emi_states = set(emi_df['state_code'].unique())

    # Filter ECHO and GHGRP to only include states in emi_df (there are weird cases where they have data for states that don't have emissions data)
    ghgrp_df = ghgrp_df[ghgrp_df['state_code'].isin(emi_states)].copy()
    echo_df = echo_df[echo_df['state_code'].isin(emi_states)].copy()

    echo_without_frs = echo_df[echo_df['data_source'] == 'echo'].copy()
    echo_with_frs = echo_df[echo_df['data_source'] != 'echo'].copy()

    # Step 2: Add GHGRP facilities to the "master" ECHO dataframe
    # Some GHGRP data are empty after trying to join the GHGRP emi and facility data and have len == 0. Only look for spatial matches if there is GHGRP data.
    if len(ghgrp_df) != 0:
        for idx, echo_row in echo_without_frs.iterrows():
            for _, ghgrp_row in ghgrp_df.iterrows():
                dist = np.sqrt((ghgrp_row['latitude'] - echo_row['latitude'])**2 +
                            (ghgrp_row['longitude'] - echo_row['longitude'])**2)
                if dist < 0.025 and ghgrp_row['year'] == echo_row['year']:
                    ghgrp_df.loc[ghgrp_row.name, 'found'] = 1
                    echo_without_frs.loc[idx, 'ghgrp_match'] = 1
                    echo_without_frs.loc[idx, 'emis_kt'] = ghgrp_row['emis_kt']
                    break
        
        
        ghgrp_to_add = ghgrp_df[ghgrp_df['found'] == 0].copy()

        # Add the fixed columns 'ghgrp_match' and 'Wastewater Flow (MGal/yr)'
        ghgrp_to_add['ghgrp_match'] = 2
        ghgrp_to_add['Wastewater Flow (MGal/yr)'] = 0

        # Add the "not found" GHGRP facilities to the ECHO dataframe. Add the FRS data back in
        echo_df = pd.concat([echo_without_frs, echo_with_frs, ghgrp_to_add], ignore_index=True)
        
        # Set the 'found' column to 2 for GHGRP facilities not location matched with ECHO but added to ECHO
        ghgrp_df.loc[ghgrp_df['found'] == 0, 'found'] = 2
        
        # Step 3: Distribute remaining emissions difference
        final_proxy_df = calculate_emissions_proxy(emi_df, ghgrp_df, echo_df, year_range)

    # If no GHGRP data is found, assign all emissions to ECHO data
    else:
        print(f"GHGRP data not found for {industry}, assigning all emissions to ECHO data.")
        
        final_proxy_df = calculate_emissions_proxy(emi_df, ghgrp_df, echo_df, year_range)
    
    return final_proxy_df

def clean_and_group_echo_data(df, facility_type):
    """Filters, groups, and adjusts flow values based on facility type."""
    # Filter the DataFrame based on facility type and whether flows are greater than 0
    df_filtered = df[(df['Facility Type Indicator'] == facility_type) & (df['Wastewater Flow (MGal/yr)']>0)].copy()

    # Group by 'Year' and 'NPDES Permit Number', aggregating necessary columns
    grouped_df = df_filtered.groupby(['Year', 'NPDES Permit Number'], as_index=False).agg({
        'CWNS ID(s)': 'first',
        'SIC Code': 'max',
        'NAICS Code': 'max',
        'City': 'first',
        'State': 'first',
        'County': 'first',
        'Facility Latitude': 'max',
        'Facility Longitude': 'max',
        'Wastewater Flow (MGal/yr)': 'max'
    })

    grouped_df.reset_index(inplace=True, drop=True)

    return grouped_df

def read_and_filter_ghgrp_data(facility_file, emissions_file, years):
    """Read and filter GHGRP facility and emissions data."""
    facility_info = pd.read_csv(facility_file)
    facility_emis = pd.read_csv(emissions_file)
    
    # Filter emissions data for methane only and years of interest
    facility_emis = facility_emis[
        (facility_emis['ghg_name'] == 'Methane') &
        (facility_emis['reporting_year'].isin(years))
    ]
    
    # Filter facility info for years of interest
    facility_info = facility_info[facility_info['year'].isin(years)]
    
    return facility_info.reset_index(drop=True), facility_emis.reset_index(drop=True)

def rename_columns(df, column_mapping):
    """Rename columns based on a mapping dictionary."""
    return df.rename(columns=column_mapping)

def merge_facility_and_emissions_data(facility_info, facility_emis):
    """Merge facility info and emissions data."""
    ghgrp_ind = pd.merge(facility_info, facility_emis)
    ghgrp_ind['emis_kt_tot'] = ghgrp_ind['ghg_quantity'] / 1e3  # convert to kt
    return ghgrp_ind

def filter_by_naics(df, naics_codes):
    """
    Filter dataframe by NAICS code prefix(es).
    
    Parameters:
    df (pd.DataFrame): DataFrame containing NAICS codes
    naics_codes (str or list): Single NAICS code/prefix or list of NAICS codes
    
    Returns:
    pd.DataFrame: Filtered DataFrame containing only rows matching NAICS code(s)
    """
    # Convert single NAICS code to list for consistency
    if isinstance(naics_codes, str):
        naics_codes = [naics_codes]
        
    # Filter using tuple of NAICS codes
    return df[df['NAICS Code'].str.startswith(tuple(naics_codes))].copy().reset_index(drop=True)

def compare_state_sets(df1, df2, state_column1, state_column2):
    """
    Function to compare the unique sets of state abbreviations between two DataFrames.
    
    Parameters:
    - df1: First DataFrame
    - df2: Second DataFrame
    - state_column1: Column name of the state abbreviations in the first DataFrame
    - state_column2: Column name of the state abbreviations in the second DataFrame
    
    Returns:
    - A message indicating if the sets of states are identical or which states are missing from the second DataFrame.
    """
    
    # Extract unique states from both DataFrames
    unique_states_df1 = set(df1[state_column1].unique())
    unique_states_df2 = set(df2[state_column2].unique())

    # Check if the sets are the same
    if unique_states_df1 == unique_states_df2:
        return "Both DataFrames have the same unique set of states."
    else:
        # Find the states that are in df1 but missing from df2
        missing_states = unique_states_df1 - unique_states_df2
        sorted_missing_states = sorted(list(missing_states))
        return f"The following states are missing from echo: {sorted_missing_states}"
    

def subset_industrial_sector(frs_facility_path, frs_naics_path, sector_name, naics_prefixes):
    """
    Subset a dataset of industrial sectors using DuckDB.
    
    Parameters:
    frs_facility_path (str): Path to the FRS facility file.
    frs_naics_path (str): Path to the FRS NAICS file.
    sector_name (str): Name of the industrial sector.
    naics_prefixes (str or list): Single NAICS code prefix or list of NAICS code prefixes.
    
    Returns:
    pd.DataFrame: A DataFrame containing the subset of facilities for the given sector.
    """
    # Convert single prefix to list for consistency
    if isinstance(naics_prefixes, str):
        naics_prefixes = [naics_prefixes]
    
    # Build WHERE clause for multiple NAICS codes
    where_conditions = " OR ".join([f"CAST(naics_code AS VARCHAR) LIKE '{prefix}%'" 
                                  for prefix in naics_prefixes])
    
    query = f"""
    SELECT 
        frs.primary_name AS name, 
        frs.latitude83 AS latitude, 
        frs.longitude83 AS longitude,
        frs.state_code AS state_code,
        frs.create_date AS create_date,
        frs.update_date AS update_date,
        frs_naics.naics_code AS naics_code
    FROM 
        (SELECT registry_id, primary_name, latitude83, longitude83, state_code, create_date, update_date 
         FROM '{frs_facility_path}') AS frs
    JOIN 
        (SELECT registry_id, naics_code FROM '{frs_naics_path}') AS frs_naics
    ON 
        frs.registry_id = frs_naics.registry_id
    WHERE 
        {where_conditions}
    """
    
    frs_df = duckdb.query(query).df()
    
    # Process the data
    frs_df = frs_df.dropna(subset=['latitude', 'longitude'])
    frs_df = frs_df.drop_duplicates(subset=['latitude', 'longitude'])
    frs_df['year'] = 2012 # We are going to use 2012 as the start year for all FRS facilities. Some states have missing ECHO, GHGRP time series data (or no data at all) and FRS will be used to fill in the gaps.
    # frs_df['year'] = pd.to_datetime(frs_df['create_date'], format='%d-%b-%y').dt.year # create a "start year" from the create date
    frs_df['data_source'] = 'frs'
    frs_df = frs_df[['year', 'state_code', 'latitude', 'longitude', 'data_source']]

    return frs_df


def expand_years(df, end_year=2022):
    """
    Duplicate rows for each year between year and end_year. This creates a time series for a single facility entry.
    
    Parameters:
    df (pd.DataFrame): Facility DataFrame with 'year' column that represents when the proxy time series starts
    end_year (int): Final year to generate data for (default 2022)
    
    Returns:
    pd.DataFrame: Expanded DataFrame with rows for each year
    """
    # Create empty list to store expanded DataFrames
    expanded_dfs = []
    
    # Process each facility
    for _, row in df.iterrows():
        # Create date range from year (starting year of data) to end_year
        years = pd.date_range(
            start=str(row['year']), 
            end=str(end_year + 1), 
            freq='YE'
        ).year
        
        # Create duplicate rows for each year
        temp_df = pd.DataFrame([row.to_dict() for _ in range(len(years))])
        temp_df['year'] = years
        
        expanded_dfs.append(temp_df)
    
    # Combine all expanded DataFrames
    result_df = pd.concat(expanded_dfs, ignore_index=True)
    
    return result_df


def geocode_address(df, address_column):
    """
    Geocode addresses using Nominatim, handling missing values
    
    Parameters:
    df: DataFrame containing addresses
    address_column: Name of column containing addresses
    
    Returns:
    DataFrame with added latitude and longitude columns
    """
    # Check if address column exists
    if address_column not in df.columns:
        raise ValueError(f"Column {address_column} not found in DataFrame")
        
    # Initialize geocoder
    geolocator = Nominatim(user_agent="my_app")
    
    # Create cache dictionary
    geocode_cache = {}
    
    def get_lat_long(address):
        # Handle missing values
        if pd.isna(address) or str(address).strip() == '':
            return (None, None)
            
        # Check cache first
        if address in geocode_cache:
            return geocode_cache[address]
        
        try:
            # Add delay to respect Nominatim's usage policy
            time.sleep(1)
            location = geolocator.geocode(str(address))
            if location:
                result = (location.latitude, location.longitude)
                geocode_cache[address] = result
                return result
            return (None, None)
            
        except (GeocoderTimedOut, GeocoderServiceError):
            return (None, None)
    
    # Create lat_long column
    df['lat_long'] = None
    
    # Only geocode where we need coordinates
    mask = (df['longitude'].isna() | (df['longitude'] == '')) & df[address_column].notna()
    df.loc[mask, 'lat_long'] = df.loc[mask, address_column].apply(get_lat_long)
    
    # Split tuple into separate columns
    df['latitude'] = df['lat_long'].apply(lambda x: x[0] if x else None)
    df['longitude'] = df['lat_long'].apply(lambda x: x[1] if x else None)
    
    # Drop temporary column
    df = df.drop('lat_long', axis=1)
    
    return df


def get_brewery_data(states: List[str], brewery_types: List[str]) -> pd.DataFrame:
    """
    Fetch brewery data for all states and specified brewery types
    
    Parameters:
    states: List of state names
    brewery_types: List of brewery types to fetch
    
    Returns:
    DataFrame containing combined brewery data

    # api info: https://www.openbrewerydb.org/documentation
    """
    url = 'https://api.openbrewerydb.org/v1/breweries?'
    all_data = []
    
    for state in states:
        for brewery_type in brewery_types:
            params = {
                'by_state': state.lower(),
                'by_type': brewery_type,
                'per_page': 200
            }
            
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    data = response.json()
                    if data:  # Check if data is not empty
                        all_data.extend(data)
                    print(f"Successfully fetched data for {state} - {brewery_type}")
                else:
                    print(f"Failed to fetch data for {state} - {brewery_type}")
                
                # Rate limiting - be nice to the API
                time.sleep(0.5)
                
            except Exception as e:
                print(f"Error fetching data for {state} - {brewery_type}: {str(e)}")
                continue
    
    # Convert to DataFrame
    if all_data:
        return pd.DataFrame(all_data)
    else:
        return pd.DataFrame()

# %% Step 2.1 Read in and process FRS data

# Subset the FRS data to only those sectors we need 

frs_naics_path = V3_DATA_PATH / "global/NATIONAL_NAICS_FILE.CSV"
frs_facility_path = V3_DATA_PATH / "global/NATIONAL_FACILITY_FILE.CSV"

# Example usage:
naics_codes = {
    'pp': '3221',
    'mp': '3116',
    'fv': ['3114', '311421', '311991', '311340', '312130'],
    'ethanol': '325193',
    'brew': '312120',
    'petrref': '32411'
}

sector_dataframes = {}
for sector_name, naics_prefix in naics_codes.items():
    sector_df = subset_industrial_sector(frs_facility_path, frs_naics_path, sector_name, naics_prefix)
    sector_dataframes[sector_name] = sector_df
    print(f"Processed {sector_name} sector. Shape: {sector_df.shape}")

# Access individual sector dataframes
frs_pp = sector_dataframes['pp']
frs_mp = sector_dataframes['mp']
frs_fv = sector_dataframes['fv']
frs_ethanol = sector_dataframes['ethanol']
frs_brew = sector_dataframes['brew']
frs_petrref = sector_dataframes['petrref']

# %% Step 2.1.1 Add missing MT fruit and veg data to the FRS data
# add a found MT fruit and vegetable processing location to the frs_fv dataset because there are no data for MT from FRS, GHGRP, or ECHO
# https://abundantmontana.com/amt-lister/mission-mountain-food-enterprise-center/
MT_fv_site = [min_year, 'MT', 47.528500, -114.103880, 'frs']
new_fv_row = pd.DataFrame([MT_fv_site], columns=frs_fv.columns)

# Concatenate the existing DataFrame with the new row
frs_fv = pd.concat([frs_fv, new_fv_row], ignore_index=True)

# %% Step 2.2 Read in and process brewery db data - these data fill in missing brewery proxy data from FRS, ECHO, and GHGRP
# We assume that these breweries exist throughout the full time series 

# List of all US states to push through the brewery data fetch function
us_states = [
    'Alabama', 'Alaska', 'Arizona', 'Arkansas', 'California', 'Colorado', 'Connecticut', 
    'Delaware', 'District_of_Columbia', 'Florida', 'Georgia', 'Hawaii', 'Idaho', 'Illinois', 'Indiana', 'Iowa', 
    'Kansas', 'Kentucky', 'Louisiana', 'Maine', 'Maryland', 'Massachusetts', 'Michigan', 
    'Minnesota', 'Mississippi', 'Missouri', 'Montana', 'Nebraska', 'Nevada', 'New_Hampshire',
    'New_Jersey', 'New_Mexico', 'New_York', 'North_Carolina', 'North_Dakota', 'Ohio', 
    'Oklahoma', 'Oregon', 'Pennsylvania', 'Rhode_Island', 'South_Carolina', 'South_Dakota',
    'Tennessee', 'Texas', 'Utah', 'Vermont', 'Virginia', 'Washington', 'West_Virginia',
    'Wisconsin', 'Wyoming'
]

brewery_df_path = proxy_data_dir_path / "wastewater"
brewery_file = brewery_df_path / "openbrewerydb_geolocated.csv"

if brewery_file.exists():
    print("Reading existing brewery data from file...")
    brewery_df = pd.read_csv(brewery_file)
else:
    print("Fetching brewery data from API...")
    # Brewery types to fetch
    brewery_types = ['micro', 'regional', 'large']

    # Fetch data
    brewery_df = get_brewery_data(us_states, brewery_types)
    
    # Display results
    print(f"Total breweries found: {len(brewery_df)}")

    # Create a full address from the address components to allow for geocoding missing lat/long values
    # Convert long zips to standard 5 digit zips since nominatim doesn't recognize long zips
    brewery_df['postal_code'] = brewery_df['postal_code'].fillna('').str.split('-').str[0]

    brewery_df['address_1'] = (
        brewery_df['address_1'].fillna('') + ', ' + 
        brewery_df['city'].fillna('') + ', ' + 
        brewery_df['state_province'].fillna('') + ' ' + 
        brewery_df['postal_code'].fillna('')
    ).str.strip(', ')

    # Clean up any double commas or spaces from missing values
    brewery_df['address_1'] = (
        brewery_df['address_1']
        .str.replace(r',\s*,', ',', regex=True)
        .str.strip()
    )

    # Subset to only those records missing lat long data
    brewery_df_missing_latlongs = brewery_df[brewery_df['latitude'].isna() | brewery_df['longitude'].isna()]

    # Geocode the missing lat longs (this can take a long time)
    added_lat_longs = geocode_address(brewery_df_missing_latlongs, 'address_1')

    # Remove any rows that still have missing lat longs
    added_lat_longs = added_lat_longs.dropna(subset=['latitude', 'longitude'])

    # Combine the original data with the newly geocoded data
    brewery_df = pd.concat([brewery_df[~brewery_df['latitude'].isna()], added_lat_longs])

    # Change state names to state codes
    brewery_df = convert_state_names_to_codes(brewery_df, 'state')

    # Rename columns to match the FRS data
    brewery_df = brewery_df.rename(columns={'state': 'state_code'})

    # Change data to match the FRS data
    brewery_df['year'] = min_year 
    brewery_df['data_source'] = 'brewerydb'
    brewery_df = brewery_df[['year', 'state_code', 'latitude', 'longitude', 'data_source']]

    # Save data to CSV
    brewery_df.to_csv(brewery_file, index=False)


# %% Step 2.2.2 Deduplicate brewery facilities by latitude and longitude - breweries will be removed if they occur within 0.25 km of each other
filter_distance = 0.25 # 0.25 km

# Extract coordinates
frs_brew_coords = np.array([(lat, lon) for lat, lon in 
                        zip(frs_brew['latitude'], 
                            frs_brew['longitude'])])

brewery_df_coords = np.array([(lat, lon) for lat, lon in 
                        zip(brewery_df['latitude'], 
                            brewery_df['longitude'])])

# Calculate distances using broadcasting
lat1 = frs_brew_coords[:, 0][:, np.newaxis]
lon1 = frs_brew_coords[:, 1][:, np.newaxis]
lat2 = brewery_df_coords[:, 0]
lon2 = brewery_df_coords[:, 1]

# Haversine formula components
R = 6371  # Earth's radius in kilometers
dlat = np.radians(lat2 - lat1)
dlon = np.radians(lon2 - lon1)
a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2
distances = 2 * R * np.arcsin(np.sqrt(a))

# Create mask where facility is far enough from all ECHO facilities
mask = ~(distances <= filter_distance).any(axis=0)

# Filter FRS facilities and combine with brewery data.
new_rows = brewery_df[mask]

frs_brew = pd.concat([frs_brew, new_rows], ignore_index=True)

### FRS now contains brewery data that will be used in the ww_brew proxy data
# %% Step 2.3 ECHO data processing

echo_file_directory = proxy_data_dir_path / "wastewater/ECHO"
combined_echo_file_path = echo_file_directory / "combined_echo_data.csv"

# Check if the combined CSV already exists, read it if it does, otherwise create it
if combined_echo_file_path.exists():
    echo_full = read_combined_file(combined_echo_file_path)
else:
    echo_full = read_and_combine_csv_files(echo_file_directory)


# %% Step 2.3.1 Process the potw and non-potw facilities

# Process POTW facilities
echo_potw = clean_and_group_echo_data(echo_full, 'POTW')

# Process NON-POTW facilities
echo_nonpotw = clean_and_group_echo_data(echo_full, 'NON-POTW')

# %% Step 2.3.2 Create dataframes for each non-potw industry

echo_nonpotw['NAICS Code'] = echo_nonpotw['NAICS Code'].astype(str)
echo_nonpotw['SIC Code'] = echo_nonpotw['SIC Code'].astype(str)

# Define the industries and their corresponding NAICS prefixes and SIC codes
industries = {
    'pp': ('3221', ['2611', '2621', '2631']),
    'mp': ('3116', ['0751', '2011', '2048', '2013', '5147', '2077', '2015']),
    'fv': (['3114', '311421', '311991', '311340', '312130'], ['2037', '2038', '2033', '2035', '2032', '2034', '2099']),
    'eth': ('325193', ['2869']),
    'brew': ('312120', ['2082']),
    'petrref': ('32411', ['2911'])
}

# Process all industries
for industry_name, (naics_prefix, sic_codes) in industries.items():
    result_df = extract_industry_facilities(echo_nonpotw, industry_name, naics_prefix, sic_codes)
    exec(f"echo_{industry_name.lower()} = result_df")

# Check if there are missing states between the ECHO and GHGI emi data.
# TO DO: this should check missing states by year to ensure that we have a complete time series
print("Comparing Pulp and Paper")
print(compare_state_sets(ww_pp_emi, echo_pp, 'state_code', 'State'))

print("Comparing Meat and Poultry")
print(compare_state_sets(ww_mp_emi, echo_mp, 'state_code', 'State'))

print("Comparing Fruits and Vegetables")
print(compare_state_sets(ww_fv_emi, echo_fv, 'state_code', 'State'))

print("Comparing Ethanol")
print(compare_state_sets(ww_ethanol_emi, echo_eth, 'state_code', 'State'))

print("Comparing Breweries")
print(compare_state_sets(ww_brew_emi, echo_brew, 'state_code', 'State'))

print("Comparing Petroleum Refining")
print(compare_state_sets(ww_petrref_emi, echo_petrref, 'state_code', 'State'))

# %% Step 2.5 Add the FRS data to the ECHO data

echo_pp = check_facility_distance_then_add(echo_pp, frs_pp)
echo_mp = check_facility_distance_then_add(echo_mp, frs_mp)
echo_fv = check_facility_distance_then_add(echo_fv, frs_fv)
echo_eth = check_facility_distance_then_add(echo_eth, frs_ethanol)
echo_brew = check_facility_distance_then_add(echo_brew, frs_brew)
echo_petrref = check_facility_distance_then_add(echo_petrref, frs_petrref)

print("Comparing Pulp and Paper")
print(compare_state_sets(ww_pp_emi, echo_pp, 'state_code', 'state_code'))

print("Comparing Meat and Poultry")
print(compare_state_sets(ww_mp_emi, echo_mp, 'state_code', 'state_code'))

print("Comparing Fruits and Vegetables")
print(compare_state_sets(ww_fv_emi, echo_fv, 'state_code', 'state_code'))

print("Comparing Ethanol")
print(compare_state_sets(ww_ethanol_emi, echo_eth, 'state_code', 'state_code'))

print("Comparing Breweries")
print(compare_state_sets(ww_brew_emi, echo_brew, 'state_code', 'state_code'))

print("Comparing Petroleum Refining")
print(compare_state_sets(ww_petrref_emi, echo_petrref, 'state_code', 'state_code'))


# %%  Step 2.6 Read in GHGRP Subpart II Data

# Main processing
facility_info, facility_emis = read_and_filter_ghgrp_data(ghgrp_facility_ii_inputfile, ghgrp_emi_ii_inputfile, years=years)

# Column renaming mappings
facility_info_columns = {
    'primary_naics': 'NAICS Code',
    'state_name': 'State',
    'year': 'Year',
}

facility_emis_columns = {
    'reporting_year': 'Year',

}

facility_info = rename_columns(facility_info, facility_info_columns)
facility_emis = rename_columns(facility_emis, facility_emis_columns)

ghgrp_ind = merge_facility_and_emissions_data(facility_info, facility_emis)
ghgrp_ind['NAICS Code'] = ghgrp_ind['NAICS Code'].astype(str)

# Filter data for different industries
industry_filters = {
    'pp': '322',      # Pulp and paper
    'mp': '3116',     # Red meat and poultry
    'fv': ['3114', '311421', '311991', '311340', '312130'],     # Fruits and vegetables
    'eth': '325193',  # Ethanol production
    'brew': '312120', # Breweries
    'ref': '324121'   # Petroleum refining
}

ghgrp_industries = {key: filter_by_naics(ghgrp_ind, value) for key, value in industry_filters.items()}

# Create distinct dataframes for each industry
ghgrp_pp = ghgrp_industries['pp']
ghgrp_mp = ghgrp_industries['mp']
ghgrp_fv = ghgrp_industries['fv']
ghgrp_eth = ghgrp_industries['eth']
ghgrp_brew = ghgrp_industries['brew']
ghgrp_ref = ghgrp_industries['ref']

dfs = [ghgrp_pp, ghgrp_mp, ghgrp_fv, ghgrp_eth, ghgrp_brew, ghgrp_ref]

ghgrp_combined = pd.concat(dfs, ignore_index=True)

total_ghgrp_emis = ghgrp_combined[['Year', 'State', 'emis_kt_tot']]

# %%
emi_df = ww_mp_emi
echo_df = echo_mp
ghgrp_df = ghgrp_mp
year = 2012

# %%
echo_df['ghgrp_match'] = 0
echo_df['emis_kt'] = 0
ghgrp_df['found'] = 0

ghgrp_df['data_source'] = 'ghgrp'

# Step 1.1 Data wrangling
# convert int to float to avoid potential math / dtype errors
echo_df['emis_kt'] = echo_df['emis_kt'].astype(float)
ghgrp_df['emis_kt_tot'] = ghgrp_df['emis_kt_tot'].astype(float)

ghgrp_df = convert_state_names_to_codes(ghgrp_df, 'State')

# Rename columns to match the desired output
ghgrp_df = ghgrp_df.rename(columns={
        'Year': 'year',
        'State': 'state_code',
        'emis_kt_tot': 'emis_kt'
    })

ghgrp_df = ghgrp_df[['year', 'state_code', 'latitude', 'longitude', 'data_source', 'emis_kt', 'found']]

# Get a list of states from emi_df
emi_states = set(emi_df['state_code'].unique())

# Filter ECHO and GHGRP to only include states in emi_df (there are weird cases where they have data for states that don't have emissions data)
ghgrp_df = ghgrp_df[ghgrp_df['state_code'].isin(emi_states)].copy()
echo_df = echo_df[echo_df['state_code'].isin(emi_states)].copy()

echo_without_frs = echo_df[echo_df['data_source'] == 'echo'].copy()
echo_with_frs = echo_df[echo_df['data_source'] != 'echo'].copy()

# Some GHGRP data are empty after trying to join the GHGRP emi and facility data and have len == 0. Only look for spatial matches if there is GHGRP data.
# Step 1. Find spatial matches between national ECHO and national GHGRP data. We are ignoring FRS data for matching because we already spatially deduplicated the FRS - ECHO data.
if len(ghgrp_df) != 0:
    for idx, echo_row in echo_without_frs.iterrows():
        for _, ghgrp_row in ghgrp_df.iterrows():
            dist = np.sqrt((ghgrp_row['latitude'] - echo_row['latitude'])**2 +
                        (ghgrp_row['longitude'] - echo_row['longitude'])**2)
            if dist < 0.025 and ghgrp_row['year'] == echo_row['year']:
                ghgrp_df.loc[ghgrp_row.name, 'found'] = 1
                echo_without_frs.loc[idx, 'ghgrp_match'] = 1
                echo_without_frs.loc[idx, 'emis_kt'] = ghgrp_row['emis_kt']
                break
    
    # Step 2: Add GHGRP facilities not found in ECHO to the "master" ECHO dataframe
    ghgrp_to_add = ghgrp_df[ghgrp_df['found'] == 0].copy()

    # Add the fixed columns 'ghgrp_match' and 'Wastewater Flow (MGal/yr)'
    ghgrp_to_add['ghgrp_match'] = 2
    ghgrp_to_add['Wastewater Flow (MGal/yr)'] = 0

    # Add the "not found" GHGRP facilities to the ECHO dataframe. Add the FRS data back in
    echo_df = pd.concat([echo_without_frs, echo_with_frs, ghgrp_to_add], ignore_index=True)
    
    # Set the 'found' column to 2 for GHGRP facilities not location matched with ECHO but added to ECHO
    ghgrp_df.loc[ghgrp_df['found'] == 0, 'found'] = 2
    
# %%
# Create the final dataframe that we will populate below
final_proxy_df = pd.DataFrame()

# # States that occur in the ghgi emi data
ghgi_states = set(emi_df['state_code'].unique())

year_emi = emi_df[emi_df['year'] == year].copy() 

# Filter echo_df for the current year
year_echo = echo_df[echo_df['year'] == year].copy() 

# Some states have ECHO facilities that blink in and out of existence. We need to generate our echo state list from the echo data for each year.
# States that occur in the echo data
echo_states = set(year_echo[(year_echo['data_source'] == 'echo') | (year_echo['data_source'] == 'ghgrp')]['state_code'].unique())

# ghgi states that have echo data
echo_in_ghgi = ghgi_states.intersection(echo_states)

# ghgi states that are missing from the echo data
ghgi_without_echo = ghgi_states - echo_states

# Filter total GHGRP emissions for the current year
year_total_ghgrp_emis = ghgrp_df[ghgrp_df['year'] == year]['emis_kt'].sum()

# Calculate the difference between national GHGI and GHGRP emissions
year_national_emis_available = year_emi['ghgi_ch4_kt'].sum() - year_total_ghgrp_emis

# Calculate the proportion of each state's GHGI methane emissions for the current year
year_emi.loc[:, 'state_proportion'] = year_emi['ghgi_ch4_kt'] / year_emi['ghgi_ch4_kt'].sum()

# Allocate emissions to each state based on their proportion of total emissions
year_emi.loc[:, 'state_ghgi_emis'] = year_emi['state_proportion'] * year_national_emis_available

# Give 75% of the emis to ECHO and 25% to FRS
year_emi.loc[:, 'echo_emis'] = year_emi.loc[:, 'state_ghgi_emis'] * 0.75
year_emi.loc[:, 'frs_emis'] = year_emi.loc[:, 'state_ghgi_emis'] * 0.25


# %% 
final_proxy_df = pd.DataFrame()
state = 'NE'

year_state_echo = year_echo[year_echo['state_code'] == state].copy()

# Filter out ECHO-GHGRP spatial matches or GHGRP that was added to ECHO. These already have emissions from GHGRP data. We'll add them at the end. 
year_state_echo_ghgrp = year_state_echo[(year_state_echo['ghgrp_match'] == 1) | (year_state_echo['ghgrp_match'] == 2)].copy() 

# ECHO with no GHGRP matches and no FRS data
year_state_echo_no_ghgrp = year_state_echo[(year_state_echo['ghgrp_match'] == 0) 
    & (year_state_echo['data_source'] != 'frs') & (year_state_echo['data_source'] != 'brewerydb')].copy()  

# Calculate the proportional flow at each facility
year_state_echo_no_ghgrp.loc[:, 'flow_proportion'] = (
    year_state_echo_no_ghgrp['Wastewater Flow (MGal/yr)'] / 
    year_state_echo_no_ghgrp['Wastewater Flow (MGal/yr)'].sum()
)

# FRS data calculation
year_state_frs = year_state_echo[(year_state_echo['data_source'] != 'echo') & (year_state_echo['data_source'] != 'ghgrp')].copy() #only return frs and brewerydb data

# If there is GHGRP data but no ECHO data, allocate all FRS emissions using the state's GHGI emissions instead of the ECHO-FRS ratio emis
if len(year_state_echo_no_ghgrp) == 0:
    frs_emis_to_allocate = year_emi[year_emi['state_code'] == state]['state_ghgi_emis'].values[0]

else: # ECHO data exist, use the ECHO-FRS ratio emis
    frs_emis_to_allocate = year_emi[year_emi['state_code'] == state]['frs_emis'].values[0]

# Allocate FRS emissions to each facility. If there are no FRS facilities, calculate ECHO without the ECHO-FRS ratio emis. Else, calculate with the ECHO-FRS ratio emis.
try:
    with np.errstate(divide='raise'):
        # Apply the FRS emissions equally across all facilities in the state
        year_state_frs.loc[:, 'emis_kt'] = (
            frs_emis_to_allocate / len(year_state_frs)
        )
except FloatingPointError:
    print(f"No FRS facilities for {state} in {year} in state with ECHO data")
    # Calculate the emissions for each ECHO facility using the state's GHGI emissions instead of the ECHO-FRS ratio emis
    state_ghgi_emis_value = year_emi[year_emi['state_code'] == state]['state_ghgi_emis'].values[0]

    year_state_echo_no_ghgrp.loc[:, 'emis_kt'] = (
    year_state_echo_no_ghgrp['flow_proportion'] * state_ghgi_emis_value
    )
    # Drop the flow proportion column
    year_state_echo_no_ghgrp = year_state_echo_no_ghgrp.drop(columns=['flow_proportion'])

    # Concatenate the GHGRP matches and non-matches for our final dataframe
    final_proxy_df = pd.concat([final_proxy_df, year_state_echo_ghgrp, year_state_echo_no_ghgrp, year_state_frs], ignore_index=True)
else:
    # Calculate the emissions for each facility using the ECHO-FRS ratio emis
    state_ghgi_emis_value = year_emi[year_emi['state_code'] == state]['echo_emis'].values[0]
    
    year_state_echo_no_ghgrp.loc[:, 'emis_kt'] = (
        year_state_echo_no_ghgrp['flow_proportion'] * state_ghgi_emis_value
    )

    # Drop the flow proportion column
    year_state_echo_no_ghgrp = year_state_echo_no_ghgrp.drop(columns=['flow_proportion'])

    # Concatenate the GHGRP matches and non-matches for our final dataframe
    final_proxy_df = pd.concat([final_proxy_df, year_state_echo_ghgrp, year_state_echo_no_ghgrp, year_state_frs], ignore_index=True)

# %%
for state in ghgi_without_echo: 
    year_state_frs = year_echo[year_echo['state_code'] == state].copy()

    state_ghgi_emis_value = year_emi[year_emi['state_code'] == state]['state_ghgi_emis'].values[0]
    
    year_state_frs.loc[:, 'emis_kt'] = (
        state_ghgi_emis_value / len(year_state_frs)
    )
    final_proxy_df = pd.concat([final_proxy_df, year_state_frs], ignore_index=True)

# %% Step 2.7 Join GHGRP, ECHO, and emi data

final_pp = process_facilities_and_emissions(echo_pp, ghgrp_pp, ww_pp_emi, years, industry='Pulp and Paper')
final_mp = process_facilities_and_emissions(echo_mp, ghgrp_mp, ww_mp_emi, years, industry='Meat and Poultry')
final_fv = process_facilities_and_emissions(echo_fv, ghgrp_fv, ww_fv_emi, years, industry='Fruit and Vegetables')
final_eth = process_facilities_and_emissions(echo_eth, ghgrp_eth, ww_ethanol_emi, years, industry='Ethanol')
final_brew = process_facilities_and_emissions(echo_brew, ghgrp_brew, ww_brew_emi, years, industry='Breweries')
final_ref = process_facilities_and_emissions(echo_petrref, ghgrp_ref, ww_petrref_emi, years, industry='Petroleum Refining')


# debug_df_no_ghgrp = final_pp[final_pp['ghgrp_match'] == 0]
# debug_df_grouped = debug_df_no_ghgrp.groupby(['year', 'state_code'])['emis_kt'].sum().reset_index()
debug_ghgrp = final_fv[final_fv['ghgrp_match'] != 0].groupby(['year', 'state_code'])['emis_kt'].sum().reset_index()

# final_fv.groupby(['year'])['emis_kt'].sum().reset_index()
# ww_fv_emi.groupby(['year'])['ghgi_ch4_kt'].sum().reset_index()

# %%
# Round values and compare
def compare_emissions(year_emi, debug_df_grouped):

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'state_code': year_emi['state_code'],
        'year_emi': year_emi['state_ghgi_emis'],
        'debug_df': debug_df_grouped['emis_kt']
    })
    
    # Add difference column
    comparison['difference'] = comparison['year_emi'] - comparison['debug_df']
    
    # Show only rows with differences
    differences = comparison[comparison['difference'] != 0]
    
    if len(differences) > 0:
        print("Found differences:")
        print(differences)
    else:
        print("All values match within 5 significant digits")
    
    return comparison

comparison_df = compare_emissions(year_emi, debug_df_grouped)

# %%
# Step 2.7 handle the domestic proxy data

non_septic_proxy = calculate_potw_emissions_proxy(ww_dom_nonseptic_emi, echo_potw, years)

# Run this to check if the emis are all the same between the proxy and the GHGI data

# for year in non_septic_proxy['Year'].unique():
#     year_emi = ww_dom_nonseptic_emi[ww_dom_nonseptic_emi['year'] == year].copy()
#     year_proxy = non_septic_proxy[non_septic_proxy['Year'] == year].copy()

#     # Calculate the total emissions for the current year
#     ghgi_total_emissions = year_emi['ghgi_ch4_kt'].sum()
#     proxy_total_emissions = year_proxy['emis_kt'].sum()

#     # Calculate the difference between the total emissions and the proxy emissions
#     emissions_difference = ghgi_total_emissions - proxy_total_emissions
#     # print(f"Year: {year}, GHGI Total Emissions: {ghgi_total_emissions:.2f}, Proxy Total Emissions: {proxy_total_emissions:.2f}")
#     # print(f"difference is:", {emissions_difference})

#     for state in year_proxy['State'].unique():
#         state_emi = year_emi[year_emi['state_code'] == state].copy()
#         state_proxy = year_proxy[year_proxy['State'] == state].copy()

#         # Calculate the total emissions for the current state
#         state_ghgi_emissions = state_emi['ghgi_ch4_kt'].sum()
#         state_proxy_emissions = state_proxy['emis_kt'].sum()

#         # Calculate the difference between the total emissions and the proxy emissions
#         state_emissions_difference = state_ghgi_emissions - state_proxy_emissions
#         print(f"State: {state}, GHGI State Emissions: {state_ghgi_emissions:.2f}, Proxy State Emissions: {state_proxy_emissions:.2f}")
#         print(f"difference is:", {state_emissions_difference})


# %%

# Step 2.8 Create final proxy dataframe that is ready for mapping
def create_final_proxy_df(final_proxy_df):   
    
    # process to create a final proxy dataframe

    final_proxy_df['est_ch4'] = final_proxy_df['emis_kt'] / sum(final_proxy_df['emis_kt'])
    
    # create geometry column
    final_proxy_df['geometry'] = final_proxy_df.apply(lambda row: Point(row['Facility Longitude'], row['Facility Latitude']) if pd.notnull(row['Facility Longitude']) and pd.notnull(row['Facility Latitude']) else None, axis=1)
    
    # subset to only include the columns we want to keep
    final_proxy_df = final_proxy_df[['State', 'Year', 'Facility Latitude', 'Facility Longitude', 'emis_kt', 'geometry']]

    final_proxy_df


