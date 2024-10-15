# %%
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile
import calendar
import datetime

from pyarrow import parquet
from io import StringIO
import pandas as pd
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

from gch4i.utils import name_formatter

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

#proxy mapping file
Wastewater_Mapping_inputfile = proxy_data_dir_path / "wastewater/WastewaterTreatment_ProxyMapping.xlsx"

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

def extract_industry_facilities(echo_nonpotw, industry_name, naics_prefix, sic_codes):
    """
    Extract industry-specific facilities from the non-POTW list.
    
    Parameters:
    echo_nonpotw (pd.DataFrame): The input DataFrame containing non-POTW facilities.
    industry_name (str): The name of the industry (used for the output variable name).
    naics_prefix (str): The NAICS code prefix for the industry.
    sic_codes (list): A list of SIC codes for the industry.
    
    Returns:
    pd.DataFrame: A DataFrame containing the extracted industry-specific facilities.
    """
    industry_df = echo_nonpotw[
        (echo_nonpotw['NAICS Code'].str.startswith(naics_prefix)) |
        (echo_nonpotw['SIC Code'].str.contains('|'.join(sic_codes)))
    ].copy()
    
    industry_df.reset_index(inplace=True, drop=True)
    
    return industry_df


def convert_state_names_to_codes(df, state_column):
    """
    Convert full state names in a DataFrame column to two-letter state codes.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the state names.
    state_column (str): The name of the column containing the full state names.

    Returns:
    pd.DataFrame: DataFrame with the state column changed to two-letter state codes.
    """
    
    # Dictionary mapping full state names to their two-letter codes
    state_name_to_code = {
        'Alabama': 'AL', 'Alaska': 'AK', 'Arizona': 'AZ', 'Arkansas': 'AR', 'California': 'CA',
        'Colorado': 'CO', 'Connecticut': 'CT', 'Delaware': 'DE', 'Florida': 'FL', 'Georgia': 'GA',
        'Hawaii': 'HI', 'Idaho': 'ID', 'Illinois': 'IL', 'Indiana': 'IN', 'Iowa': 'IA', 'Kansas': 'KS',
        'Kentucky': 'KY', 'Louisiana': 'LA', 'Maine': 'ME', 'Maryland': 'MD', 'Massachusetts': 'MA',
        'Michigan': 'MI', 'Minnesota': 'MN', 'Mississippi': 'MS', 'Missouri': 'MO', 'Montana': 'MT',
        'Nebraska': 'NE', 'Nevada': 'NV', 'New Hampshire': 'NH', 'New Jersey': 'NJ', 'New Mexico': 'NM',
        'New York': 'NY', 'North Carolina': 'NC', 'North Dakota': 'ND', 'Ohio': 'OH', 'Oklahoma': 'OK',
        'Oregon': 'OR', 'Pennsylvania': 'PA', 'Rhode Island': 'RI', 'South Carolina': 'SC', 'South Dakota': 'SD',
        'Tennessee': 'TN', 'Texas': 'TX', 'Utah': 'UT', 'Vermont': 'VT', 'Virginia': 'VA', 'Washington': 'WA',
        'West Virginia': 'WV', 'Wisconsin': 'WI', 'Wyoming': 'WY'
    }
    # Convert all caps to title case
    df[state_column] = df[state_column].str.title()

    # Map the full state names to their two-letter codes using the dictionary
    df[state_column] = df[state_column].map(state_name_to_code)
    
    return df


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

    # Get the unique states in the ECHO data
    unique_states = emi_df['state_code'].unique()

    for year in year_range:
        
        # Filter total GHGI emissions for the current year
        year_emi = emi_df[emi_df['year'] == year].copy()  

        # Filter total GHGRP emissions for the current year
        year_total_ghgrp_emis = ghgrp_df[ghgrp_df['Year'] == year]['emis_kt_tot'].sum()

        # Calculate the difference between national GHGI and GHGRP emissions
        year_national_emis_available = year_emi['ghgi_ch4_kt'].sum() - year_total_ghgrp_emis

        # Filter echo_df for the current year
        year_echo = echo_df[echo_df['Year'] == year].copy() 

        # Calculate the proportion of each state's GHGI methane emissions for the current year
        year_emi.loc[:, 'state_proportion'] = year_emi['ghgi_ch4_kt'] / year_emi['ghgi_ch4_kt'].sum()

        # Allocate emissions to each state based on their proportion of total emissions
        year_emi.loc[:, 'state_ghgi_emis'] = year_emi['state_proportion'] * year_national_emis_available

        # Process emissions for each state
        for state in unique_states:

            year_state_echo = year_echo[year_echo['State'] == state].copy()  

            # Keep GHGRP matches to concatenate later
            year_state_echo_ghgrp = year_state_echo[year_state_echo['ghgrp_match'] == 1].copy() 

            # Exclude GHGRP matches that already have emission values
            year_state_echo_no_ghgrp = year_state_echo[year_state_echo['ghgrp_match'] == 0].copy()  

            # Calculate the proportional flow at each facility
            year_state_echo_no_ghgrp.loc[:, 'flow_proportion'] = (
                year_state_echo_no_ghgrp['Wastewater Flow (MGal/yr)'] / 
                year_state_echo_no_ghgrp['Wastewater Flow (MGal/yr)'].sum()
            )
            
            # Calculate the emissions for each facility
            state_ghgi_emis_value = year_emi[year_emi['state_code'] == state]['state_ghgi_emis'].values[0]
            year_state_echo_no_ghgrp.loc[:, 'emis_kt'] = (
                year_state_echo_no_ghgrp['flow_proportion'] * state_ghgi_emis_value
            )

            # Drop the flow proportion column
            year_state_echo_no_ghgrp = year_state_echo_no_ghgrp.drop(columns=['flow_proportion'])

            # Concatenate the GHGRP matches and non-matches for our final dataframe
            final_proxy_df = pd.concat([final_proxy_df, year_state_echo_ghgrp, year_state_echo_no_ghgrp], ignore_index=True)

    return final_proxy_df



# def calculate_emissions_proxy(emi_df, ghgrp_df, echo_df, year_range):
#     """
#     Function to calculate emissions proxy for states based on GHGI and GHGRP data.

#     Parameters:
#     - emi_df: DataFrame containing GHGI emissions data.
#     - ghgrp_df: DataFrame containing GHGRP emissions data.
#     - echo_df: DataFrame containing ECHO facility data.
#     - year_range: List or range of years to process.

#     Returns:
#     - final_proxy_df: DataFrame containing the processed emissions data for each state.
#     """

#     # Create the final dataframe that we will populate below
#     final_proxy_df = pd.DataFrame()

#     # Get the unique states in the ECHO data
#     unique_states = emi_df['state_code'].unique()

#     for year in year_range:
        
#         # Filter total GHGI emissions for the current year
#         year_emi = emi_df[emi_df['year'] == year]

#         # Filter total GHGRP emissions for the current year
#         year_total_ghgrp_emis = ghgrp_df[ghgrp_df['Year'] == year]['emis_kt_tot'].sum()

#         # Calculate the difference between national GHGI and GHGRP emissions
#         year_national_emis_available = year_emi['ghgi_ch4_kt'].sum() - year_total_ghgrp_emis

#         # Filter echo_df for the current year
#         year_echo = echo_df[echo_df['Year'] == year]

#         # Calculate the proportion of each state's GHGI methane emissions for current year
#         year_emi.loc[:, 'state_proportion'] = year_emi['ghgi_ch4_kt'] / year_emi['ghgi_ch4_kt'].sum()

#         # Allocate emissions to each state based on their proportion of total emissions
#         year_emi.loc[:, 'state_ghgi_emis'] = year_emi['state_proportion'] * year_national_emis_available

#         # Process emissions for each state
#         for state in unique_states:

#             year_state_echo = year_echo[year_echo['State'] == state]

#             # Keep only GHGRP matches to concatenate later
#             year_state_echo_ghgrp = year_state_echo[year_state_echo['ghgrp_match'] == 1]

#             # Exclude GHGRP matches that already have emission values
#             year_state_echo_no_ghgrp = year_state_echo[year_state_echo['ghgrp_match'] == 0]

#             # Calculate the proportional flow at each facility
#             year_state_echo_no_ghgrp.loc[:, 'flow_proportion'] = (
#                 year_state_echo_no_ghgrp['Wastewater Flow (MGal/yr)'] / 
#                 year_state_echo_no_ghgrp['Wastewater Flow (MGal/yr)'].sum()
#             )
            
#             # Calculate the emissions for each facility
#             year_state_echo_no_ghgrp.loc[:, 'emis_kt'] = (
#                 year_state_echo_no_ghgrp['flow_proportion'] * 
#                 year_emi[year_emi['state_code'] == state]['state_ghgi_emis'].values[0]
#             )

#             # Drop the flow proportion column
#             year_state_echo_no_ghgrp = year_state_echo_no_ghgrp.drop(columns=['flow_proportion'])

#             # Concatenate the GHGRP matches and non-matches for our final dataframe
#             final_proxy_df = pd.concat([final_proxy_df, year_state_echo_ghgrp, year_state_echo_no_ghgrp], ignore_index=True)

#     return final_proxy_df

def process_facilities_and_emissions(echo_df, ghgrp_df, emi_df, year_range, industry='Pulp and Paper'):
    """
    Process facilities data, match with GHGRP data, and distribute emissions.
    
    Parameters:
    echo_df (pd.DataFrame): ECHO dataset with facilities
    ghgrp_df (pd.DataFrame): GHGRP dataset
    epa_df (pd.DataFrame): EPA emissions data
    year_range (list): List of years to process
    industry (str): Industry name for EPA emissions filtering
    
    Returns:
    pd.DataFrame: Processed ECHO dataset with matched and distributed emissions
    """

    # Step 1: Match facilities to GHGRP based on location
    echo_df['ghgrp_match'] = 0
    echo_df['emis_kt'] = 0
    ghgrp_df['found'] = 0
    
    # convert int to float to avoid potential math / dtype errors
    echo_df['emis_kt'] = echo_df['emis_kt'].astype(float)
    ghgrp_df['emis_kt_tot'] = ghgrp_df['emis_kt_tot'].astype(float)

    ghgrp_df = convert_state_names_to_codes(ghgrp_df, 'State')

    # Some GHGRP data are empty after trying to join the GHGRP emi and facility data
    if len(ghgrp_df) != 0:
        for idx, echo_row in echo_df.iterrows():
            for _, ghgrp_row in ghgrp_df.iterrows():
                dist = np.sqrt((ghgrp_row['latitude'] - echo_row['Facility Latitude'])**2 +
                            (ghgrp_row['longitude'] - echo_row['Facility Longitude'])**2)
                if dist < 0.025 and ghgrp_row['Year'] == echo_row['Year']:
                    ghgrp_df.loc[ghgrp_row.name, 'found'] = 1
                    echo_df.loc[idx, 'ghgrp_match'] = 1
                    echo_df.loc[idx, 'emis_kt'] = ghgrp_row['emis_kt_tot']
                    break
        print(f"Results from: {industry}")
        print(f"Found (%): {100 * echo_df['ghgrp_match'].sum() / len(echo_df):.2f}")
        print(f"GHGRP not found: {(ghgrp_df['found'] == 0).sum()}")
        print(f"GHGRP found: {(ghgrp_df['found'] == 1).sum()}")
        print(f"Total Emis (kt): {echo_df['emis_kt'].sum():.2f}")

        # Step 2: Add GHGRP facilities not found in ECHO to the "master" ECHO dataframe

        ghgrp_to_add = ghgrp_df[ghgrp_df['found'] == 0][['Year', 'State', 'latitude', 'longitude', 'emis_kt_tot']]

        # Rename columns to match the desired output
        ghgrp_to_add = ghgrp_to_add.rename(columns={
            'latitude': 'Facility Latitude',
            'longitude': 'Facility Longitude',
            'emis_kt_tot': 'emis_kt'
        })

        # Add the fixed columns 'ghgrp_match' and 'Wastewater Flow (MGal/yr)'
        ghgrp_to_add['ghgrp_match'] = 2
        ghgrp_to_add['Wastewater Flow (MGal/yr)'] = 0

        # Add the "not found" GHGRP facilities to the ECHO dataframe
        echo_df = pd.concat([echo_df, ghgrp_to_add], ignore_index=True)
        
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

def filter_by_naics(df, naics_prefix):
    """Filter dataframe by NAICS code prefix."""
    return df[df['NAICS Code'].str.startswith(naics_prefix)].copy().reset_index(drop=True)

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
# %%
# Step 2.2 Read in full ECHO dataset

echo_file_directory = proxy_data_dir_path / "wastewater/ECHO"
combined_echo_file_path = echo_file_directory / "combined_echo_data.csv"

# Check if the combined CSV already exists, read it if it does, otherwise create it
if combined_echo_file_path.exists():
    echo_full = read_combined_file(combined_echo_file_path)
else:
    echo_full = read_and_combine_csv_files(echo_file_directory)


# %%

# Step 2.3 Process the potw and non-potw facilities

# use reported daily flow rate. If not reported, use total design flow rate (if available and adjusted based on
# nationally-avaialable ratios of reported flow to capacity). Do for both POTW and non-POTW facilities
# Also Group data based on Permit Number (retain the maximum flow reported for the permit)
# also filter out water supply SIC codes

# Process POTW facilities
echo_potw = clean_and_group_echo_data(echo_full, 'POTW')

# Process NON-POTW facilities
echo_nonpotw = clean_and_group_echo_data(echo_full, 'NON-POTW')


# %%

# Step 2.4 Create dataframes for each non-potw industry

echo_nonpotw['NAICS Code'] = echo_nonpotw['NAICS Code'].astype(str)
echo_nonpotw['SIC Code'] = echo_nonpotw['SIC Code'].astype(str)

industries = {
    'pp': ('3221', ['2611', '2621', '2631']),
    'mp': ('3116', ['0751', '2011', '2048', '2013', '5147', '2077', '2015']),
    'fv': ('3114', ['2037', '2038', '2033', '2035', '2032', '2034', '2099']),
    'ethanol': ('325193', ['2869']),
    'brew': ('312120', ['2082']),
    'petrref': ('32411', ['2911'])
}

# Process all industries
for industry_name, (naics_prefix, sic_codes) in industries.items():
    result_df = extract_industry_facilities(echo_nonpotw, industry_name, naics_prefix, sic_codes)
    exec(f"echo_{industry_name.lower()} = result_df")

# %%

# Check if there are missing states between the ECHO and GHGI emi data

# for emi_df in [ww_pp_emi, ww_mp_emi, ww_fv_emi, ww_ethanol_emi, ww_brew_emi, ww_petrref_emi]:
#     for echo_df in [echo_pp, echo_mp, echo_fv, echo_ethanol, echo_brew, echo_petrref]:
#         # for names in ["echo_pp", "echo_mp", "echo_fv", "echo_ethanol", "echo_brew", "echo_petrref"]:
#         print("Comparing", names)
#         print(compare_state_sets(emi_df, echo_df, 'state_code', 'State'))
#         print("\n")

print("Comparing Pulp and Paper")
print(compare_state_sets(ww_pp_emi, echo_pp, 'state_code', 'State'))

print("Comparing Meat and Poultry")
print(compare_state_sets(ww_mp_emi, echo_mp, 'state_code', 'State'))

print("Comparing Fruits and Vegetables")
print(compare_state_sets(ww_fv_emi, echo_fv, 'state_code', 'State'))

print("Comparing Ethanol")
print(compare_state_sets(ww_ethanol_emi, echo_ethanol, 'state_code', 'State'))

print("Comparing Breweries")
print(compare_state_sets(ww_brew_emi, echo_brew, 'state_code', 'State'))

print("Comparing Petroleum Refining")
print(compare_state_sets(ww_petrref_emi, echo_petrref, 'state_code', 'State'))

# %%

# Step 2.5 Read in GHGRP Subpart II Data

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

print('NAICS CODES in Subpart II:', np.unique(ghgrp_ind['NAICS Code']))

# Filter data for different industries
industry_filters = {
    'pp': '322',      # Pulp and paper
    'mp': '3116',     # Red meat and poultry
    'fv': '3114',     # Fruits and vegetables
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

# Step 2.6 Join GHGRP, ECHO, and emi data

industrial_emi_totals = pd.DataFrame(industrial_emis.groupby(['year'])['ghgi_ch4_kt'].sum().reset_index())

final_pp = process_facilities_and_emissions(echo_pp, ghgrp_pp, ww_pp_emi, years, industry='Pulp and Paper')
final_mp = process_facilities_and_emissions(echo_mp, ghgrp_mp, ww_mp_emi, years, industry='Meat and Poultry')
final_fv = process_facilities_and_emissions(echo_fv, ghgrp_fv, ww_fv_emi, years, industry='Fruit and Vegetables')
final_eth = process_facilities_and_emissions(echo_eth, ghgrp_eth, ww_ethanol_emi, years, industry='Ethanol')
final_brew = process_facilities_and_emissions(echo_brew, ghgrp_brew, ww_brew_emi, years, industry='Breweries')
final_ref = process_facilities_and_emissions(echo_ref, ghgrp_ref, ww_petrref_emi, years, industry='Petroleum Refining')

# %%
# summary of the counts of datasets 

def summarize_ghgrp_match(df, column_name, industry):
    """
    Summarize the counts and percentages of each unique value in the specified column.

    Parameters:
    df (pd.DataFrame): The DataFrame containing the 'ghgrp_match' column.
    column_name (str): The name of the column to summarize.

    Returns:
    pd.DataFrame: DataFrame showing the count and percentage of each unique value.
    """

    df[column_name] = df[column_name].replace({
        0: "ECHO",
        1: "GHGRP-ECHO Match",
        2: "GHGRP"
    })
    
    # Count the occurrences of each unique value
    counts = df[column_name].value_counts()
    
    # Calculate the percentage of each value
    percentages = (counts / len(df)) * 100
    
    print(f"Value counts for {industry}:")
    print(df[column_name].value_counts())
    print(f"Percentage of each value: {percentages}")
    print("\n")

summarize_ghgrp_match(final_pp, 'ghgrp_match', 'Pulp and Paper')
summarize_ghgrp_match(final_mp, 'ghgrp_match', 'Meat and Poultry')
summarize_ghgrp_match(final_fv, 'ghgrp_match', 'Fruit and Vegetables')
summarize_ghgrp_match(final_eth, 'ghgrp_match', 'Ethanol')
summarize_ghgrp_match(final_brew, 'ghgrp_match', 'Breweries')
summarize_ghgrp_match(final_ref, 'ghgrp_match', 'Petroleum Refining')









# %%

# Step 2.7 Create final dataframe that is ready for mapping
def create_final_proxy_df(final_proxy_df):   
    
    # process to create a final proxy dataframe

    final_proxy_df['est_ch4'] = final_proxy_df['emis_kt'] / sum(final_proxy_df['emis_kt'])
    
    # create geometry column
    final_proxy_df['geometry'] = final_proxy_df.apply(lambda row: Point(row['Facility Longitude'], row['Facility Latitude']) if pd.notnull(row['Facility Longitude']) and pd.notnull(row['Facility Latitude']) else None, axis=1)
    
    # subset to only include the columns we want to keep
    final_proxy_df = final_proxy_df[['State', 'Year', 'Facility Latitude', 'Facility Longitude', 'emis_kt', 'geometry']]

    final_proxy_df


# %%
# Step 2.7 Population data





############################################
# WILL NEED TO GET NEWEST POP DATA HERE
############################################
