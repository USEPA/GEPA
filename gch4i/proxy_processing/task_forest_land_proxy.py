"""
Name:                   task_forest_land_proxy.py
Date Last Modified:     2024-03-11
Authors Name:           C. Coxen (RTI International)
Purpose:                Generate proxy data for forest land remaining forest land emissions
Input Files:            - {sector_data_dir_path}/forestlands_grasslands/MTBS_byEventFuelFuelbed_09Sep2024.csv
                        - {sector_data_dir_path}/forestlands_grasslands/fccs_fuelbed_Aug2023_jesModified.csv
                        - {sector_data_dir_path}/forestlands_grasslands/nawfd_fuelbed_Aug2023_jesModified.csv
                        - {sector_data_dir_path}/forestlands_grasslands/mtbs_perims_DD.shp
                        - {global_data_dir_path}/tl_2020_us_state/tl_2020_us_state.shp
                        - {emi_data_dir_path}/forest_land_emi.csv

Output Files:           - forest_land_proxy.parquet

Notes:                  - This script assigns proxy GHGI emissions for forest land remaining forest land using MTBS fire data.
                        - The proxy geometries are brought in from the MTBS fire permiter data. Some year-state combinations are missing
                            MTBS emissions data and are given a proportion of 1.0 for the entire state. These state-years are given the
                            geometry of the state from the tl_2020_us_state shapefile to allocate emissions across the entire state.
"""
# %%
from pathlib import Path
from typing import Annotated
from shapely.geometry import MultiPolygon, Polygon
from pyarrow import parquet
import pandas as pd
import geopandas as gpd
import numpy as np
from shapely.validation import make_valid
from pytask import Product, task, mark

from gch4i.config import (
    proxy_data_dir_path,
    emi_data_dir_path,
    sector_data_dir_path,
    global_data_dir_path,
    years
)

from gch4i.utils import convert_FIPS_to_two_letter_code

forest_land_proxy_path = sector_data_dir_path / "forestlands_grasslands/MTBS_byEventFuelFuelbed_09Sep2024.csv"
fccs_fuelbed_path = sector_data_dir_path / "forestlands_grasslands/fccs_fuelbed_Aug2023_jesModified.csv"
nawfd_fuelbed_path = sector_data_dir_path / "forestlands_grasslands/nawfd_fuelbed_Aug2023_jesModified.csv"
mtbs_burn_perimeter_path = sector_data_dir_path / "forestlands_grasslands/mtbs_perims_DD.shp"
state_path = global_data_dir_path / 'tl_2020_us_state/tl_2020_us_state.shp'
emi_path = emi_data_dir_path / "forest_land_emi.csv"

forest_land_output_path = proxy_data_dir_path / "forest_land_proxy.parquet"


# %% pytask function
@mark.persist
@task(id='forest_land_proxy')
def task_forest_land_proxy_data(
    forest_land_proxy_path: Path = forest_land_proxy_path,
    fccs_fuelbed_path: Path = fccs_fuelbed_path,
    nawfd_fuelbed_path: Path = nawfd_fuelbed_path,
    mtbs_burn_perimeter_path: Path = mtbs_burn_perimeter_path,
    state_path: Path = state_path,
    emi_path: pd.DataFrame = emi_path,
    forest_land_output_path: Annotated[Path, Product] = forest_land_output_path
    ) -> None:
    """
    This function processes the forest land proxy data and calculates the proxy emissions.
    
    Args:
    proxy_path: Path to the forest land proxy data.
    mtbs_lat_long_path: Path to the MTBS lat long data.
    fccs_fuelbed_path: Path to the FCCS fuelbed data.
    nawfd_fuelbed_path: Path to the NAWFD fuelbed data.
    state_path: Path to the state shapefile.
    output_path: Path to save the final proxy data.
    forest_land_emi: DataFrame containing the forest land emissions data.

    Returns:
    None. Proxy data is saved to a parquet file at the output_path.
    """

    # Read in the proxy data
    forest_land_proxy = pd.read_csv(forest_land_proxy_path)
    fccs_fuelbed = pd.read_csv(fccs_fuelbed_path)
    nawfd_fuelbed = pd.read_csv(nawfd_fuelbed_path)

    mtbs_burn_perimeters = (
    gpd.read_file(mtbs_burn_perimeter_path)
    .to_crs(4326)
    )

    state_gdf = (
    gpd.read_file(state_path)
    .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
    .rename(columns=str.lower)
    .rename(columns={"stusps": "state_code", "name": "state_name"})
    .astype({"statefp": int})
    # get only lower 48 + DC
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
    .to_crs(4326)
    .drop(columns=["state_name", "statefp"])
    )

    # Read in the emi data
    forest_land_emi = pd.read_csv(emi_path)

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

            # Filter proxy_df for the current year
            year_proxy = proxy_df[proxy_df['year'] == year].copy() 

            # Process emissions for each state
            for state in unique_states:

                year_state_proxy = year_proxy[year_proxy['state_code'] == state].copy()

                # Filter out MTBS ch4 emissions that are 0 so they can be replaced with a dummy row
                year_state_proxy = year_state_proxy[year_state_proxy['ch4_mg'] > 0]

                # Create a dummy row for state-years with no proxy data, proportion = 1 for these state-years
                if len(year_state_proxy) == 0:
                    # Create a dummy row for states with no data
                    dummy_row = pd.DataFrame({
                        'eventID': [f'dummy_{state}_{year}'],
                        'state_code': [state],
                        'year': [year],
                        'rel_emi': [1.0]
                    })
                    final_proxy_df = pd.concat([final_proxy_df, dummy_row], ignore_index=True)
                    continue

                # Group by eventID and sum the ch4_mg and burnBndAc columns
                year_state_proxy = year_state_proxy.groupby(['eventID'], as_index=False).agg({
                    'eventID': 'first',
                    'ch4_mg': 'sum',
                    'year': 'first',
                    'state_code': 'first'
                })
                
                # Calculate the proportion of the total MTBS proxy emissions for the year-state
                year_state_proxy.loc[:, 'rel_emi'] = (
                    year_state_proxy['ch4_mg'] / 
                    year_state_proxy['ch4_mg'].sum()
                )       

                # Concatenate to the final dataframe
                final_proxy_df = pd.concat([final_proxy_df, year_state_proxy], ignore_index=True)

        return final_proxy_df


    # %% Step 1 - Data wrangling

    # Edit the fuelbed_aggregate column to remove the 'evg' string and convert to an integer
    forest_land_proxy['fuelbed_aggregate'] = forest_land_proxy['fuelbed_aggregate'].str.replace('evg', '').astype(float).astype(int)

    # Convert the FIPS state codes to two-letter state codes
    forest_land_proxy = convert_FIPS_to_two_letter_code(forest_land_proxy, 'originstatecd')

    # Get the forest and nonforest habitat types from the FCCS data
    fccs_forest = get_habitat_types(fccs_fuelbed, 'FUELBED', 'forest')

    nawfd_forest = get_habitat_types(nawfd_fuelbed, 'name', 'forest')

    # Filter the MTBS data to only include forest habitat types in the FCCS and NAWFD data

    forest_land_proxy = forest_land_proxy[
        forest_land_proxy['fuelbed_aggregate'].isin(fccs_forest['FCCS']) |
        forest_land_proxy['fuelbed_aggregate'].isin(nawfd_forest['nawfd_id'])
    ]


    # %% Step 2 - Calculate proxy emissions for each state
    forestland_proxy_df = calculate_state_emissions(forest_land_emi, forest_land_proxy, years)

    # %% Join the MTBS lat long data to the proxy data
    mtbs_burn_perimeters = mtbs_burn_perimeters[['Event_ID', 'geometry']]
    mtbs_burn_perimeters.rename(columns={'Event_ID': 'eventID'}, inplace=True)
    forestland_proxy_df = forestland_proxy_df.merge(mtbs_burn_perimeters, on='eventID', how='left')

    # %% Join the state geometry data for states that are missing geometry. This is due to there being no MTBS eventID for that state-year combination to join MTBS geometry on.
    # State-years that have missing gemoetry data will receive the geometry data from the state shapefile and be given a proportion of 1.0.
    # All emis will be equally allocated across the entire state for those state-years

    # States that have geometry data
    states_with_geometry = forestland_proxy_df[forestland_proxy_df['geometry'].isnull() == False].copy()

    # States that are missing geometry data
    states_missing_geometry = forestland_proxy_df[forestland_proxy_df['geometry'].isnull() == True].copy()

    # Drop the geometry column
    states_missing_geometry = states_missing_geometry.drop(columns=['geometry'])

    # Subset the state data to only include the state code and geometry
    state_gdf = state_gdf[['state_code', 'geometry']]

    # Merge the state data with the states missing geometry
    states_missing_geometry = states_missing_geometry.merge(state_gdf, on='state_code', how='left')

    # Concatenate the two dataframes
    forestland_proxy_df = pd.concat([states_with_geometry, states_missing_geometry], ignore_index=True)
                                    
    # Only keep states in the lower 48
    forestland_proxy_df = forestland_proxy_df[forestland_proxy_df['state_code'].isin(state_gdf['state_code'])]

    # %% Step 3 Create the final proxy dataframe
    
    # Check that the relative emissions sum to 1 for each state
    sums = forestland_proxy_df.groupby(["state_code", 'year'])["rel_emi"].sum() #get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}" # assert that the sums are close to 1

    final_forestland_proxy_df = forestland_proxy_df[['state_code', 'year', 'rel_emi', 'geometry']]

    final_forestland_proxy_df = gpd.GeoDataFrame(final_forestland_proxy_df, geometry='geometry')

    # verify the geometries are valied
    # Fix invalid geometries
    final_forestland_proxy_df['geometry'] = final_forestland_proxy_df['geometry'].apply(lambda geom: make_valid(geom) if not geom.is_valid else geom)

    # Verify fix
    still_invalid = final_forestland_proxy_df[~final_forestland_proxy_df.geometry.is_valid]
    assert len(still_invalid) == 0, f"Invalid geometries still present in the proxy data: {still_invalid}"

    final_forestland_proxy_df.to_parquet(forest_land_output_path, index=False)

# %%
