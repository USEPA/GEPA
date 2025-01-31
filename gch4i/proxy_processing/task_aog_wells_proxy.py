"""
Name:                   task_aog_wells_proxy.py
Date Last Modified:     2025-01-14
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of abandoned wells oil/gas proxy emissions
Input Files:            - Lat/Lon: NEI_Reference_Grid_LCC_to_WGS84_latlon.shp
                        - Enverus Path: DIDSK_HEADERS_API10_2019_abandoned_wells.csv
                        - NEI Input (prefix): CONUS_SA_FILES_
Output Files:           - aog_gas_wells_proxy.parquet
                        - aog_oil_wells_proxy.parquet
Notes:                  - Indiana and Illinois data from Enverus Path is inaccurate and
                        was corrected in V2 with NEI data, with 2018 coming from a
                        separate MS Access source. V3 has updated NEI data for 2018-2022
                        and was used to correct V3.
                        - Slight differences in outputs occur from V2 to V3 due to the
                        switch to geopandas and the use of centroid points for NEI data
                        (2018-2022). Enverus and NEI data was joined to state_code
                        based on geometry, whereas V2 used lat/lon gridding.
                        - Three states have emi data that do not have proxy data:
                            - FL, 2012-2022, GAS
                            - ID, 2012-2022, OIL
                            - MD, 2012-2022, OIL
                        Proxy data was created for these states by assigning rel_emi = 1
                        and distributing emissions evenly across the state. NOTE: Next
                        version should consider evaluting NEI data as source of proxy
                        data for these states.
                        - NOTE: get_aog_wells_proxy_data function is base function for
                        cleaning and assigning proxy data. It is used in concert with
                        two pytask functions to retrieve relevant state/year/well_type
                        data and calculate relative emissions.

"""

########################################################################################
# %% Load Packages

from pathlib import Path
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import numpy as np
import geopandas as gpd

from gch4i.config import (
    V3_DATA_PATH,
    global_data_dir_path,
    emi_data_dir_path,
    proxy_data_dir_path,
    max_year,
    min_year,
)

########################################################################################
# %% Load Path Files & Repeated Variables

year_range = [*range(min_year, max_year + 1)]
num_years = len(year_range)
year_range_str = [str(i) for i in year_range]

# State Path
state_path: Path = global_data_dir_path / "tl_2020_us_state.zip"

# NEI Lat/Lon Path
# loc_path = '/Users/aburnette/Library/CloudStorage/
# OneDrive-SharedLibraries-EnvironmentalProtectionAgency(EPA)/
# Gridded CH4 Inventory - RTI 2024 Task Order/Task 2/ghgi_v3_working/GEPA_Source_Code/
# Global_InputData/Gridded/NEI_Reference_Grid_LCC_to_WGS84_latlon.shp'
loc_path = (
    V3_DATA_PATH / "sector" / "nei_og" / "NEI_Reference_Grid_LCC_to_WGS84_latlon.shp"
)
# Read in shape file for Lat/Lon
shape = gpd.read_file(loc_path)

# Enverus Path: Base Proxy Data
Enverus_path = (
    V3_DATA_PATH.parent
    / "GEPA_Source_Code"
    / "Global_InputData"
    / "Enverus"
    / "AOG"
    / "DIDSK_HEADERS_API10_2019_abandoned_wells.csv"
)

# Dir prefix for ERG NEI files
ERG_NEI_input = V3_DATA_PATH / "sector" / "nei_og" / "CONUS_SA_FILES_"

# Oil and Gas ERG NEI files (2012-2017)
ERG_NEI_gas = "/USA_698_NOFILL.txt"
ERG_NEI_oil = "/USA_695_NOFILL.txt"


########################################################################################
# %% Proxy Function


def get_aog_wells_proxy_data(
    filter_condition,
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    loc_path: Path = V3_DATA_PATH
    / "sector"
    / "nei_og"
    / "NEI_Reference_Grid_LCC_to_WGS84_latlon.shp",
    Enverus_path: Path = V3_DATA_PATH.parent
    / "GEPA_Source_Code"
    / "Global_InputData"
    / "Enverus"
    / "AOG"
    / "DIDSK_HEADERS_API10_2019_abandoned_wells.csv",
    ERG_NEI_input: Path = V3_DATA_PATH / "sector" / "nei_og" / "CONUS_SA_FILES_",
    ERG_NEI_gas: str = "/USA_698_NOFILL.txt",
    ERG_NEI_oil: str = "/USA_695_NOFILL.txt",
):
    """
    This base proxy function is used in concert with 2 pytask functions to retrieve
    relevent state/year/well_type data and calculate relative emissions.

    Step 1: Read in State and Location data
    Step 2: Calculate the state-level fraction of abandoned gas to oil wells
    Step 3: Make CONUS Grid Array
    Step 4: Correct IL & IN Data.
    STEP 5. Combine data into proxy_gdf and remove empty proxies
    Step 6: Calculate grouped_proxy
    STEP 7. Check for missing proxy data AND Create alternative proxy data
    """

    # STEP 1. Read in State and Location data
    ####################################################################################
    # Read in list of included states
    state_list = (
        gpd.read_file(state_path)
        .loc[:, ["STATEFP", "STUSPS"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .sort_values("statefp")
        .drop(columns="statefp")
        .reset_index(drop=True)
    )

    # Read in State gdf
    state_gdf = (
        gpd.read_file(state_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
    )

    # Read in Enverus Abandoned Well Data (from ERG)
    location_data = (
        pd.read_csv(Enverus_path, low_memory=False)
        # Drop columns
        .drop(
            columns=[
                "API10",
                "COUNTYPARISH",
                "CUM_GAS",
                "CUM_OIL",
                "WELL_STATUS",
                "PRODUCTION_TYPE",
                "ERG_WELL_TYPE",
                "GOR",
            ]
        )
        # Rename columns
        .rename(
            {
                "SURFACE_HOLE_LATITUDE_WGS84": "LAT",
                "SURFACE_HOLE_LONGITUDE_WGS84": "LON",
            },
            axis=1,
        )
        # Filter data
        .query('OFFSHORE == "N"').query("STATE in @state_list.state_code")
        # Make columns lowercase
        .rename(columns=lambda x: str(x).lower())
        # Make columns datetime
        .assign(
            prod_year=lambda x: pd.to_datetime(x["last_prod_date"]).dt.year,
            comp_year=lambda x: pd.to_datetime(x["completion_date"]).dt.year,
            spud_year=lambda x: pd.to_datetime(x["spud_date"]).dt.year,
        )
    )

    # STEP 2. Calculate the state-level fraction of abandoned gas to oil wells
    ####################################################################################
    """
    Calculate the state-level fraction of abandoned gas to oil wells
    - This will be applied to the 'DRY' well counts in the next step
    - Also used to allocate the dry well population to either oil or gas types

    gas wells = gas wells + dry wells * (gas wells / (gas wells + oil wells))
    """

    # Create a list to store results
    results = []

    # Iterate through each year
    for year in year_range:
        # Filter wells abandoned by the given year
        abandoned_wells = location_data[
            (location_data["prod_year"] < year)
            | (
                (location_data["prod_year"].isna())
                & (location_data["comp_year"] < year)
            )
            | (
                (location_data["prod_year"].isna())
                & (location_data["comp_year"].isna())
                & (location_data["spud_year"] < year)
            )
        ]

        # Group by state and well type, sum producing entity counts
        well_counts = (
            abandoned_wells.groupby(["state", "abandoned_well_type"])[
                "producing_entity_count"
            ]
            .sum()
            .unstack(fill_value=0)
        )

        # Ensure both 'GAS' and 'OIL' columns exist
        if "GAS" not in well_counts.columns:
            well_counts["GAS"] = 0
        if "OIL" not in well_counts.columns:
            well_counts["OIL"] = 0

        # Calculate the gas to oil ratio
        well_counts["gas_to_oil_ratio"] = well_counts["GAS"] / (
            well_counts["GAS"] + well_counts["OIL"]
        ).replace(0, np.nan)

        # Reset index to make state a column and add year
        year_results = well_counts.reset_index()
        year_results["year"] = year

        # Filter to only include states in state_list
        year_results = year_results[
            year_results["state"].isin(state_list["state_code"])
        ]

        results.append(year_results[["state", "year", "gas_to_oil_ratio"]])

    # Combine results into a single DataFrame
    abandoned_well_ratios = pd.concat(results, ignore_index=True)
    abandoned_well_ratios = abandoned_well_ratios.fillna(0)

    # STEP 3. Make CONUS Grid Array
    ####################################################################################
    """
    For each year, calculate the number of abandoned wells for gas and oil wells

    Follow same filtering procedure as previous step (prod_year, comp_year, spud_year)

    - Add the relevant number of wells to the map if the location is within the CONUS
    - V3:
        - Transform Lat/Lon to Point geometry
        - Use a GeoDataFrame to filter by state
        - Filter by Gas or Oil or Dry
    """

    # location_data_corr: Convert and Filter location_data to only include lower 48 + DC
    location_data_corr = gpd.GeoDataFrame(
        location_data,
        geometry=gpd.points_from_xy(location_data.lon, location_data.lat),
        crs=4326,
    )

    # Filter data and select columns
    location_data_corr = (
        gpd.sjoin(
            location_data_corr,
            state_gdf[["state_code", "geometry"]],
            how="inner",
            predicate="within",
        )
        .query("state_code.notna()")
        .loc[
            :,
            [
                "state_code",
                "producing_entity_count",
                "abandoned_well_type",
                "prod_year",
                "comp_year",
                "spud_year",
                "geometry",
            ],
        ]
        .rename(columns={"state_code": "state"})
    )

    # Create a list to store results
    results = []

    # Iterate through each year
    for year in year_range:
        # Filter wells abandoned by the given year
        temp = location_data_corr[
            (location_data_corr["prod_year"] < year)
            | (
                (location_data_corr["prod_year"].isna())
                & (location_data_corr["comp_year"] < year)
            )
            | (
                (location_data_corr["prod_year"].isna())
                & (location_data_corr["comp_year"].isna())
                & (location_data_corr["spud_year"] < year)
            )
        ]
        # Add year to temp
        temp["year"] = year
        # Allocate Dry wells to either gas or oil
        temp = temp.merge(abandoned_well_ratios, on=["state", "year"], how="left")
        # Create Gas portion of Dry wells
        temp_gas = temp.query('abandoned_well_type == "DRY"').assign(
            abandoned_well_type="GAS",
            producing_entity_count=lambda x: x["producing_entity_count"]
            * x["gas_to_oil_ratio"],
        )
        # Create Oil portion of Dry wells
        temp_oil = temp.query('abandoned_well_type == "DRY"').assign(
            abandoned_well_type="OIL",
            producing_entity_count=lambda x: x["producing_entity_count"]
            * (1 - x["gas_to_oil_ratio"]),
        )
        # Join Gas and Oil portions with original data
        temp = temp.query('abandoned_well_type != "DRY"')
        temp = (
            pd.concat([temp, temp_gas, temp_oil], ignore_index=True)
            .rename(columns={"state": "state_code"})
            .loc[
                :,
                [
                    "state_code",
                    "year",
                    "producing_entity_count",
                    "abandoned_well_type",
                    "geometry",
                ],
            ]
        )
        # Append to results
        results.append(temp)
    # Combine results into a single DataFrame
    base_results = pd.concat(results, ignore_index=True)

    # Remove IL and IN data. It is inaccurate and will be fixed in the next step
    base_results = (
        base_results.query('state_code != "IL" & state_code != "IN"')
        # Query for filter condition in abandoned_well_type
        .query("abandoned_well_type == @filter_condition")
    )

    # STEP 4. Correct IL/IN Data
    ####################################################################################
    # Determine file extension
    if filter_condition == "GAS":
        file_extension = ERG_NEI_gas
    elif filter_condition == "OIL":
        file_extension = ERG_NEI_oil
    else:
        print("Invalid filter condition")

    # Initialize results
    IL_IN_adj = []

    # Iterate through each year
    for iyear in np.arange(0, num_years):
        report_year = year_range_str[iyear]

        # Determine year data based on report_year
        # If report_year is 2012, use 2011 data
        if report_year == "2012":
            year = "2011"
        # If report_year is 2013-2015, use 2014 data
        elif report_year == "2013" or report_year == "2014" or report_year == "2015":
            year = "2014"
        # If report_year is 2016-2017, use 2017 data
        elif report_year == "2016" or report_year == "2017":
            year = "2017"
        # If report_year is 2018-2022, use report_year data
        elif report_year in year_range_str:
            year = report_year
        # If report_year is not in range, print error
        else:
            print("ERROR: NEI DATA MISSING FOR YEAR ", report_year)
        # Transform year to integer
        year = int(year)
        # Create path to NEI data
        if year >= 2011 and year <= 2017:
            path = f"{ERG_NEI_input}{year}{file_extension}"
        elif year == 2018:
            path = f"{ERG_NEI_input}{year}/{filter_condition}_WELLS.shp"
        elif year == 2019:
            path = f"{ERG_NEI_input}{year}/{filter_condition}_WELLS.shp"
        elif year == 2020:
            path = f"{ERG_NEI_input}{year}/{filter_condition}_WELL.shp"
        elif year == 2021:
            if filter_condition == "GAS":
                path = f"{ERG_NEI_input}{year}/_698.shp"
            else:
                path = f"{ERG_NEI_input}{year}/_695.shp"
        elif year == 2022:
            if filter_condition == "GAS":
                path = f"{ERG_NEI_input}{year}/GasWells.shp"
            else:
                path = f"{ERG_NEI_input}{year}/OilWells.shp"
        # If year is not in range, print error
        else:
            print("Error: NEI DATA MISSING FOR YEAR ", report_year)

        # Clean table based on year
        # data_temp: NEI Data
        if year >= 2011 and year <= 2017:
            data_temp = pd.read_csv(path, sep="\t", skiprows=25)
            data_temp = data_temp.drop(["!"], axis=1)
            data_temp.columns = [
                "Code",
                "FIPS",
                "COL",
                "ROW",
                "Frac",
                "Abs",
                "FIPS_Total",
                "FIPS_Running_Sum",
            ]

            # Create shape dataframe
            # temp_df: Shape Data
            temp_df = (
                # Extract ROW and COL from cellid
                shape.assign(
                    ROW=lambda x: x["cellid"].str.split("!").str[1].astype(int),
                    COL=lambda x: x["cellid"].str.split("!").str[0].astype(int),
                ).reset_index(drop=True)
            )

            # Merge NEI data with shape data
            temp_result = (
                data_temp.merge(temp_df, on=["COL", "ROW"], how="left")
                .filter(["Abs", "Latitude", "Longitude"])
                .rename(columns={"Abs": "producing_entity_count"})
            )

            # Convert to GeoDataFrame
            temp_result = gpd.GeoDataFrame(
                temp_result,
                geometry=gpd.points_from_xy(
                    temp_result["Longitude"], temp_result["Latitude"]
                ),
                crs=4326,
            ).drop(columns=["Longitude", "Latitude"])

            # Merge with state data
            temp_result = (
                gpd.sjoin(
                    temp_result,
                    state_gdf[["state_code", "geometry"]],
                    how="left",
                    predicate="within",
                ).drop(columns="index_right")
                # Filter for only IL and IN data
                .query('state_code == "IL" | state_code == "IN"')
                # Assign year and abandoned_well_type
                .assign(year=int(report_year), abandoned_well_type=filter_condition)
            )
            # Append to IL_IN_adj
            IL_IN_adj.append(temp_result)

        # If year is 2018-2022, data is in shapefile format
        elif year >= 2018 and year <= 2022:
            # Read in shapefile
            temp_result = gpd.read_file(path)
            # Convert crs to calculate centroid
            temp_result = temp_result.to_crs(3857)
            temp_result.loc[:, "geometry"] = temp_result.loc[:, "geometry"].centroid
            # Convert back to 4326
            temp_result = temp_result.to_crs(4326)
            # Merge with state data
            temp_result = (
                gpd.sjoin(
                    temp_result,
                    state_gdf[["state_code", "geometry"]],
                    how="left",
                    predicate="within",
                ).drop(columns="index_right")
                # Filter for only IL and IN data
                .query('state_code == "IL" | state_code == "IN"')
                # Assign year and abandoned_well_type
                .assign(year=year, abandoned_well_type=filter_condition)
            )
            # Filter and rename columns based on year
            if year <= 2020:
                temp_result = temp_result.filter(
                    [
                        "ACTIVITY",
                        "state_code",
                        "geometry",
                        "year",
                        "abandoned_well_type",
                    ]
                )
                temp_result = temp_result.rename(
                    columns={"ACTIVITY": "producing_entity_count"}
                )
            elif year == 2021:
                temp_result = temp_result.filter(
                    ["NUMBER_", "state_code", "geometry", "year", "abandoned_well_type"]
                )
                temp_result = temp_result.rename(
                    columns={"NUMBER_": "producing_entity_count"}
                )
            elif year == 2022:
                temp_result = temp_result.filter(
                    [
                        "NUMBER_WEL",
                        "state_code",
                        "geometry",
                        "year",
                        "abandoned_well_type",
                    ]
                )
                temp_result = temp_result.rename(
                    columns={"NUMBER_WEL": "producing_entity_count"}
                )
            # If year is not in range, print error
            else:
                print("MID: NEI DATA MISSING FOR YEAR ", report_year)
            # Append to IL_IN_adj
            IL_IN_adj.append(temp_result)
        # If year is not in range, print error
        else:
            print("END: NEI DATA MISSING FOR YEAR ", report_year)
    # Combine results into a single DataFrame
    IL_IN_adj = pd.concat(IL_IN_adj, ignore_index=True)

    ####################################################################################
    # STEP 5. Combine data into proxy_gdf and remove empty proxies

    # Combine base_results and IL_IN_adj
    proxy_gdf = pd.concat([base_results, IL_IN_adj], ignore_index=True)

    """
    Remove empty state/year/well_type combinations.
    This will prevent division by zero errors in the next step.
    This will enable checking for missing proxy data for emi data
    """

    # Remove empty proxies
    proxy_gdf = (
        proxy_gdf
        # Generate group sum for state_code, year, abandoned_well_type
        .assign(
            group_sum=lambda x: x.groupby(
                ["state_code", "year", "abandoned_well_type"]
            )["producing_entity_count"].transform("sum")
        )
        # Filter out empty groups
        .query("group_sum != 0")
        # Drop group_sum column
        .drop(columns="group_sum")
    )

    """
    Emi Data exists, but no proxy data for these:
    FL, 2012-2022, GAS
    ID, 2012-2022, OIL
    MD, 2012-2022, OIL
    """

    ####################################################################################
    # STEP 6. Calculate grouped_proxy
    """
    Calculate the relative emissions for each state/year/well_type.
    The rel_emi will be used to allocate emissions to the CONUS region

    rel_emi = geometry[producing_entity_count]  / state_year_sum[producing_entity_count]
    """
    grouped_gdf = (
        proxy_gdf
        # Sum emissions
        .groupby(
            ["state_code", "year", "abandoned_well_type", "geometry"], as_index=False
        )
        .agg({"producing_entity_count": "sum"})
        # Calculate state relative emissions (emissions / state_year emissions)
        .assign(
            state_year_sum=lambda x: x.groupby(
                ["state_code", "year", "abandoned_well_type"]
            )["producing_entity_count"].transform("sum"),
            rel_emi=lambda x: x["producing_entity_count"] / x["state_year_sum"],
        )
        # Drop columns
        .drop(columns=["state_year_sum", "producing_entity_count"])
        # Set geometry and CRS
        .set_geometry("geometry")
        .set_crs("EPSG:4326")
    )

    ####################################################################################
    # STEP 7. Check for missing proxy data AND Create alternative proxy data
    """
    Steps:
        - Check if Proxy data is missing for a state/year/well_type
        - Create Alternative Proxy Data: rel_emi = 1, geometry = state polygon
            - This distributes emissions evenly across the state
    """
    # Build dictionary to map well to well emissions data
    proxy_dict = {"GAS": "aog_gas_wells_emi", "OIL": "aog_oil_wells_emi"}

    # Filter grouped_gdf to ensure only relevant proxy data
    filtered_proxy = grouped_gdf.query("abandoned_well_type == @filter_condition")

    # Create filtered dictionary
    filtered_dict = {filter_condition: proxy_dict[filter_condition]}

    # Check if proxy data exists for emissions data
    for key, value in filtered_dict.items():
        emi_df = (
            pd.read_csv(f"{emi_data_dir_path}/{value}.csv")
            .query("ghgi_ch4_kt != 0")
            .query("state_code != 'AK'")
            .drop(columns=["Unnamed: 0"])
        )

    # Retrieve unique state codes for emissions without proxy data
    # This step is necessary, as not all emissions data excludes emission-less states
    emi_states = set(emi_df[["state_code", "year"]].itertuples(index=False, name=None))
    proxy_states = set(
        filtered_proxy[["state_code", "year"]].itertuples(index=False, name=None)
    )

    # Find missing states
    missing_states = emi_states.difference(proxy_states)

    # Add missing states alternative data to grouped_proxy
    if missing_states:
        # Create alternative proxy from missing states
        alt_proxy = (
            pd.DataFrame(missing_states, columns=["state_code", "year"])
            # Assign well type and make rel_emi = 1
            .assign(abandoned_well_type=filter_condition, rel_emi=1)
            # Merge state polygon geometry
            .merge(state_gdf[["state_code", "geometry"]], on="state_code", how="left")
        )
        # Convert to GeoDataFrame
        alt_proxy = gpd.GeoDataFrame(alt_proxy, geometry="geometry", crs="EPSG:4326")
        # Append to grouped_proxy
        grouped_gdf = pd.concat([grouped_gdf, alt_proxy], ignore_index=True)

    return grouped_gdf


########################################################################################
# %% Pytask Process


# Gas Proxy
@mark.persist
@task(id="aog_proxy")
def task_aog_gas_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    loc_path: Path = V3_DATA_PATH
    / "sector"
    / "nei_og"
    / "NEI_Reference_Grid_LCC_to_WGS84_latlon.shp",
    Enverus_path: Path = V3_DATA_PATH.parent
    / "GEPA_Source_Code"
    / "Global_InputData"
    / "Enverus"
    / "AOG"
    / "DIDSK_HEADERS_API10_2019_abandoned_wells.csv",
    # ERG_NEI_input: Path = V3_DATA_PATH / "sector" / "nei_og" / "CONUS_SA_FILES_",
    ERG_NEI_gas: str = "/USA_698_NOFILL.txt",
    ERG_NEI_oil: str = "/USA_695_NOFILL.txt",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "aog_gas_wells_proxy.parquet",
):
    proxy_gdf = get_aog_wells_proxy_data(
        filter_condition="GAS", ERG_NEI_gas="/USA_698_NOFILL.txt"
    )

    proxy_gdf.to_parquet(output_path)
    return None


# Oil Proxy
@mark.persist
@task(id="aog_proxy")
def task_aog_oil_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    loc_path: Path = V3_DATA_PATH
    / "sector"
    / "nei_og"
    / "NEI_Reference_Grid_LCC_to_WGS84_latlon.shp",
    Enverus_path: Path = V3_DATA_PATH.parent
    / "GEPA_Source_Code"
    / "Global_InputData"
    / "Enverus"
    / "AOG"
    / "DIDSK_HEADERS_API10_2019_abandoned_wells.csv",
    # ERG_NEI_input: Path = V3_DATA_PATH / "sector" / "nei_og" / "CONUS_SA_FILES_",
    ERG_NEI_gas: str = "/USA_698_NOFILL.txt",
    ERG_NEI_oil: str = "/USA_695_NOFILL.txt",
    output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "aog_oil_wells_proxy.parquet",
):
    proxy_gdf = get_aog_wells_proxy_data(
        filter_condition="OIL", ERG_NEI_oil="/USA_695_NOFILL.txt"
    )

    proxy_gdf.to_parquet(output_path)
    return None
