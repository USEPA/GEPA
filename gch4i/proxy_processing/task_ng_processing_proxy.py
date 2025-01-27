"""
Name:                   task_ng_processing_proxy.py
Date Last Modified:     2025-01-24
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Process natural gas processing proxy data for methane emissions
Input Files:            - State Geodata: {global_data_dir_path}/tl_2020_us_state.zip
                        - Enverus : {sector_data_dir_path}/enverus/midstream/
                            Rextag_Natural_Gas.gdb
                        - GHGRP Facility: {sector_data_dir_path}/ng_processing/
                            GHGRP_Facility_Info_Jan2025.csv
                        - GHGRP Subpart W: {sector_data_dir_path}/ng_processing/
                            EF_W_EMISSION_SOURCE_GHG_Jan2025.xlsb
                        - NG Processing: {emi_data_dir_path}/processing_emi.csv
Output Files:           - {proxy_data_dir_path}/ng_processing_proxy.parquet
"""

# %% Import Libraries
from pathlib import Path
from typing import Annotated
import pandas as pd
import geopandas as gpd
import numpy as np
from pytask import Product, task, mark
from pyxlsb import open_workbook

from gch4i.config import (
    proxy_data_dir_path,
    global_data_dir_path,
    sector_data_dir_path,
    emi_data_dir_path,
    max_year,
    min_year,
    years,
)

from gch4i.utils import us_state_to_abbrev


# %% Pytask Function
@mark.persist
@task(id="ng_processing_proxy")
def task_get_ng_processing_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    enverus_midstream_ng_path: Path = sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb",
    ghgrp_facilities_path: Path = sector_data_dir_path / "ng_processing/GHGRP_Facility_Info_Jan2025.csv",
    ghgrp_subpart_w_path: Path = sector_data_dir_path / "ng_processing/EF_W_EMISSION_SOURCE_GHG_Jan2025.xlsb",
    ng_processing_emi_path: Path = emi_data_dir_path / "processing_emi.csv",
    proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_processing_proxy.parquet",
):
    """
    Process natural gas processing proxy data for methane emissions

    Args:
        state_path (Path): Path to the state geodata
        enverus_midstream_ng_path (Path): Path to the Enverus Midstream Natural Gas data
        ghgrp_facilities_path (Path): Path to the GHGRP Facility data
        ghgrp_subpart_w_path (Path): Path to the GHGRP Subpart W data
        ng_processing_emi_path (Path): Path to the natural gas processing emissions data
        proxy_output_path (Path): Path to the output proxy data

    Returns:
        None. Outputs the processed proxy data to a parquet file.
    """
    # Load in State ANSI data
    state_gdf = (
        gpd.read_file(state_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .reset_index(drop=True)
        .to_crs(4326)
    )

    # Enverus Midstream Processing Plant Locations
    gas_processing_plants = (gpd.read_file(
        enverus_midstream_ng_path,
        layer="GasProcessingPlants",
        columns=["NAME", "TYPE", "STATUS", "CAPACITY", "CNTY_NAME", "STATE_NAME", "CNTRY_NAME", "geometry"])
        # Filter type, status, country, and state columns
        .query("TYPE == 'Processing Plant'")
        .query("STATUS == 'Operational'")
        .query("CNTRY_NAME == 'United States'")
        .query("STATE_NAME.isin(@state_gdf['state_name'])")
        .drop(columns=["STATUS", "CNTRY_NAME"])
        .rename(columns={"NAME": "facility_name",
                         "TYPE": "type",
                         "CAPACITY": "capacity",
                         "CNTY_NAME": "county",
                         "STATE_NAME": "state_name",
                         })
        .assign(state_code='NaN')
        # Convert CRS to 4326
        .to_crs(4326)
        .reset_index(drop=True)
        )

    central_processing_facilities = (gpd.read_file(
        enverus_midstream_ng_path,
        layer="CentralProcessingFacilities",
        columns=["NAME", "TYPE", "STATUS", "CAPACITY", "CNTY_NAME", "STATE_NAME", "CNTRY_NAME", "geometry"])
        # Filter type, status, country, and state columns
        .query("TYPE == 'Central Processing Facility'")
        .query("STATUS == 'Operational'")
        .query("CNTRY_NAME == 'United States'")
        .query("STATE_NAME.isin(@state_gdf['state_name'])")
        .drop(columns=["STATUS", "CNTRY_NAME"])
        .rename(columns={"NAME": "facility_name",
                         "TYPE": "type",
                         "CAPACITY": "capacity",
                         "CNTY_NAME": "county",
                         "STATE_NAME": "state_name",
                         })
        .assign(state_code='NaN')
        # Convert CRS to 4326
        .to_crs(4326)
        .reset_index(drop=True)
        )

    # Combine gas processing plants and central processing facilities
    enverus_processing_plants = pd.concat([gas_processing_plants, central_processing_facilities]).reset_index(drop=True)

    # Assign state codes to Enverus processing plants
    for istation in np.arange(0, len(enverus_processing_plants)):
        enverus_processing_plants.loc[istation, "state_code"] = us_state_to_abbrev(enverus_processing_plants.loc[istation, "state_name"])

    # Extract latitude and longitude
    enverus_processing_plants['latitude'] = enverus_processing_plants.loc[:, 'geometry'].y
    enverus_processing_plants['longitude'] = enverus_processing_plants.loc[:, 'geometry'].x

    # GHGRP Subpart W Processing Plant Emissions
    # Read in the GHGRP facility locations
    ghgrp_facility_info = (pd.read_csv(ghgrp_facilities_path)
                           .loc[:, ["facility_name", "facility_id", "latitude", "longitude", "state", "county", "city", "zip"]]
                           .rename(columns={"state": "state_code",
                                            "zip": "zip_code"})
                           .astype({"facility_id": str})
                           .drop_duplicates(subset=['facility_id'], keep='last')
                           )

    # Read in the GHGRP facility emissions
    def read_xlsb(file_path, sheet_name):
        with open_workbook(file_path) as wb:
            with wb.get_sheet(sheet_name) as sheet:
                data = []
                for row in sheet.rows():
                    data.append([item.v for item in row])
                return pd.DataFrame(data[1:], columns=data[0])

    # Read in data from GHGRP Subpart W
    ghgrp_facility_emissions = (read_xlsb(ghgrp_subpart_w_path, 'GGDSPUBV2.EF_W_EMISSIONS_SOURC')
                                .rename(columns=str.lower)
                                .rename(columns={"total_reported_ch4_emissions": "ch4_emi", "reporting_year": "year"})
                                .astype({"facility_id": int, "year": int})
                                .astype({"facility_id": str})
                                # Filter for onshore natural gas processing
                                .query("industry_segment == 'Onshore natural gas processing [98.230(a)(3)]'")
                                # Grab years of interest
                                .query("year.between(@min_year, @max_year)")
                                .loc[:, ["facility_name", "facility_id", "year", "ch4_emi"]]
                                .reset_index(drop=True)
                                )

    # Filter for emissions greater than 0
    ghgrp_facility_emissions['ch4_emi'] = pd.to_numeric(ghgrp_facility_emissions['ch4_emi'], errors='coerce')
    ghgrp_facility_emissions = ghgrp_facility_emissions.astype({"ch4_emi": float}).query("ch4_emi > 0").reset_index(drop=True)

    # Merge GHGRP facility locations with emissions
    ghgrp_facility_emissions = ghgrp_facility_emissions.merge(ghgrp_facility_info, how='left', on='facility_id')

    # Add missing county information
    for iplant in np.arange(0, len(ghgrp_facility_emissions)):
        if (pd.isna(ghgrp_facility_emissions['county'][iplant])):
            #DEBUG# print(iplant, ghgrp_facility_emissions.loc[iplant,'city'], ghgrp_facility_emissions.loc[iplant,'state_code'], ghgrp_facility_emissions.loc[iplant,'zip_code'])
            if ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 78162:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Bee'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 77463:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Brazoria'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 71115:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Caddo'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 44427:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Columbiana'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] in [71052, 71063]:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'De Soto'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 79762:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Ector'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 88220:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Eddy'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 75860:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Freestone'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 79739:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Glasscock'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] in [73004, 73089]:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Grady'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 77002:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Harris'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 74570:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Hughes'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 76035:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Johnson'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 93251:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Kern'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 78014:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'La Salle'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 88252:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Lea'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 79782:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Martin'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 58847:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'McKenzie'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 79706:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Midland'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 79058:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Moore'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 58784:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Mountrail'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 75946:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Nacogdoches'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 92821:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Orange'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] in [79730, 79743]:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Pecos'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 71019:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Red River'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] in [79718, 79770, 79772, 79785]:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Reeves'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 81648:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Rio Blanco'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 75935:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Shelby'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 58622:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Stark'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] in [80611, 80621, 80651]:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Weld'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 79014:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Wheeler'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 58755:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Williams'
            elif ghgrp_facility_emissions.loc[iplant, 'zip_code'] == 79754:
                ghgrp_facility_emissions.loc[iplant, 'county'] = 'Young'

    # Format GHGRP facility data
    ghgrp_facility_emissions = (ghgrp_facility_emissions
                                .rename(columns={"facility_name_x": "facility_name"})
                                .drop(columns="facility_name_y"))

    # For each year of GHGRP data, match GHGRP plants to Enverus data
    # note there is only one available year of Enverus data
    GHGRP_temp_data = ghgrp_facility_emissions.copy()
    GHGRP_temp_data['match_flag'] = 0
    GHGRP_temp_data['Env_name'] = ''
    GHGRP_temp_data['Env_county'] = ''
    GHGRP_temp_data['Env_state'] = ''
    GHGRP_temp_data['Env_capacity'] = 0
    enverus_processing_plants_notmatched = enverus_processing_plants.copy()
    rows_to_delete = []

    # First, find exact matching lat/lon facilities
    for iplant in np.arange(0, len(GHGRP_temp_data)):
        # Round lat and lon to have a better chance to finding matches
        lat_temp = round(GHGRP_temp_data['latitude'][iplant], 2)
        lon_temp = round(GHGRP_temp_data['longitude'][iplant], 2)

        # Index of plant locations that match lat or lon
        match_lat = np.where(round(enverus_processing_plants['latitude'], 2) == lat_temp)
        match_lon = np.where(round(enverus_processing_plants['longitude'], 2) == lon_temp)

        # A match exists by comparing lat and lon
        if np.size(match_lat) > 0 and np.size(match_lon) > 0:
            # There is one matching plant by comparing lat
            if np.size(match_lat) == 1:
                # Check to make sure this plant also matches for lon, if it matches,
                # assign the Enverus data to the GHGRP plant
                # Check whether the same index is found in the lat/lon lists
                if match_lat[0][0] in match_lon[0]:
                    GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                    GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_processing_plants.loc[match_lat[0][0], 'facility_name']
                    GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_processing_plants.loc[match_lat[0][0], 'county']
                    GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_processing_plants.loc[match_lat[0][0], 'state_code']
                    GHGRP_temp_data.loc[iplant, 'Env_capacity'] = enverus_processing_plants.loc[match_lat[0][0], 'capacity']
                    rows_to_delete = np.append(rows_to_delete, match_lat[0][0])
            # There are more than one matching plant by comparing lat
            # Choose the plant with > 0 capacity
            else:
                # Loop through the matching lat values
                for idx in np.arange(0, np.size(match_lat)):
                    # Check whether the same index is found in the lat/lon lists
                    if match_lat[0][idx] in match_lon[0] and \
                        enverus_processing_plants.loc[match_lat[0][idx], 'capacity'] > 0:
                        GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                        GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_processing_plants.loc[match_lat[0][idx], 'facility_name']
                        GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_processing_plants.loc[match_lat[0][idx], 'county']
                        GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_processing_plants.loc[match_lat[0][idx], 'state_code']
                        GHGRP_temp_data.loc[iplant, 'Env_capacity'] = enverus_processing_plants.loc[match_lat[0][idx], 'capacity']
                        rows_to_delete = np.append(rows_to_delete, match_lat[0][idx])

    # Drop rows
    rows_to_delete = rows_to_delete.astype(int)
    enverus_processing_plants_notmatched = enverus_processing_plants_notmatched.drop(rows_to_delete).reset_index(drop=True)

    # Second, find all Enverus plants within each county that did not match and try to match based on proximity to GHGRP plants
    rows_to_delete = []
    for iplant in np.arange(0, len(GHGRP_temp_data)):

        if GHGRP_temp_data.loc[iplant, 'match_flag'] != 1:
            if pd.isna(GHGRP_temp_data['county'][iplant]):
                continue  # this should no longer trigger since couty data are corrected above
            else:
                if 'county' in GHGRP_temp_data['county'][iplant].lower():
                    match = np.where(enverus_processing_plants_notmatched['county'].str.contains(GHGRP_temp_data['county'][iplant][:-7], case=False))
                elif 'parish' in GHGRP_temp_data['county'][iplant].lower():
                    match = np.where(enverus_processing_plants_notmatched['county'].str.contains(GHGRP_temp_data['county'][iplant][:-7], case=False))
                elif 'borough' in GHGRP_temp_data['county'][iplant].lower():
                    match = np.where(enverus_processing_plants_notmatched['county'].str.contains(GHGRP_temp_data['county'][iplant][:-8], case=False))
                else:
                    match = np.where(enverus_processing_plants_notmatched['county'].str.contains(GHGRP_temp_data['county'][iplant], case=False))

                if np.size(match) > 0:
                    # match is the list of enverus plants within the given county (not already matched)
                    # 1. if both lat and lon is closest to a single plant, assign that plant regardless of capacity
                    # 2. if lat and lon are closest to two different plants, then filter the capacity for non-zeros
                    # 2a. if all capacities are zeros, assign based on whichever lat/lons are closest
                    # 2b. if some capacities are non-zero, then assign based on whichever of those are closest to GHGRP 

                    lat_temp = round(GHGRP_temp_data['latitude'][iplant], 2)
                    lon_temp = round(GHGRP_temp_data['longitude'][iplant], 2)

                    list_envmat = enverus_processing_plants_notmatched.iloc[match]

                    # Minimum difference between the GHGRP lat/lon and matching Enverus plant(s) lat/lon and the Enverus plant index
                    vallat, idxlat = min((val, idx) for (idx, val) in enumerate(abs(list_envmat.loc[:, 'latitude']-lat_temp)))
                    vallon, idxlon = min((val, idx) for (idx, val) in enumerate(abs(list_envmat.loc[:, 'longitude']-lon_temp)))
                    # If both lat and lon is closest to a single plant, assign that plant regardless of capacity
                    if idxlat == idxlon:
                        GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                        GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlat], 'facility_name']
                        GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlat],'county']
                        GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlat], 'state_code']
                        GHGRP_temp_data.loc[iplant, 'Env_capacity'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlat], 'capacity']
                        rows_to_delete = np.append(rows_to_delete, list_envmat.index[idxlat])
                    else:
                        # If lat and lon are closest to two different plants, then filter the capacity for non-zeros
                        list_envmat_filter = list_envmat[list_envmat['capacity'] > 0]
                        # If all capacities are zeros, assign based on whichever lat/lons are closest
                        if np.size(list_envmat_filter) == 0:
                            # Assign to the plant with the closest lat
                            if vallat < vallon:
                                GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                                GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlat], 'facility_name']
                                GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlat], 'county']
                                GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlat], 'state_code']
                                GHGRP_temp_data.loc[iplant, 'Env_capacity'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlat], 'capacity']
                                rows_to_delete = np.append(rows_to_delete, list_envmat.index[idxlat])
                            # Assign to the plant with the closest lon
                            elif vallat > vallon:
                                GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                                GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlon], 'facility_name']
                                GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlon], 'county']
                                GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlon], 'state_code']
                                GHGRP_temp_data.loc[iplant, 'Env_capacity'] = enverus_processing_plants_notmatched.loc[list_envmat.index[idxlon], 'capacity']
                                rows_to_delete = np.append(rows_to_delete, list_envmat.index[idxlon])
                        else:
                            # If some capacities are non-zero, then assign based on whichever of those are closest to GHGRP 
                            vallat, idxlat = min((val, idx) for (idx, val) in enumerate(abs(list_envmat_filter.loc[:, 'latitude']-lat_temp)))
                            vallon, idxlon = min((val, idx) for (idx, val) in enumerate(abs(list_envmat_filter.loc[:, 'longitude']-lon_temp)))
                            # Assign to the plant with the closest lat
                            if vallat < vallon:
                                GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                                GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_processing_plants_notmatched.loc[list_envmat_filter.index[idxlat], 'facility_name']
                                GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_processing_plants_notmatched.loc[list_envmat_filter.index[idxlat], 'county']
                                GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_processing_plants_notmatched.loc[list_envmat_filter.index[idxlat], 'state_code']
                                GHGRP_temp_data.loc[iplant, 'Env_capacity'] = enverus_processing_plants_notmatched.loc[list_envmat_filter.index[idxlat], 'capacity']
                                rows_to_delete = np.append(rows_to_delete, list_envmat_filter.index[idxlat])
                            # Assign to the plant with the closest lon
                            elif vallat > vallon:
                                GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                                GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_processing_plants_notmatched.loc[list_envmat_filter.index[idxlon], 'facility_name']
                                GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_processing_plants_notmatched.loc[list_envmat_filter.index[idxlon], 'county']
                                GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_processing_plants_notmatched.loc[list_envmat_filter.index[idxlon], 'state_code']
                                GHGRP_temp_data.loc[iplant, 'Env_capacity'] = enverus_processing_plants_notmatched.loc[list_envmat_filter.index[idxlon], 'capacity']
                                rows_to_delete = np.append(rows_to_delete, list_envmat_filter.index[idxlon])

    # Ensure rows to delete are integers
    rows_to_delete = rows_to_delete.astype(int)
    # Copy temp data to GHGRP facility emissions
    ghgrp_facility_emissions = GHGRP_temp_data.copy()

    # Matched and not matched GHGRP and Enverus data sets
    # Unmatched Enverus plants
    enverus_processing_plants_notmatched = enverus_processing_plants_notmatched.drop(rows_to_delete)
    # All GHGRP plants
    ghgrp_facility_emissions = GHGRP_temp_data.copy()
    # Unique GHGRP plants
    unique_GHGRP_plants = ghgrp_facility_emissions['facility_id'].unique()
    # Unmatched GHGRP plants
    unmatched_GHGRP_plants = ghgrp_facility_emissions[ghgrp_facility_emissions['match_flag'] == 0]
    # Unique unmatched GHGRP plants
    unique_unmatched_GHGRP_plants = unmatched_GHGRP_plants['facility_id'].unique()
    # Matched GHGRP plants
    matched_GHGRP_plants = GHGRP_temp_data[GHGRP_temp_data['match_flag'] == 1].reset_index(drop=False)

    # QA/QC number of plants matched
    print('QA/QC: Number of GHGRP plants not in Enverus data set')
    print(len(unique_unmatched_GHGRP_plants), ' of ', len(unique_GHGRP_plants))
    print('QA/QC: Number of Enverus plants not in GHGRP data set')
    print(len(enverus_processing_plants_notmatched), ' of ', len(enverus_processing_plants))

    # Calculate the average emissions/capacity ratio for matched plants
    # Note that in v2, throughput was used instead of capacity. In the future, check to
    # see if the Enverus Midstream data download includes throughput and use if available.
    matched_GHGRP_plants['emis_cap_ratio'] = 0.0
    for iplant in np.arange(0, len(matched_GHGRP_plants)):
        matched_GHGRP_plants.loc[iplant, 'emis_cap_ratio'] = (matched_GHGRP_plants.loc[iplant, 'ch4_emi'] / matched_GHGRP_plants.loc[iplant, 'Env_capacity'] if matched_GHGRP_plants.loc[iplant, 'Env_capacity'] > 0 else 0)

    # Calculate the average emissions/capacity ratio by year
    avg_emis_capacity_ratio = np.zeros(len(years))
    for iyear in np.arange(0, len(years)):
        avg_emis_capacity_ratio[iyear] = np.mean(matched_GHGRP_plants.query(f"year == {years[iyear]}")['emis_cap_ratio'])
    print('QA/QC: Average Emissions to Capacity Ratio')
    print(avg_emis_capacity_ratio)

    # Assign emissions to the unmatched Enverus plants using the annual average emissions/capacity ratios
    enverus_processing_plants_notmatched_ch4_added = pd.DataFrame()
    for iyear in np.arange(0, len(years)):
        emi_cap_ratio_iyear = avg_emis_capacity_ratio[iyear]
        unmatched_enverus_iyear = enverus_processing_plants_notmatched.assign(ch4_emi=lambda df: df['capacity'] * emi_cap_ratio_iyear).assign(year=years[iyear])
        enverus_processing_plants_notmatched_ch4_added = pd.concat([enverus_processing_plants_notmatched_ch4_added, unmatched_enverus_iyear]).reset_index(drop=True)

    # Combine matched GHGRP plants with unmatched Enverus plants
    matched_GHGRP_plants = matched_GHGRP_plants.loc[:, ["facility_name", "state_code", "year", "ch4_emi", "latitude", "longitude"]]
    enverus_processing_plants_notmatched_ch4_added = enverus_processing_plants_notmatched_ch4_added.loc[:, ["facility_name", "state_code", "year", "ch4_emi", "latitude", "longitude"]]
    processing_plants = pd.concat([matched_GHGRP_plants, enverus_processing_plants_notmatched_ch4_added]).reset_index(drop=True)

    # Calculate facility relative emissions for each state-year combination
    processing_plants['rel_emi'] = processing_plants.groupby(["state_code", "year"])['ch4_emi'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    processing_plants = processing_plants.drop(columns='ch4_emi')

    # Check that relative emissions sum to 1.0 each state/year combination
    sums = processing_plants.groupby(["state_code", "year"])["rel_emi"].sum()  # get sums to check normalization
    # assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"  # assert that the sums are close to 1

    # Convert proxy to geodataframe
    processing_plants_gdf = (
        gpd.GeoDataFrame(
            processing_plants,
            geometry=gpd.points_from_xy(
                processing_plants["longitude"],
                processing_plants["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["facility_name", "latitude", "longitude"])
        .loc[:, ["year", "state_code", "rel_emi", "geometry"]]
    )

    # Check for missing proxy data and create alternative proxy data
    """
    Steps:
        - Check if proxy data is missing for a state and year
        - Create alternative proxy data by assigning rel_emi = 1 and geometry = state polygon
            - This distributes emissions evenly across the state
    """
    # Check if proxy data exists for emissions data
    emi_df = (pd.read_csv(ng_processing_emi_path)
              .query("state_code.isin(@state_gdf['state_code'])")
              .query("ghgi_ch4_kt > 0")
              .reset_index(drop=True)
              )

    # Retrieve unique state codes for emissions without proxy data
    # This step is necessary, as not all emissions data excludes emission-less states
    emi_states = set(emi_df[['state_code', 'year']].itertuples(index=False, name=None))
    proxy_states = set(processing_plants_gdf[['state_code', 'year']].itertuples(index=False, name=None))

    # Find missing states
    missing_states = emi_states.difference(proxy_states)

    # Add missing states alternative data to grouped_proxy
    if missing_states:
        # Create alternative proxy from missing states
        alt_proxy = (
            pd.DataFrame(missing_states, columns=['state_code', 'year'])
            # Assign well type and make rel_emi = 1
            .assign(
                rel_emi=1
            )
            # Merge state polygon geometry
            .merge(
                state_gdf[['state_code', 'geometry']],
                on='state_code',
                how='left'
            )
        )

        # Convert to GeoDataFrame
        alt_proxy = gpd.GeoDataFrame(alt_proxy, geometry='geometry', crs='EPSG:4326')

        # Append to grouped_proxy
        processing_plants_gdf = pd.concat([processing_plants_gdf, alt_proxy], ignore_index=True)

    # Output proxy parquet files
    processing_plants_gdf.to_parquet(proxy_output_path)

    return None

# %%
