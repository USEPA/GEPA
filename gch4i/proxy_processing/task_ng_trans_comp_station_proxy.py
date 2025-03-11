"""
Name:                   task_ng_trans_comp_station_proxy.py
Date Last Modified:     2025-01-30
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Mapping of natural gas transmission compressor station proxy emissions
Input Files:            State Geo: global_data_dir_path / "tl_2020_us_state.zip"
                        Enverus Prism/DI: sector_data_dir_path / "enverus/production/intermediate_outputs"
                        NEI: sector_data_dir_path / "nei_og"
Output Files:           proxy_data_dir_path / "ng_trans_comp_station_proxy.parquet"
"""

# %% Import Libriaries
from pathlib import Path
import os
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
    years,
    min_year,
    max_year,
)

from gch4i.utils import(
    us_state_to_abbrev,
)


# %% Pytask Functions


@mark.persist
@task(id="ng_trans_comp_station_proxy")
def task_get_ng_trans_comp_station_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    enverus_midstream_ng_path: Path = sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb",
    ghgrp_facilities_path: Path = sector_data_dir_path / "ng_processing/GHGRP_Facility_Info_Jan2025.csv",
    ghgrp_subpart_w_path: Path = sector_data_dir_path / "ng_processing/EF_W_EMISSION_SOURCE_GHG_Jan2025.xlsb",
    ghgrp_missing_counties_path: Path = sector_data_dir_path / "ng_processing/ghgrp_w_missing_counties.xlsx",
    proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_trans_comp_station_proxy.parquet",
):

    # Function to calculate average when denominator is non-zero
    def safe_div(x,y):
        if y == 0:
            return 0
        return x / y

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

    # Read in Enverus Midstream Transmission Compressor Station Data (Onshore Only)
    enverus_stations = (gpd.read_file(
        enverus_midstream_ng_path,
        layer="CompressorStations",
        columns=['NAME', 'OPERATOR', 'TYPE', 'STATUS', 'FUEL_MCFD', 'HP', 'STATE_NAME', 'CNTY_NAME', 'CNTRY_NAME', 'geometry'])
        .query("STATUS == 'Operational'")
        .query("CNTRY_NAME == 'United States'")
        .query("STATE_NAME.isin(@state_gdf['state_name'])")
        .query("TYPE == 'Transmission'")
        .drop(columns=["TYPE", "STATUS", "CNTRY_NAME"])
        .rename(columns={"NAME": "facility_name",
                         "OPERATOR": "operator",
                         "FUEL_MCFD": "fuel",
                         "HP": "hp",
                         "STATE_NAME": "state_name",
                         "CNTY_NAME": "county",
                         })
        .assign(state_code='NaN')
        .to_crs(4326)
        .reset_index(drop=True)
        )

    # Add state_code to data
    for istation in np.arange(0, len(enverus_stations)):
        enverus_stations.loc[istation, "state_code"] = (
            us_state_to_abbrev(enverus_stations.loc[istation, "state_name"])
        )

    # Extract latitude and longitude
    enverus_stations['latitude'] = enverus_stations.loc[:, 'geometry'].y
    enverus_stations['longitude'] = enverus_stations.loc[:, 'geometry'].x

    # Correct the fuel useage data (locations where FUEL_MCFD = 0)
    #   - For locations with non-zero HP, estimate FUEL_MCFD by multiplying HF by the 
    #     average fuel useage to HP ratio.
    #   - For locations with zero HP, assign FUEL_MCFD as the median FUEL_MCFD.
    fuel_hp_ratio = np.mean(enverus_stations['fuel'][(enverus_stations['fuel'] > 0) & (enverus_stations['hp'] > 0)]/enverus_stations['hp'][(enverus_stations['fuel'] > 0) & (enverus_stations['hp'] > 0)])
    median_mcfd = np.median(enverus_stations.loc[enverus_stations['fuel'] > 0, 'fuel'])
    # estimate FUEL_MCFD by multiplying HF by the average fuel useage to HP ratio
    for istation in np.arange(0, len(enverus_stations)):
        if enverus_stations['fuel'][istation] == 0:
            enverus_stations.loc[istation, 'fuel'] = fuel_hp_ratio * enverus_stations['hp'][istation]
    # assign remaining missing FUEL_MCFD as the median FUEL_MCFD.
    for istation in np.arange(0, len(enverus_stations)):
        if enverus_stations['fuel'][istation] == 0:
            enverus_stations.loc[istation, 'fuel'] = median_mcfd
    enverus_stations = enverus_stations.loc[:, ['facility_name', 'state_code', 'county', 'geometry', 'latitude', 'longitude', 'fuel']]

    # Read in the GHGRP Subpart W data
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
                                # Filter for onshore natural gas transmission compression
                                .query("industry_segment == 'Onshore natural gas transmission compression [98.230(a)(4)]'")
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

    # Format GHGRP facility data
    ghgrp_facility_emissions = (ghgrp_facility_emissions
                                .rename(columns={"facility_name_x": "facility_name"})
                                .drop(columns="facility_name_y"))
    
    # Determine facilities missing counties
    # This code is commented out after the counties have been identified
    # missing_counties = ghgrp_facility_emissions[ghgrp_facility_emissions['county'].isna()]
    # missing_counties.to_csv('missing_counties.csv')

    # Read in missing counties
    missing_counties = pd.read_excel(ghgrp_missing_counties_path, sheet_name='ng_trans_comp_station_proxy')

    # Add missing county information
    for istation in np.arange(0, len(missing_counties)):
        ifacility_id = str(missing_counties.loc[istation, 'facility_id'])
        icounty = missing_counties.loc[istation, 'county']
        ghgrp_facility_emissions.loc[ghgrp_facility_emissions['facility_id'] == ifacility_id, 'county'] = icounty

    # For each year of GHGRP data, match GHGRP plants to Enverus data
    # note there is only one available year of Enverus data
    GHGRP_temp_data = ghgrp_facility_emissions.copy()
    GHGRP_temp_data['match_flag'] = 0
    GHGRP_temp_data['Env_name'] = ''
    GHGRP_temp_data['Env_county'] = ''
    GHGRP_temp_data['Env_state'] = ''
    GHGRP_temp_data['Env_fuel'] = 0
    enverus_notmatched = enverus_stations.copy()
    rows_to_delete = []

    # First, find exact matching lat/lon facilities
    for iplant in np.arange(0, len(GHGRP_temp_data)):
        # Round lat and lon to have a better chance to finding matches
        lat_temp = round(GHGRP_temp_data['latitude'][iplant], 2)
        lon_temp = round(GHGRP_temp_data['longitude'][iplant], 2)

        # Index of plant locations that match lat or lon
        match_lat = np.where(round(enverus_stations['latitude'], 2) == lat_temp)
        match_lon = np.where(round(enverus_stations['longitude'], 2) == lon_temp)

        # A match exists by comparing lat and lon
        if np.size(match_lat) > 0 and np.size(match_lon) > 0:
            # There is one matching plant by comparing lat
            if np.size(match_lat) == 1:
                # Check to make sure this plant also matches for lon, if it matches,
                # assign the Enverus data to the GHGRP plant
                # Check whether the same index is found in the lat/lon lists
                if match_lat[0][0] in match_lon[0]:
                    GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                    GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_stations.loc[match_lat[0][0], 'facility_name']
                    GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_stations.loc[match_lat[0][0], 'county']
                    GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_stations.loc[match_lat[0][0], 'state_code']
                    GHGRP_temp_data.loc[iplant, 'Env_fuel'] = enverus_stations.loc[match_lat[0][0], 'fuel']
                    rows_to_delete = np.append(rows_to_delete, match_lat[0][0])
            # There are more than one matching plant by comparing lat
            # Choose the plant with > 0 fuel mcfd
            else:
                # Loop through the matching lat values
                for idx in np.arange(0, np.size(match_lat)):
                    # Check whether the same index is found in the lat/lon lists
                    if match_lat[0][idx] in match_lon[0] and \
                        enverus_stations.loc[match_lat[0][idx], 'fuel'] > 0:
                        GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                        GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_stations.loc[match_lat[0][idx], 'facility_name']
                        GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_stations.loc[match_lat[0][idx], 'county']
                        GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_stations.loc[match_lat[0][idx], 'state_code']
                        GHGRP_temp_data.loc[iplant, 'Env_fuel'] = enverus_stations.loc[match_lat[0][idx], 'fuel']
                        rows_to_delete = np.append(rows_to_delete, match_lat[0][idx])

    # Drop rows
    rows_to_delete = rows_to_delete.astype(int)
    enverus_notmatched = enverus_notmatched.drop(rows_to_delete).reset_index(drop=True)

    # Second, find all Enverus plants within each county that did not match and try to match based on proximity to GHGRP plants
    rows_to_delete = []
    for iplant in np.arange(0, len(GHGRP_temp_data)):

        if GHGRP_temp_data.loc[iplant, 'match_flag'] != 1:
            if pd.isna(GHGRP_temp_data['county'][iplant]):
                continue  # this should no longer trigger since couty data are corrected above
            else:
                if 'county' in GHGRP_temp_data['county'][iplant].lower():
                    match = np.where(enverus_notmatched['county'].str.contains(GHGRP_temp_data['county'][iplant][:-7], case=False))
                elif 'parish' in GHGRP_temp_data['county'][iplant].lower():
                    match = np.where(enverus_notmatched['county'].str.contains(GHGRP_temp_data['county'][iplant][:-7], case=False))
                elif 'borough' in GHGRP_temp_data['county'][iplant].lower():
                    match = np.where(enverus_notmatched['county'].str.contains(GHGRP_temp_data['county'][iplant][:-8], case=False))
                else:
                    match = np.where(enverus_notmatched['county'].str.contains(GHGRP_temp_data['county'][iplant], case=False))

                if np.size(match) > 0:
                    # match is the list of enverus plants within the given county (not already matched)
                    # 1. if both lat and lon is closest to a single plant, assign that plant regardless of fuel
                    # 2. if lat and lon are closest to two different plants, then filter the fuel for non-zeros
                    # 2a. if all fuels are zeros, assign based on whichever lat/lons are closest
                    # 2b. if some fuels are non-zero, then assign based on whichever of those are closest to GHGRP 

                    lat_temp = round(GHGRP_temp_data['latitude'][iplant], 2)
                    lon_temp = round(GHGRP_temp_data['longitude'][iplant], 2)

                    list_envmat = enverus_notmatched.iloc[match]

                    # Minimum difference between the GHGRP lat/lon and matching Enverus plant(s) lat/lon and the Enverus plant index
                    vallat, idxlat = min((val, idx) for (idx, val) in enumerate(abs(list_envmat.loc[:, 'latitude']-lat_temp)))
                    vallon, idxlon = min((val, idx) for (idx, val) in enumerate(abs(list_envmat.loc[:, 'longitude']-lon_temp)))
                    # If both lat and lon is closest to a single plant, assign that plant regardless of fuel
                    if idxlat == idxlon:
                        GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                        GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_notmatched.loc[list_envmat.index[idxlat], 'facility_name']
                        GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_notmatched.loc[list_envmat.index[idxlat],'county']
                        GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_notmatched.loc[list_envmat.index[idxlat], 'state_code']
                        GHGRP_temp_data.loc[iplant, 'Env_fuel'] = enverus_notmatched.loc[list_envmat.index[idxlat], 'fuel']
                        rows_to_delete = np.append(rows_to_delete, list_envmat.index[idxlat])
                    else:
                        # If lat and lon are closest to two different plants, then filter the fuel for non-zeros
                        list_envmat_filter = list_envmat[list_envmat['fuel'] > 0]
                        # If all fuels are zeros, assign based on whichever lat/lons are closest
                        if np.size(list_envmat_filter) == 0:
                            # Assign to the plant with the closest lat
                            if vallat < vallon:
                                GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                                GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_notmatched.loc[list_envmat.index[idxlat], 'facility_name']
                                GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_notmatched.loc[list_envmat.index[idxlat], 'county']
                                GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_notmatched.loc[list_envmat.index[idxlat], 'state_code']
                                GHGRP_temp_data.loc[iplant, 'Env_fuel'] = enverus_notmatched.loc[list_envmat.index[idxlat], 'fuel']
                                rows_to_delete = np.append(rows_to_delete, list_envmat.index[idxlat])
                            # Assign to the plant with the closest lon
                            elif vallat > vallon:
                                GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                                GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_notmatched.loc[list_envmat.index[idxlon], 'facility_name']
                                GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_notmatched.loc[list_envmat.index[idxlon], 'county']
                                GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_notmatched.loc[list_envmat.index[idxlon], 'state_code']
                                GHGRP_temp_data.loc[iplant, 'Env_fuel'] = enverus_notmatched.loc[list_envmat.index[idxlon], 'fuel']
                                rows_to_delete = np.append(rows_to_delete, list_envmat.index[idxlon])
                        else:
                            # If some fuels are non-zero, then assign based on whichever of those are closest to GHGRP 
                            vallat, idxlat = min((val, idx) for (idx, val) in enumerate(abs(list_envmat_filter.loc[:, 'latitude']-lat_temp)))
                            vallon, idxlon = min((val, idx) for (idx, val) in enumerate(abs(list_envmat_filter.loc[:, 'longitude']-lon_temp)))
                            # Assign to the plant with the closest lat
                            if vallat < vallon:
                                GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                                GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_notmatched.loc[list_envmat_filter.index[idxlat], 'facility_name']
                                GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_notmatched.loc[list_envmat_filter.index[idxlat], 'county']
                                GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_notmatched.loc[list_envmat_filter.index[idxlat], 'state_code']
                                GHGRP_temp_data.loc[iplant, 'Env_fuel'] = enverus_notmatched.loc[list_envmat_filter.index[idxlat], 'fuel']
                                rows_to_delete = np.append(rows_to_delete, list_envmat_filter.index[idxlat])
                            # Assign to the plant with the closest lon
                            elif vallat > vallon:
                                GHGRP_temp_data.loc[iplant, 'match_flag'] = 1
                                GHGRP_temp_data.loc[iplant, 'Env_name'] = enverus_notmatched.loc[list_envmat_filter.index[idxlon], 'facility_name']
                                GHGRP_temp_data.loc[iplant, 'Env_county'] = enverus_notmatched.loc[list_envmat_filter.index[idxlon], 'county']
                                GHGRP_temp_data.loc[iplant, 'Env_state'] = enverus_notmatched.loc[list_envmat_filter.index[idxlon], 'state_code']
                                GHGRP_temp_data.loc[iplant, 'Env_fuel'] = enverus_notmatched.loc[list_envmat_filter.index[idxlon], 'fuel']
                                rows_to_delete = np.append(rows_to_delete, list_envmat_filter.index[idxlon])

    # Ensure rows to delete are integers
    rows_to_delete = rows_to_delete.astype(int)
    # Copy temp data to GHGRP facility emissions
    ghgrp_facility_emissions = GHGRP_temp_data.copy()

    # Matched and not matched GHGRP and Enverus data sets
    # Unmatched Enverus plants
    enverus_notmatched = enverus_notmatched.drop(rows_to_delete)
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
    print(len(enverus_notmatched), ' of ', len(enverus_stations))

    # Calculate the average emissions/fuel ratio for matched plants
    matched_GHGRP_plants['emis_fuel_ratio'] = 0.0
    for iplant in np.arange(0, len(matched_GHGRP_plants)):
        matched_GHGRP_plants.loc[iplant, 'emis_fuel_ratio'] = (matched_GHGRP_plants.loc[iplant, 'ch4_emi'] / matched_GHGRP_plants.loc[iplant, 'Env_fuel'] if matched_GHGRP_plants.loc[iplant, 'Env_fuel'] > 0 else 0)

    # Calculate the average emissions/fuel ratio by year
    avg_emis_fuel_ratio = np.zeros(len(years))
    for iyear in np.arange(0, len(years)):
        avg_emis_fuel_ratio[iyear] = np.mean(matched_GHGRP_plants.query(f"year == {years[iyear]}")['emis_fuel_ratio'])

    # Assign emissions to the unmatched Enverus stations using the annual average emissions/fuel ratios
    enverus_notmatched_ch4_added = pd.DataFrame()
    for iyear in np.arange(0, len(years)):
        emi_fuel_ratio_iyear = avg_emis_fuel_ratio[iyear]
        unmatched_enverus_iyear = enverus_notmatched.assign(ch4_emi=lambda df: df['fuel'] * emi_fuel_ratio_iyear).assign(year=years[iyear])
        enverus_notmatched_ch4_added = pd.concat([enverus_notmatched_ch4_added, unmatched_enverus_iyear]).reset_index(drop=True)

    # Combine matched GHGRP stations with unmatched Enverus stations
    matched_GHGRP_plants = matched_GHGRP_plants.loc[:, ["facility_name", "state_code", "year", "ch4_emi", "latitude", "longitude"]]
    enverus_notmatched_ch4_added = enverus_notmatched_ch4_added.loc[:, ["facility_name", "state_code", "year", "ch4_emi", "latitude", "longitude"]]
    all_stations = pd.concat([matched_GHGRP_plants, enverus_notmatched_ch4_added]).reset_index(drop=True)

    # Calculate facility relative emissions for each year combination
    # NOTE: Emissions are at a national level, so the relative emissions are normalized on a national level with no state_code column
    all_stations['rel_emi'] = all_stations.groupby(["year"])['ch4_emi'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    all_stations = all_stations.drop(columns='ch4_emi')

    # Convert proxy to geodataframe
    all_stations_gdf = (
        gpd.GeoDataFrame(
            all_stations,
            geometry=gpd.points_from_xy(
                all_stations["longitude"],
                all_stations["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["state_code", "latitude", "longitude"])
        .loc[:, ["year", "facility_name", "geometry", "rel_emi"]]
        .astype({"rel_emi":float})
    )

    # Check that relative emissions sum to 1.0 each state/year combination
    sums = all_stations_gdf.groupby(["year"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"  # assert that the sums are close to 1

    # Output proxy parquet files
    all_stations_gdf.to_parquet(proxy_output_path)

    return None
