"""
Name:                   task_ng_storage_comp_station_proxy.py
Date Last Modified:     2025-02-25
Authors Name:           Hannah Lohman (RTI International)
Purpose:                Mapping of natural gas storage compressor station proxy emissions
Input Files:            State Geo: global_data_dir_path / "tl_2020_us_state.zip"
                        Enverus Prism/DI: sector_data_dir_path / "enverus/production/intermediate_outputs"
                        NEI: sector_data_dir_path / "nei_og"
Output Files:           proxy_data_dir_path / "ng_storage_comp_station_proxy.parquet"
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
@task(id="ng_storage_comp_station_proxy")
def task_get_ng_storage_comp_station_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    enverus_midstream_ng_path: Path = sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb",
    eia_storage_fields_path: Path = sector_data_dir_path / "eia/191 Field Level Storage Data (Annual).csv",
    ghgrp_facilities_path: Path = sector_data_dir_path / "ng_processing/GHGRP_Facility_Info_Jan2025.csv",
    ghgrp_subpart_w_path: Path = sector_data_dir_path / "ng_processing/EF_W_EMISSION_SOURCE_GHG_Jan2025.xlsb",
    ghgrp_missing_counties_path: Path = sector_data_dir_path / "ng_processing/ghgrp_w_missing_counties.xlsx",
    emi_path: Path = emi_data_dir_path / "storage_comp_station_emi.csv",
    proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_storage_comp_station_proxy.parquet",
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

    # Read in EIA Storage Field Data
    EIA_StorFields = (pd.read_csv(eia_storage_fields_path)
                      .rename(columns={"Year": "Year",
                                       "Report<BR>State<BR>": "state_code",
                                       "Company<BR>Name": "Company Name",
                                       "Field<BR>Name": "Field Name",
                                       "Reservoir<BR>Name": "Reservoir Name",
                                       "County<BR>Name": "County Name",
                                       "Status": "Status",
                                       "Total Field<BR>Capacity(Mcf)": "Total Field Capacity(Mcf)"})
                      .query("Status == 'Active'")
                      .query("state_code.isin(@state_gdf['state_code'])")
                      .rename(columns={"state_code": "Report State"})
                      .loc[:, ["Year", "Report State", "Company Name", "Field Name",
                               "Reservoir Name", "County Name", "Total Field Capacity(Mcf)"]]
                      .reset_index(drop=True))

    # Read in Enverus Midstream Gas Storage Field Data
    Env_StorFields = (gpd.read_file(
        enverus_midstream_ng_path,
        layer="GasStorage",
        columns=['STATUS', 'RESERVOIR', 'NAME', 'OPERATOR', 'STATE_NAME', 'CNTY_NAME',
                 'CNTRY_NAME', 'FLDCAPMMCF', 'geometry'])
        .query("STATUS == 'Operational'")
        .query("CNTRY_NAME == 'United States'")
        .query("STATE_NAME.isin(@state_gdf['state_name'])")
        .drop(columns=["STATUS", "CNTRY_NAME"])
        .merge(state_gdf[['state_name', 'state_code']], on='state_name')
        .to_crs(4326)
        .dropna(subset=["geometry"])
        .assign(Latitude=lambda df: df['geometry'].y)
        .assign(Longitude=lambda df: df['geometry'].x)
        .reset_index(drop=True)
        )

    # Read in Enverus Midstream Storage Compressor Station Data (Onshore Only)
    enverus_stations = (gpd.read_file(
        enverus_midstream_ng_path,
        layer="CompressorStations",
        columns=['NAME', 'OPERATOR', 'TYPE', 'STATUS', 'FUEL_MCFD', 'HP', 'STATE_NAME',
                 'CNTY_NAME', 'CNTRY_NAME', 'geometry'])
        .query("STATUS == 'Operational'")
        .query("CNTRY_NAME == 'United States'")
        .query("STATE_NAME.isin(@state_gdf['state_name'])")
        .query("TYPE == 'Storage'")
        .drop(columns=["TYPE", "STATUS", "CNTRY_NAME"])
        .merge(state_gdf[['state_name', 'state_code']], on='state_name')
        .dropna(subset=["geometry"])
        .to_crs(4326)
        .assign(Latitude=lambda df: df['geometry'].y)
        .assign(Longitude=lambda df: df['geometry'].x)
        .reset_index(drop=True)
        )

    # Loop through each Enverus storage field to find where there is also a storage
    # compressor station.
    #   - First try matching on name, then operator to narrow down
    #   - If no match on name, then try matching based on location (within 0.01 degrees)
    #   - If no match on name, or location (within 0.01 degrees), try matching on 0.05 degrees and county name
    #   - If none of these criteria are met, there is no compressor station at that field. 

    Env_StorFields['Comp_flag'] = 0
    Env_StorFields['Comp_lat'] = 0
    Env_StorFields['Comp_lon'] = 0

    nomatch = 0
    for ifield in np.arange(0, len(Env_StorFields)):
        # Try to match facility names
        matched = np.where((Env_StorFields['NAME'][ifield] == enverus_stations['NAME']))[0]
        if np.size(matched) > 1:
            best_match = np.where(Env_StorFields['OPERATOR'][ifield] == enverus_stations.loc[matched, 'OPERATOR'])[0]
            if np.size(best_match) == 1:
                Env_StorFields.loc[ifield, 'Comp_flag'] = 1
                Env_StorFields.loc[ifield, 'Comp_lat'] = enverus_stations.loc[matched[best_match[0]], 'Latitude']
                Env_StorFields.loc[ifield, 'Comp_lon'] = enverus_stations.loc[matched[best_match[0]], 'Longitude']
            elif np.size(best_match) > 1:
                # This is occuring when there is a double count of the compressor stations (entries are identical except for Enverus ID)
                # In this case, assign one compressor station to the field
                Env_StorFields.loc[ifield, 'Comp_flag'] = 1  # could alternatively set this to the number of matches (if actually >1 station per field)
                Env_StorFields.loc[ifield, 'Comp_lat'] = enverus_stations.loc[matched[best_match[0]], 'Latitude']
                Env_StorFields.loc[ifield, 'Comp_lon'] = enverus_stations.loc[matched[best_match[0]], 'Longitude']
            else:
                # More than one case identified where name matches, but operator does not
                nomatch += 1
        elif np.size(matched) == 1:
            Env_StorFields.loc[ifield, 'Comp_flag'] = 1
            Env_StorFields.loc[ifield, 'Comp_lat'] = enverus_stations.loc[matched[0], 'Latitude']
            Env_StorFields.loc[ifield, 'Comp_lon'] = enverus_stations.loc[matched[0], 'Longitude']
            
        elif np.size(matched) < 1:
            # If they don't match based on name, then match based on location (likely due to slight spelling differences)
            best_match = np.where((np.abs(Env_StorFields['Latitude'][ifield]- enverus_stations['Latitude']) < 0.01) & 
                                  (np.abs(Env_StorFields['Longitude'][ifield]- enverus_stations['Longitude']) < 0.01))[0]
            if np.size(best_match) == 1:
                Env_StorFields.loc[ifield, 'Comp_flag'] = 1
                Env_StorFields.loc[ifield, 'Comp_lat'] = enverus_stations.loc[best_match[0], 'Latitude']
                Env_StorFields.loc[ifield, 'Comp_lon'] = enverus_stations.loc[best_match[0], 'Longitude']
            elif np.size(best_match) > 1:
                # This is occuring when there is a double count of the compressor stations (entries are identical except for Enverus ID)
                # In this case, assign one compressor station to the field
                Env_StorFields.loc[ifield, 'Comp_flag'] = 1  # could alternatively set this to the number of matches (if actually >1 station per field)
                Env_StorFields.loc[ifield, 'Comp_lat'] = enverus_stations.loc[best_match[0], 'Latitude']
                Env_StorFields.loc[ifield, 'Comp_lon'] = enverus_stations.loc[best_match[0], 'Longitude']
            else:
                best_match = np.where((np.abs(Env_StorFields['Latitude'][ifield]- enverus_stations['Latitude']) < 0.05) & 
                                      (np.abs(Env_StorFields['Longitude'][ifield]- enverus_stations['Longitude']) < 0.05) &
                                      (Env_StorFields['CNTY_NAME'][ifield] == enverus_stations.loc[:, 'CNTY_NAME']))[0]
                if np.size(best_match) >= 1: 
                    Env_StorFields.loc[ifield, 'Comp_flag'] = 1  # could alternatively set this to the number of matches (if actually >1 station per field)
                    Env_StorFields.loc[ifield, 'Comp_lat'] = enverus_stations.loc[best_match[0], 'Latitude']
                    Env_StorFields.loc[ifield, 'Comp_lon'] = enverus_stations.loc[best_match[0], 'Longitude']
                else:
                    nomatch += 1

    print('Number of Enverus Fields w/ Compressor Stations: ', len(Env_StorFields[Env_StorFields['Comp_flag'] == 1]))
    print('Number of Enverus Fields w/out Compressor Stations: ', len(Env_StorFields[Env_StorFields['Comp_flag'] == 0]))

    # Match the EIA and Enverus Storage Field/Station Data, record associated Storage
    # Compressor Station locations

    # First clean up/correct mistakes in arrays
    Env_StorFields['CNTY_NAME'] = Env_StorFields['CNTY_NAME'].str.lower()
    Env_StorFields['RESERVOIR'] = Env_StorFields['RESERVOIR'].str.lower()
    Env_StorFields['NAME'] = Env_StorFields['NAME'].str.lower()
    Env_StorFields['OPERATOR'] = Env_StorFields['OPERATOR'].str.lower()

    EIA_StorFields['Reservoir Name'] = EIA_StorFields['Reservoir Name'].replace({np.nan: 'NaN'})
    EIA_StorFields['County Name'] = EIA_StorFields['County Name'].replace({np.nan: 'NaN'})

    Env_StorFields['RESERVOIR'] = Env_StorFields['RESERVOIR'].str.replace("-", "")
    EIA_StorFields['Reservoir Name'] = EIA_StorFields['Reservoir Name'].str.replace("-", "")
    Env_StorFields['RESERVOIR'] = Env_StorFields['RESERVOIR'].str.replace(".", "")
    EIA_StorFields['Reservoir Name'] = EIA_StorFields['Reservoir Name'].str.replace("(", "")
    EIA_StorFields['Reservoir Name'] = EIA_StorFields['Reservoir Name'].str.replace(")", "")
    Env_StorFields['NAME'] = Env_StorFields['NAME'].str.replace(".", "")
    EIA_StorFields['Reservoir Name'] = EIA_StorFields['Reservoir Name'].str.replace(".", "")
    EIA_StorFields['Field Name'] = EIA_StorFields['Field Name'].str.rstrip()
    EIA_StorFields['Reservoir Name'] = EIA_StorFields['Reservoir Name'].str.rstrip()

    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'CA') & (EIA_StorFields['County Name'] == 'Butte')),'County Name'] = 'colusa'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IA') & (EIA_StorFields['County Name'] == 'Winnebago')),'County Name'] = 'washington'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'La Salle')),'County Name'] = 'lasalle'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'Coles') & (EIA_StorFields['Reservoir Name'] == 'NIAGARIAN')),'County Name'] = 'peoria'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'Coles') & (EIA_StorFields['Reservoir Name'] == 'NIAGARAN')),'County Name'] = 'peoria'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'Coles') & (EIA_StorFields['Reservoir Name'] == 'GLASFORD')),'County Name'] = 'peoria'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'Logan') & (EIA_StorFields['Reservoir Name'] == 'GALESVILLE')),'County Name'] = 'warren'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'Coles') & (EIA_StorFields['Reservoir Name'] == 'BENOIST')),'County Name'] = 'bond'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'Douglas') & (EIA_StorFields['Reservoir Name'] == 'CYPRESS  ROSICL')),'County Name'] = 'moultrie'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'Douglas') & (EIA_StorFields['Reservoir Name'] == 'CYPRESS ROSICLARE')),'County Name'] = 'moultrie'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'Mclean') & (EIA_StorFields['Field Name'] == 'PECATONICA')),'County Name'] = 'winnebago'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IL') & (EIA_StorFields['County Name'] == 'Montgomery') & (EIA_StorFields['Field Name'] == 'HILLSBORO')),'County Name'] = 'st. clair'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IN') & (EIA_StorFields['County Name'] == 'Daviess') & (EIA_StorFields['Field Name'] == 'WHITE RIVER')),'County Name'] = 'pike'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'IN') & (EIA_StorFields['County Name'] == 'Clark') & (EIA_StorFields['Field Name'] == 'WOLCOTT')),'County Name'] = 'white'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'KS') & (EIA_StorFields['County Name'] == 'Woodson') & (EIA_StorFields['Field Name'] == 'PIQUA')),'County Name'] = 'allen'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'KS') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'PIQUA')),'County Name'] = 'allen'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'KS') & (EIA_StorFields['County Name'] == 'Morris') & (EIA_StorFields['Field Name'] == 'BOEHM')),'County Name'] = 'morton'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'KY') & (EIA_StorFields['County Name'] == 'Hart') & (EIA_StorFields['Field Name'] == 'MAGNOLIA UPPER')),'County Name'] = 'larue'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'KY') & (EIA_StorFields['County Name'] == 'Hart') & (EIA_StorFields['Field Name'] == 'MAGNOLIA DEEP')),'County Name'] = 'larue'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'KY') & (EIA_StorFields['County Name'] == 'Meade') & (EIA_StorFields['Field Name'] == 'DOE RUN')),'County Name'] = 'hardin'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'KY') & (EIA_StorFields['County Name'] == 'Daviess') & (EIA_StorFields['Field Name'] == 'EAST DIAMOND')),'County Name'] = 'hopkins'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'KY') & (EIA_StorFields['County Name'] == 'Christian') & (EIA_StorFields['Field Name'] == 'CROFTON EAST')),'County Name'] = 'hopkins'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'LA') & (EIA_StorFields['County Name'] == 'Ascension') & (EIA_StorFields['Field Name'] == 'NAPOLEON')),'County Name'] = 'assumption parish'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'LA') & (EIA_StorFields['County Name'] == 'Ascension') & (EIA_StorFields['Field Name'] == 'NAPOLEONVILLE')),'County Name'] = 'assumption parish'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'LA') & (EIA_StorFields['County Name'] == 'East Carroll') & (EIA_StorFields['Field Name'] == 'EPPS')),'County Name'] = 'west carroll parish'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'LA') & (EIA_StorFields['County Name'] == 'W. Carroll') & (EIA_StorFields['Field Name'] == 'EPPS')),'County Name'] = 'west carroll parish'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'LA') & (EIA_StorFields['County Name'] == 'Iberia') & (EIA_StorFields['Field Name'] == 'JEFFERSON ISLAN')),'County Name'] = 'vermilion parish'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'LA') & (EIA_StorFields['County Name'] == 'Iberia') & (EIA_StorFields['Field Name'] == 'JEFFERSON ISLAND')),'County Name'] = 'vermilion parish'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'Oakland') & (EIA_StorFields['Field Name'] == 'LYON 29')),'County Name'] = 'washtenaw'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'St. Clair') & (EIA_StorFields['Field Name'] == 'MARYSVILLE STORAGE')),'Reservoir Name'] = 'morton 16'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'St. Clair') & (EIA_StorFields['Field Name'] == 'MARYSVILLE STOR')),'Reservoir Name'] = 'morton 16'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'St. Clair') & (EIA_StorFields['Field Name'] == 'LEE 2')),'County Name'] = 'calhoun'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'St. Clair') & (EIA_StorFields['Field Name'] == 'LEE 11')),'County Name'] = 'calhoun'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'WINTERFIELD')),'County Name'] = 'clare'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'CRANBERRY LAKE')),'County Name'] = 'clare'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'HESSEN')),'County Name'] = 'st. clair'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'IRA')),'County Name'] = 'st. clair'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'FOUR CORNERS')),'County Name'] = 'st. clair'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'SWAN CREEK')),'County Name'] = 'st. clair'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'PUTTYGUT')),'County Name'] = 'st. clair'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'WINFIELD')),'County Name'] = 'montcalm'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'St. Clair') & (EIA_StorFields['Field Name'] == 'TAGGART')),'County Name'] = 'montcalm'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MI') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'LOREED')),'County Name'] = 'osceola'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MN') & (EIA_StorFields['County Name'] == 'Waseca') & (EIA_StorFields['Field Name'] == 'WATERVILLE')),'County Name'] = 'steele'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MS') & (EIA_StorFields['County Name'] == 'Adams') & (EIA_StorFields['Field Name'] == 'NEW HOME DOME')),'County Name'] = 'smith'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MS') & (EIA_StorFields['County Name'] == 'Jasper') & (EIA_StorFields['Field Name'] == 'NEW HONE DOME')),'County Name'] = 'smith'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MS') & (EIA_StorFields['County Name'] == 'Monroe') & (EIA_StorFields['Field Name'] == 'GOODWIN')),'County Name'] = 'itawamba'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MS') & (EIA_StorFields['County Name'] == 'Monroe') & (EIA_StorFields['Field Name'] == 'GOODWIN STORAGE')),'County Name'] = 'itawamba'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MS') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'HATTIESBURG')),'County Name'] = 'forrest'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MS') & (EIA_StorFields['County Name'] == 'Montgomery') & (EIA_StorFields['Field Name'] == 'SOUTHERN PINES')),'County Name'] = 'greene'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'MT') & (EIA_StorFields['County Name'] == 'Blaine') & (EIA_StorFields['Field Name'] == 'DRY CREEK')),'County Name'] = 'carbon'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'NY') & (EIA_StorFields['County Name'] == 'Medina') & (EIA_StorFields['Field Name'] == 'BENNINGTON STOR')),'County Name'] = 'wyoming'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'NY') & (EIA_StorFields['County Name'] == 'Erie') & (EIA_StorFields['Field Name'] == 'BENNINGTON STOR')),'County Name'] = 'wyoming'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'NY') & (EIA_StorFields['County Name'] == 'Erie') & (EIA_StorFields['Field Name'] == 'BENNINGTON STORAGE')),'County Name'] = 'wyoming'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'NY') & (EIA_StorFields['County Name'] == 'Kings') & (EIA_StorFields['Field Name'] == 'BEECH HILL STORAGE')),'County Name'] = 'allegany'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'NY') & (EIA_StorFields['County Name'] == 'Putnam') & (EIA_StorFields['Field Name'] == 'SENECA LAKE STORAGE')),'County Name'] = 'schuyler'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'NY') & (EIA_StorFields['County Name'] == 'Putnam') & (EIA_StorFields['Field Name'] == 'DUNDEE')),'County Name'] = 'schuyler'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'OH') & (EIA_StorFields['County Name'] == 'Hocking') & (EIA_StorFields['Field Name'] == 'CRAWFORD')),'County Name'] = 'fairfield'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'OH') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'BRINKER')),'County Name'] = 'columbiana'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'OH') & (EIA_StorFields['County Name'] == 'Wayne') & (EIA_StorFields['Field Name'] == 'GABOR WERTZ')),'County Name'] = 'summit'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'OH') & (EIA_StorFields['County Name'] == 'Hancock') & (EIA_StorFields['Field Name'] == 'BENTON')),'County Name'] = 'hocking'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'OH') & (EIA_StorFields['County Name'] == 'Wayne') & (EIA_StorFields['Field Name'] == 'HOLMES')),'County Name'] = 'holmes'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'OK') & (EIA_StorFields['County Name'] == 'Grady') & (EIA_StorFields['Field Name'] == 'SALT PLAINS STO')),'County Name'] = 'grant'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'PA') & (EIA_StorFields['County Name'] == 'Warren') & (EIA_StorFields['Field Name'] == 'EAST BRANCH STO')),'County Name'] = 'mckean'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'PA') & (EIA_StorFields['County Name'] == 'Warren') & (EIA_StorFields['Field Name'] == 'EAST BRANCH STORAGE')),'County Name'] = 'mckean'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'PA') & (EIA_StorFields['County Name'] == 'Potter') & (EIA_StorFields['Field Name'] == 'LEIDY TAMARACK')),'County Name'] = 'clinton'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'PA') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'LEIDY TAMARACK')),'County Name'] = 'clinton'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'PA') & (EIA_StorFields['County Name'] == 'Allegheny') & (EIA_StorFields['Field Name'] == 'WEBSTER')),'County Name'] = 'westmoreland'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'PA') & (EIA_StorFields['County Name'] == 'Allegheny') & (EIA_StorFields['Field Name'] == 'RAGER MOUNTAIN')),'County Name'] = 'cambria'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'PA') & (EIA_StorFields['County Name'] == 'Mercer') & (EIA_StorFields['Field Name'] == 'HENDERSON STORA')),'County Name'] = 'venango'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'PA') & (EIA_StorFields['County Name'] == 'Mercer') & (EIA_StorFields['Field Name'] == 'HENDERSON STORAGE')),'County Name'] = 'venango'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'TX') & (EIA_StorFields['County Name'] == 'Fort Bend') & (EIA_StorFields['Field Name'] == 'KATY HUB & STOR')),'County Name'] = 'waller'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'TX') & (EIA_StorFields['County Name'] == 'Fort Bend') & (EIA_StorFields['Field Name'] == 'KATY HUB & STORA')),'County Name'] = 'waller'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'WV') & (EIA_StorFields['County Name'] == 'Doddridge') & (EIA_StorFields['Field Name'] == 'SHIRLEY')),'County Name'] = 'tyler'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'WV') & (EIA_StorFields['County Name'] == 'Raleigh') & (EIA_StorFields['Field Name'] == 'RALEIGH CITY')),'County Name'] = 'wyoming'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'WV') & (EIA_StorFields['County Name'] == 'Kanawha') & (EIA_StorFields['Field Name'] == 'RALEIGH CITY')),'County Name'] = 'wyoming'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'WY') & (EIA_StorFields['County Name'] == 'Fremont') & (EIA_StorFields['Field Name'] == 'BUNKER HILL')),'County Name'] = 'carbon'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'WV') & (EIA_StorFields['County Name'] == 'Wirt') & (EIA_StorFields['Field Name'] == 'ROCKPORT')),'County Name'] = 'wood'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'WV') & (EIA_StorFields['County Name'] == 'Ritchie') & (EIA_StorFields['Field Name'] == 'RACKET  NEW BER')),'County Name'] = 'gilmer'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'WV') & (EIA_StorFields['County Name'] == 'Ritchie') & (EIA_StorFields['Field Name'] == 'RACHET-NEWBERNE')),'County Name'] = 'gilmer'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'TX') & (EIA_StorFields['County Name'] == 'NaN') & (EIA_StorFields['Field Name'] == 'WEST CLEAR LAKE')),'County Name'] = 'harris'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'TX') & (EIA_StorFields['County Name'] == 'Bastrop') & (EIA_StorFields['Field Name'] == 'PIERCE JUNCTION')),'County Name'] = 'harris'
    Env_StorFields.loc[((Env_StorFields['State'] == 'KS') & (Env_StorFields['NAME'] == 'welda (north)')), 'NAME'] = 'north welda'
    Env_StorFields.loc[((Env_StorFields['State'] == 'KS') & (Env_StorFields['NAME'] == 'welda (south)')), 'NAME'] = 'south welda'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'TX') & (EIA_StorFields['Reservoir Name'] == 'DW69')), 'Reservoir Name'] = 'dw 6'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'TX') & (EIA_StorFields['Reservoir Name'] == 'DW 69')), 'Reservoir Name'] = 'dw 6'
    EIA_StorFields.loc[((EIA_StorFields['Report State'] == 'PA') & (EIA_StorFields['Field Name'] == 'ST  MARYS STORAGE')), 'Field Name'] = 'ST MARYS STORAGE'

    # Second, loop through each EIA storage field, find the matching field in Enverus (by county, state, reservoir,
    # company, operator, etc) and record the associated storage compressor station location (if the field has one)
    # NOTE: As of March 2025, there are ~11 EIA fields that could not be matched to the Enverus dataset. In this
    # case, these fields are assumed to have zero compressor stations and are not accounted for in the national
    # or CONUS total capacity calculations (used to calculate emissions based on ratio to GHGRP data)

    EIA_StorFields['Lat'] = 0
    EIA_StorFields['Lon'] = 0
    EIA_StorFields['Comp_flag'] = 0
    DEBUG = 0

    print('QA/QC: The following EIA fields could not be matched to Enverus Gas Storage Fields')
    print('Assume that these fields have 0 storage compressor stations')

    for ifield in np.arange(0,len(EIA_StorFields)):
        # first match based on state and county, then match either reservoir or name
        matched = np.where((EIA_StorFields['Report State'][ifield] == Env_StorFields['State']) & \
                            (Env_StorFields['CNTY_NAME'].str.contains(EIA_StorFields['County Name'][ifield][0:5].lower())))[0]
        if EIA_StorFields['Reservoir Name'][ifield][0:10] != 'NaN':
            # for all the fields in the same state and county, choose the field that is in the same reservoir
            best_match = np.where(Env_StorFields['RESERVOIR'][matched].str.contains(EIA_StorFields['Reservoir Name'][ifield][0:10].lower()))[0]
            
            if np.size(best_match) == 1:
                loc = matched[best_match[0]]
                EIA_StorFields.loc[ifield, 'Lat'] = Env_StorFields.loc[loc, 'Comp_lat']
                EIA_StorFields.loc[ifield, 'Lon'] = Env_StorFields.loc[loc, 'Comp_lon']
                EIA_StorFields.loc[ifield, 'Comp_flag'] = Env_StorFields.loc[loc, 'Comp_flag']
            elif np.size(best_match) > 1:
                # there is more than one field in the state, county, and reservoir - assign based on either matching company or field name, if
                # still more than one match, assign manually
                if 'liberty north' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains('liberty north'))[0]
                    loc = matched[best_match[better_match[0]]]
                elif 'liberty south' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains('liberty south'))[0]
                    loc = matched[best_match[better_match[0]]]
                elif 'st  charles' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains('st charles'))[0]
                    loc = matched[best_match[better_match[0]]] 
                elif 'east diamond' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched].str.contains('east diamond'))[0]
                    loc = matched[better_match[0]]
                elif 'crofton east' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains('kirkwood springs'))[0]
                    loc = matched[best_match[better_match[0]]] 
                elif 'cold springs' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched][0:14].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()))[0]
                    loc = matched[better_match[0]]
                elif 'niagaran' in EIA_StorFields.loc[ifield,'Reservoir Name'].lower() :
                    if 'belle river' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                        better_match = np.where(Env_StorFields['NAME'][matched].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()))[0]
                        loc = matched[better_match[0]]
                    else:
                        better_match = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains(EIA_StorFields.loc[ifield,'Field Name'][0:12].lower()))[0]
                        loc = matched[best_match[better_match[0]]]
                elif EIA_StorFields.loc[ifield,'Reservoir Name'].lower() == 'niagaran':
                    better_match = np.where(Env_StorFields['NAME'][matched].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()))[0]
                    loc = matched[better_match[0]]
                elif 'washington ' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched][0:13].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()))[0]
                    loc = matched[better_match[0]]
                elif 'greenwood' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched][0:13].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()))[0]
                    loc = matched[better_match[0]]
                elif 'amory storage' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()[0:5]))[0]
                    loc = matched[better_match[0]]
                elif 'derby storage' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()[0:5]))[0]
                    loc = matched[better_match[0]]
                elif 'zane storage' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains('zane'))[0]
                    loc = matched[best_match[better_match[0]]]
                elif 'artemas ' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched][0:11].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()))[0]
                    loc = matched[better_match[0]]
                elif 'ellisburg' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['OPERATOR'][matched[best_match]].str.contains('berkshire'))[0]
                    loc = matched[best_match[better_match[0]]]
                elif 'stratton ridge' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['OPERATOR'][matched[best_match]].str.contains('freeport'))[0]
                    loc = matched[best_match[better_match[0]]]
                elif 'spindletop' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['OPERATOR'][matched].str.contains(EIA_StorFields.loc[ifield,'Company Name'].lower()[0:6]))[0]
                    loc = matched[better_match[0]]
                elif 'ambassador' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains('la-pan'))[0]
                    loc = matched[best_match[better_match[0]]]
                elif 'terra alta' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    better_match = np.where(Env_StorFields['NAME'][matched][0:15].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()))[0]
                    loc = matched[better_match[0]]
                elif EIA_StorFields.loc[ifield,'Field Name'].lower() == 'eden':
                    better_match = np.where(Env_StorFields['NAME'][matched] == 'eden')[0]
                    loc = matched[better_match[0]]
                else:
                    better_match = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains(EIA_StorFields['Company Name'][ifield][0:6].lower()) | \
                                (Env_StorFields['NAME'][matched[best_match]].str.contains(EIA_StorFields['Field Name'][ifield][0:6].lower())))[0]
                    loc = matched[best_match[better_match[0]]]
                    if np.size(better_match) != 1:
                        print('> 1 match - Check Manually')
                        print(EIA_StorFields.loc[ifield,:])
                        display(Env_StorFields.loc[matched[best_match[better_match]],:])
                
                # Assign lat/Lon, whether there is compressor station there
                EIA_StorFields.loc[ifield, 'Lat'] = Env_StorFields.loc[loc, 'Comp_lat']
                EIA_StorFields.loc[ifield, 'Lon'] = Env_StorFields.loc[loc, 'Comp_lon']
                EIA_StorFields.loc[ifield, 'Comp_flag'] = Env_StorFields.loc[loc, 'Comp_flag']
            else:
                # this will occur if there is a match based on county/state, but not on reservoir
                # (or some cases where no match on county/state)
                # in this case, look at company and field name and assign mannually if needed
                if 'totem storage' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched].str.contains('totem'))[0]
                elif 'lincoln storage' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched].str.contains('lincoln'))[0]
                elif 'cecilia storage' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched].str.contains('cecilia'))[0]
                elif 'egan storage do' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched].str.contains('egan'))[0]
                elif 'washington ' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched].str.contains(EIA_StorFields.loc[ifield,'Field Name'].lower()[0:13]))[0]
                elif 'petal' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched][0:5].str.contains('hattiesburg'))[0]
                elif 'zoar storage' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched].str.contains('zoar'))[0]
                elif 'love storage' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched][0:5].str.contains('perry'))[0]
                elif 'swarts and swar' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched].str.contains('swarts'))[0]
                elif 'clemens  n.e.' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched][0:5].str.contains('clemens'))[0]
                elif 'worsham steed' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched][0:6].str.contains('worsham-steed'))[0]
                elif 'early grove' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched][0:6].str.contains('early grove'))[0]
                elif 'terra alta' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    if 'south' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                        best_match = np.where(Env_StorFields['NAME'][matched].str.contains('south'))[0]
                    else:
                        best_match = np.where(~Env_StorFields['NAME'][matched].str.contains('south'))[0]           
                elif 'racket ' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched][0:15].str.contains('rachet-newberne'))[0]
                elif 'rachet' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched][0:15].str.contains('rachet-newberne'))[0]
                elif 'ryckman creek' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched][0:15].str.contains('belle butte'))[0]
                elif 'east mahoney' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched][0:15].str.contains('oil springs'))[0]
                else:
                    best_match = np.where(Env_StorFields['NAME'][matched][0:8].str.contains(EIA_StorFields['Company Name'][ifield][0:8].lower()) | \
                                (Env_StorFields['NAME'][matched][0:8].str.contains(EIA_StorFields['Field Name'][ifield][0:8].lower())))[0]
                if np.size(best_match) != 1:
                    # continue
                    if DEBUG ==1:
                        print('No best match - Check Mannually')
                        print(EIA_StorFields.loc[ifield, :])
                        display(Env_StorFields.loc[matched[best_match], :])
                else:
                    loc = matched[best_match[0]]
                    # Assign lat/Lon, whether there is compressor station there
                    EIA_StorFields.loc[ifield, 'Lat'] = Env_StorFields.loc[loc, 'Comp_lat']
                    EIA_StorFields.loc[ifield, 'Lon'] = Env_StorFields.loc[loc, 'Comp_lon']
                    EIA_StorFields.loc[ifield, 'Comp_flag'] = Env_StorFields.loc[loc, 'Comp_flag']

        else:
            # in this case, the EIA data has no reservoir information and need to match based on company/field name
            best_match = np.where((Env_StorFields['NAME'][matched][0:6].str.contains(EIA_StorFields['Company Name'][ifield][0:6].lower())) | \
                                (Env_StorFields['NAME'][matched][0:6].str.contains(EIA_StorFields['Field Name'][ifield][0:6].lower())))[0]
            if np.size(best_match) == 1:
                loc = matched[best_match[0]]
                EIA_StorFields.loc[ifield, 'Lat'] = Env_StorFields.loc[loc, 'Comp_lat']
                EIA_StorFields.loc[ifield, 'Lon'] = Env_StorFields.loc[loc, 'Comp_lon']
                EIA_StorFields.loc[ifield, 'Comp_flag'] = Env_StorFields.loc[loc, 'Comp_flag']
            elif np.size(best_match) > 1:
                # if more than one match based on county/state, and no reservoir data...look at operator
                better_match = np.where((Env_StorFields['OPERATOR'][matched[best_match]][0:6].str.contains(EIA_StorFields['Company Name'][ifield][0:6].lower())))[0]
                if np.size(better_match) == 1:
                    # if one operator match...
                    loc = matched[best_match[better_match[0]]]
                    EIA_StorFields.loc[ifield, 'Lat'] = Env_StorFields.loc[loc, 'Comp_lat']
                    EIA_StorFields.loc[ifield, 'Lon'] = Env_StorFields.loc[loc, 'Comp_lon']
                    EIA_StorFields.loc[ifield, 'Comp_flag'] = Env_StorFields.loc[loc, 'Comp_flag']
                elif np.size(better_match) > 1:
                    # if more than one operator match...
                    if EIA_StorFields.loc[ifield, 'Field Name'].lower() == 'kirby hills wagenet':
                        finalmatch = np.where(Env_StorFields['RESERVOIR'][matched[best_match[better_match]]].str.contains('wagenet'))[0]
                        loc = matched[best_match[better_match[finalmatch[0]]]]
                    elif 'early grove' in EIA_StorFields.loc[ifield, 'Field Name'].lower():
                        finalmatch = np.where(Env_StorFields['NAME'][matched[best_match[better_match]]].str.contains('early grove'))[0] 
                        loc = matched[best_match[better_match[finalmatch[0]]]]
                    else:
                        print('HERE STOP*****************************')
                        loc = -99
                else:
                    # if no operator match
                    if EIA_StorFields.loc[ifield, 'Field Name'].lower() == 'markham':
                        finalmatch = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains('markham'))[0]
                        loc = matched[best_match[finalmatch[0]]]
                    elif EIA_StorFields.loc[ifield, 'Field Name'].lower() == 'kirby hills wagenet':
                        finalmatch = np.where(Env_StorFields['NAME'][matched[best_match]].str.contains('kirby hills ii'))[0]
                        loc = matched[best_match[finalmatch[0]]]
                    else:
                        print('STOP HERE2*******************************')
                        loc = -99
                # Assign lat/lon values
                EIA_StorFields.loc[ifield,'Lat'] = Env_StorFields.loc[loc,'Comp_lat']
                EIA_StorFields.loc[ifield,'Lon'] = Env_StorFields.loc[loc,'Comp_lon']
                EIA_StorFields.loc[ifield,'Comp_flag'] = Env_StorFields.loc[loc,'Comp_flag']     
        
            else:
                # if no reservoir, or company or field match
                if 'egan storage dome' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'][matched].str.contains('egan'))[0]
                    loc = matched[best_match[0]]
                elif 'new home dome' in EIA_StorFields.loc[ifield,'Field Name'].lower():
                    best_match = np.where(Env_StorFields['NAME'].str.contains('new home dome'))[0]
                    loc = best_match[0]
                else:
                    # print(best_match)
                    print('No Res Data - Check Mannually')
                    print(EIA_StorFields.loc[ifield,:])
                    display(Env_StorFields.loc[matched,:])
                    loc = -99
                if loc > 0:
                    # Assign lat/lon values
                    EIA_StorFields.loc[ifield, 'Lat'] = Env_StorFields.loc[loc, 'Comp_lat']
                    EIA_StorFields.loc[ifield, 'Lon'] = Env_StorFields.loc[loc, 'Comp_lon']
                    EIA_StorFields.loc[ifield, 'Comp_flag'] = Env_StorFields.loc[loc, 'Comp_flag']  

    print('QA/QC: Report the number of EIA fields and Enverus storage compressor stations')
    for iyear in years:
        total_stations = len(EIA_StorFields[(EIA_StorFields['Year'] == iyear) & (EIA_StorFields['Comp_flag'] == 1)])
        total_fields = len(EIA_StorFields[EIA_StorFields['Year'] == iyear])
        print('Year: ', iyear)
        print('Fields, stations, (%):', total_fields,',', total_stations, ',', round((total_stations/total_fields) * 100, 2))

    # Read in GHGRP Storage Compressor Station Data
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
                                # Filter for underground natural gas storage
                                .query("industry_segment == 'Underground natural gas storage [98.230(a)(5)]'")
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
    # Manually add the missing counties to the spreadsheet in the
    # ghgrp_missing_counties_path and add the county names.
    # missing_counties = ghgrp_facility_emissions[ghgrp_facility_emissions['county'].isna()]
    # missing_counties.to_csv('missing_counties.csv')

    # Read in missing counties
    missing_counties = pd.read_excel(ghgrp_missing_counties_path, sheet_name='ng_storage_comp_station_proxy')

    # Add missing county information
    for istation in np.arange(0, len(missing_counties)):
        ifacility_id = str(missing_counties.loc[istation, 'facility_id'])
        icounty = missing_counties.loc[istation, 'county']
        ghgrp_facility_emissions.loc[ghgrp_facility_emissions['facility_id'] == ifacility_id, 'county'] = icounty
    
    # Match EIA storage compressor stations to GHGRP based on location
    # For each year of GHGRP data, match GHGRP stroage stations to EIA data (based on nearest location, not name)
    GHGRP_matched = pd.DataFrame()
    GHGRP_notmatched = pd.DataFrame()
    EIA_matched = pd.DataFrame()
    EIA_notmatched = pd.DataFrame()
    for iyear in years:
        # Use the correct year of data and filter only for fields with storage compressor stations
        GHGRP_temp_data = ghgrp_facility_emissions.query(f"year == {iyear}").reset_index(drop=True)
        EIA_StorFields_temp = (EIA_StorFields[(EIA_StorFields['Year'] == iyear) & (EIA_StorFields['Comp_flag'] == 1)]).reset_index(drop=True)
        # Add columns to GHGRP data to store matching EIA data
        GHGRP_temp_data.loc[:, 'match_flag'] = 0
        GHGRP_temp_data.loc[:, 'EIA_name'] = ''
        GHGRP_temp_data.loc[:, 'EIA_county'] = ''
        GHGRP_temp_data.loc[:, 'EIA_state'] = ''
        GHGRP_temp_data.loc[:, 'EIA_fieldcap'] = 0
        EIA_StorFields_temp.loc[:, 'GHGRP_match'] = 0
        # First, find exact matching lat/lon facilities (within 0.12)
        for istation in np.arange(0, len(GHGRP_temp_data)):
            matched = np.where((np.abs(EIA_StorFields_temp['Lat'] - GHGRP_temp_data['latitude'][istation]) < 0.12) & 
                               (np.abs(EIA_StorFields_temp['Lon'] - GHGRP_temp_data['longitude'][istation]) < 0.12))[0]
            # Assign EIA data to GHGRP if there is exactly one match
            if np.size(matched) == 1:
                EIA_StorFields_temp.loc[matched[0], 'GHGRP_match'] = 1
                GHGRP_temp_data.loc[istation, 'match_flag'] = 1
                GHGRP_temp_data.loc[istation, 'EIA_name'] = EIA_StorFields_temp.loc[matched[0], 'Company Name']
                GHGRP_temp_data.loc[istation, 'EIA_county'] = EIA_StorFields_temp.loc[matched[0], 'County Name']
                GHGRP_temp_data.loc[istation, 'EIA_state'] = EIA_StorFields_temp.loc[matched[0], 'Report State']
                GHGRP_temp_data.loc[istation, 'EIA_fieldcap'] = EIA_StorFields_temp.loc[matched[0], 'Total Field Capacity(Mcf)']
            # If there is more than one match, loop through the matching stations to find the closest match
            elif np.size(matched) > 1:
                dist_calc = np.zeros(len(matched))
                GHGRP_temp_data.loc[istation, 'match_flag'] = 1
                # Calculate the distance between each potential EIA match and the GHGRP location
                for imatch in np.arange(len(dist_calc)):
                    dist_calc[imatch] = (np.abs(GHGRP_temp_data.loc[istation, 'latitude']
                                                - EIA_StorFields_temp.loc[matched[imatch], 'Lat'])**2)
                    + np.abs(GHGRP_temp_data.loc[istation, 'longitude'] 
                             - EIA_StorFields_temp.loc[matched[imatch], 'Lon'])**2
                # Select where the distance between the two locations (GHGRP and the
                # potential EIA matches) is minimized.
                bestpick = np.where(dist_calc == dist_calc.min())[0][0]
                # If there is only one location that is the closest, assign the data to
                # the GHGRP array.
                if np.size(np.where(dist_calc == dist_calc.min())[0]) == 1:
                    EIA_StorFields_temp.loc[matched[bestpick],'GHGRP_match'] = 1
                    GHGRP_temp_data.loc[istation, 'match_flag'] = 1
                    GHGRP_temp_data.loc[istation, 'EIA_name'] = EIA_StorFields_temp.loc[matched[bestpick], 'Company Name']
                    GHGRP_temp_data.loc[istation, 'EIA_county'] = EIA_StorFields_temp.loc[matched[bestpick], 'County Name']
                    GHGRP_temp_data.loc[istation, 'EIA_state'] = EIA_StorFields_temp.loc[matched[bestpick], 'Report State']
                    GHGRP_temp_data.loc[istation, 'EIA_fieldcap'] = EIA_StorFields_temp.loc[matched[bestpick], 'Total Field Capacity(Mcf)']
                # If there are more than one locations that are the closest, sum the
                # field capacity from all matching stations and assign the average to
                # the GHGRP array
                else:
                    best_array = np.where(dist_calc == dist_calc.min())[0]
                    total_stor = 0.0
                    nonzero_stor = 0
                    for ibest in np.arange(0, len(best_array)):
                        if EIA_StorFields_temp.loc[matched[best_array[ibest]], 'Total Field Capacity(Mcf)'] > 0:
                            total_stor += EIA_StorFields_temp.loc[matched[best_array[ibest]], 'Total Field Capacity(Mcf)']
                            nonzero_stor += 1
                        EIA_StorFields_temp.loc[matched[best_array[ibest]], 'GHGRP_match'] = 1
                    GHGRP_temp_data.loc[istation, 'match_flag'] = 1
                    GHGRP_temp_data.loc[istation, 'EIA_county'] = EIA_StorFields_temp.loc[matched[best_array[0]], 'County Name']
                    GHGRP_temp_data.loc[istation, 'EIA_state'] = EIA_StorFields_temp.loc[matched[best_array[0]], 'Report State']
                    GHGRP_temp_data.loc[istation, 'EIA_fieldcap'] = safe_div(total_stor, nonzero_stor)
            # Manually add these missing matches
            else:
                if GHGRP_temp_data.loc[istation, 'facility_name'] == 'SNG Station 4020 Bear Creek Storage, LA':
                    matched = np.where((EIA_StorFields_temp['Company Name'] == 'BEAR CREEK STORAGE COMPANY') & 
                                       (EIA_StorFields_temp['Report State'] == 'LA'))[0]
                    EIA_StorFields_temp.loc[matched[0], 'GHGRP_match'] = 1
                    GHGRP_temp_data.loc[istation, 'match_flag'] = 1
                    GHGRP_temp_data.loc[istation, 'EIA_name'] = EIA_StorFields_temp.loc[matched[0], 'Company Name']
                    GHGRP_temp_data.loc[istation, 'EIA_county'] = EIA_StorFields_temp.loc[matched[0], 'County Name']
                    GHGRP_temp_data.loc[istation, 'EIA_state'] = EIA_StorFields_temp.loc[matched[0], 'Report State']
                    GHGRP_temp_data.loc[istation, 'EIA_fieldcap'] = EIA_StorFields_temp.loc[matched[0], 'Total Field Capacity(Mcf)']

                elif GHGRP_temp_data.loc[istation,'facility_id'] == 1009849:
                    matched = np.where((EIA_StorFields_temp['Field Name'] == 'BOLING') & 
                                       (EIA_StorFields_temp['Report State'] == 'TX'))[0]
                    EIA_StorFields_temp.loc[matched[0], 'GHGRP_match'] = 1
                    GHGRP_temp_data.loc[istation, 'match_flag'] = 1
                    GHGRP_temp_data.loc[istation, 'EIA_name'] = EIA_StorFields_temp.loc[matched[0], 'Company Name']
                    GHGRP_temp_data.loc[istation, 'EIA_county'] = EIA_StorFields_temp.loc[matched[0], 'County Name']
                    GHGRP_temp_data.loc[istation, 'EIA_state'] = EIA_StorFields_temp.loc[matched[0], 'Report State']
                    GHGRP_temp_data.loc[istation, 'EIA_fieldcap'] = EIA_StorFields_temp.loc[matched[0], 'Total Field Capacity(Mcf)']
        
        # Save the final matching and non-matching datasets
        # All GHGRP fields (matched and non-matched)
        GHGRP_all = pd.concat([GHGRP_matched, GHGRP_temp_data]).reset_index(drop=True)
        # GHGRP fields with a matching EIA field
        GHGRP_matched = pd.concat([GHGRP_matched, GHGRP_temp_data.query("match_flag == 1")]).reset_index(drop=True)
        # GHGRP fields without a matching EIA field
        GHGRP_notmatched = pd.concat([GHGRP_notmatched, GHGRP_temp_data.query("match_flag == 0")]).reset_index(drop=True)
        # EIA fields that have a matching GHGRP field
        EIA_matched = pd.concat([EIA_matched, EIA_StorFields_temp.query("GHGRP_match == 1")]).reset_index(drop=True)
        # EIA fields that do not have a matching GHGRP field
        EIA_notmatched = pd.concat([EIA_notmatched, EIA_StorFields_temp.query("GHGRP_match == 0")]).reset_index(drop=True)
        
        print('QA/QC: Number of GHGRP plants not in EIA data set')
        print('Year ', iyear,': ', len(GHGRP_notmatched.query(f"year == {iyear}")), ' of ', len(GHGRP_temp_data))

    # Verify that the GHGRP and EIA fields without matches do not actually have matches.
    # If you find some matches, manually add the matches to the final else statement above.
    # Code is commented out unless you need to manually check matches.
    # GHGRP_notmatched.to_csv('GHGRP_notmatched.csv')
    # EIA_notmatched.to_csv('EIA_notmatched.csv')

    # Calculate average ratio of emissions per total field capacity (for fields with compressor stations)
    GHGRP_all['Emis_cap_ratio'] = 0.0
    for iplant in np.arange(0, len(GHGRP_all)):
        GHGRP_all.loc[iplant, 'Emis_cap_ratio'] = (GHGRP_all.loc[iplant, 'ch4_emi'] / GHGRP_all.loc[iplant, 'EIA_fieldcap'] if GHGRP_all.loc[iplant, 'EIA_fieldcap'] > 0 else 0)

    # Calculate the average emissions:total field capacity ratio by year
    avg_emis_cap_ratio = np.zeros(len(years))
    for iyear in np.arange(0, len(years)):
        avg_emis_cap_ratio[iyear] = np.mean(GHGRP_all.query(f"year == {years[iyear]}")['Emis_cap_ratio'])

    # Print the calculated average emissions:total field capacity ratios by year
    print('QA/QC: Average Emissions to Field Capacity Ratio')
    for iyear in np.arange(0, len(years)):
        print('Year ', years[iyear],': ', avg_emis_cap_ratio[iyear])
    
    # Assign emissions to the unmatched EIA fields using the annual average emissions/capacity ratios
    EIA_notmatched_ch4_added = pd.DataFrame()
    for iyear in np.arange(0, len(years)):
        avg_emis_cap_ratio_iyear = avg_emis_cap_ratio[iyear]
        EIA_notmatched_iyear = (EIA_notmatched
                                .query(f"Year == {years[iyear]}")
                                .assign(ch4_emi=lambda df: df['Total Field Capacity(Mcf)'] * avg_emis_cap_ratio_iyear))
        EIA_notmatched_ch4_added = pd.concat([EIA_notmatched_ch4_added, EIA_notmatched_iyear]).reset_index(drop=True)
    
    # Rename EIA columns to match GHGRP
    EIA_notmatched_ch4_added = (EIA_notmatched_ch4_added
                                .rename(columns={"Year": "year",
                                                 "Lat": "latitude",
                                                 "Lon": "longitude"})
                                .dropna(subset=["latitude", "longitude"])
                                )

    # Combine matched GHGRP fields with unmatched EIA fields
    GHGRP_matched = GHGRP_matched.loc[:, ["year", "ch4_emi", "latitude", "longitude"]]
    EIA_notmatched_ch4_added = EIA_notmatched_ch4_added.loc[:, ["year", "ch4_emi", "latitude", "longitude"]]
    proxy_df = pd.concat([GHGRP_matched, EIA_notmatched_ch4_added]).reset_index(drop=True)

    # Calculate facility relative emissions for each year combination
    # NOTE: Emissions are at a national level, so the relative emissions are normalized on a national level with no state_code column
    proxy_df['rel_emi'] = proxy_df.groupby(["year"])['ch4_emi'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    proxy_df = proxy_df.drop(columns='ch4_emi')

    # Convert proxy to geodataframe
    proxy_gdf = (
        gpd.GeoDataFrame(
            proxy_df,
            geometry=gpd.points_from_xy(
                proxy_df["longitude"],
                proxy_df["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["latitude", "longitude"])
        .loc[:, ["year", "geometry", "rel_emi"]]
        .astype({"rel_emi":float})
    )

    # Check that relative emissions sum to 1.0 each state/year combination
    sums = proxy_gdf.groupby(["year"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"  # assert that the sums are close to 1

    # Output proxy parquet files
    proxy_gdf.to_parquet(proxy_output_path)

    return None
