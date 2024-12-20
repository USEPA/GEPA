"""
Name:                   task_stat_comb_proxy.py
Date Last Modified:     2024-10-31
Authors Name:           A. Burnette (RTI International)
Purpose:                Mapping of stationary combustion proxy emissions
Input Files:            -
Output Files:           - elec_coal_proxy.parquet, elec_gas_proxy.parquet,
                        elec_oil_proxy.parquet, elec_wood_proxy.parquet,
                        indu_proxy.parquet
Notes:                  -
"""
########################################################################################
# %% STEP 0.1. Load Packages

from pathlib import Path
import os
from typing import Annotated
from pytask import Product, mark, task

import pandas as pd
import numpy as np

import geopandas as gpd

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    max_year,
    min_year,
    years
)

########################################################################################
# %% Load Path Files

# Pathways
GEPA_Stat_Path = V3_DATA_PATH.parent / "GEPA_Source_Code" / "GEPA_Combustion_Stationary"
Global_Func_Path = V3_DATA_PATH.parent / "Global_Functions" / "Global_Functions"

# Activity Data
EPA_ARP_facilities = GEPA_Stat_Path / "InputData" / "ARP_Data" / "EPA_ARP_2012-2022_Facility_Info.csv"
EPA_ARP_inputfile = GEPA_Stat_Path / "InputData/ARP_Data/EPA_ARP_2012-2022.csv"

#NEI_resi_wood_inputfile = GEPA_Stat_Path / "InputData/NEII 2020 RWC Throughputs.xlsx"

# GHGRP Data (reporting format changed in 2015)
GHGRP_subC_inputfile = GEPA_Stat_Path / "InputData/GHGRP/GHGRP_SubpartCEmissions_2010-2023.csv" #subpart C facility IDs and emissions (locations not available)
GHGRP_subD_inputfile = GEPA_Stat_Path / "InputData/GHGRP/GHGRP_SubpartDEmissions_2010-2023.csv" #subpart D facility IDs and emissions 
GHGRP_subDfacility_loc_inputfile = GEPA_Stat_Path / "InputData/GHGRP/GHGRP_FacilityInfo_2010-2023.csv" #subpart D facility info (for all years, with ID & lat and lons)


########################################################################################
# %% elec_coal_proxy

@mark.persist
@task(id='elec_coal_proxy')
def task_get_reporting_electric_coal_proxy_data(
    facility_path=EPA_ARP_facilities,
    input_path=EPA_ARP_inputfile,
    reporting_electric_coal_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path 
    / 'elec_coal_proxy.parquet'
):
    """
    Relative emissions and location information for reporting facilities are taken from
    the Clean Air Markets Program Data (CAMPD) and the Acid Rain Program (ARP) Data.
    """

    # Read in ARP Facility Data
    ARP_Facility = (
        pd.read_csv(facility_path, index_col=False)
        .filter(items=['Facility ID', 'Facility Name', 'State', 'Latitude', 'Longitude'])
        .drop_duplicates(['Facility ID', 'Latitude', 'Longitude'], keep='last')
        .rename(columns={'Facility ID': 'facility_id'})
        .reset_index(drop=True)
    )

    ARP_Facility_gdf = (
        gpd.GeoDataFrame(
            ARP_Facility,
            geometry=gpd.points_from_xy(
                ARP_Facility['Longitude'],
                ARP_Facility['Latitude'],
                crs=4326
            )
        )
    )

    # Read in Raw ARP Data & Clean Fuel Type
    ARP_Raw = (
        pd.read_csv(input_path, index_col=False)
        .filter(items=['State', 'Facility ID', 'Facility Name', 'Unit ID', 'Year', 'Month',
                       'Heat Input (mmBtu)', 'Primary Fuel Type', 'Unit Type'])
        .query('State not in ["VI", "MP", "GU", "AS", "PR", "AK", "HI"]')
    )

    ARP_Raw = (
        ARP_Raw.assign(
            unit_clean=np.select(
                [
                    ARP_Raw['Unit Type'].str.contains('combustion turbine', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('combined cycle', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('wet bottom', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('dry bottom', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('bubbling', case=False, na=False)
                ],
                [
                    'combustion turbine',
                    'combined cycle',
                    'wet bottom',
                    'dry bottom',
                    'bubbling'
                ],
                default=ARP_Raw['Unit Type'].str.lower()
            ),
            fuel_clean=np.select(
                [
                    ARP_Raw['Primary Fuel Type'].isin(['Pipeline Natural Gas',
                                                       'Natural Gas',
                                                       'Other Gas',
                                                       'Other Gas, Pipeline Natural Gas',
                                                       'Natural Gas, Pipeline Natural Gas',
                                                       'Process Gas']),
                    ARP_Raw['Primary Fuel Type'].isin(['Petroleum Coke',
                                                       'Coal',
                                                       'Coal, Pipeline Natural Gas',
                                                       'Coal, Natural Gas',
                                                       'Coal, Wood',
                                                       'Coal, Coal Refuse',
                                                       'Coal Refuse',
                                                       'Coal, Process Gas',
                                                       'Coal, Diesel Oil',
                                                       'Other Solid Fuel']),
                    ARP_Raw['Primary Fuel Type'].isin(['Other Oil',
                                                       'Diesel Oil',
                                                       'Diesel Oil, Residual Oil',
                                                       'Diesel Oil, Pipeline Natural Gas',
                                                       'Residual Oil',
                                                       'Residual Oil, Pipeline Natural Gas']),
                    ARP_Raw['Primary Fuel Type'].isin(['Wood',
                                                       'Other Solid Fuel, Wood'])
                ],
                [
                    'Gas',
                    'Coal',
                    'Oil',
                    'Wood'
                ],
                default=ARP_Raw['Primary Fuel Type']
            )
        )
        .query('fuel_clean == "Coal"')
    )

    # Coerce Heat Input (mmBtu)
    ARP_Raw['Heat Input (mmBtu)'] = pd.to_numeric(ARP_Raw['Heat Input (mmBtu)'], errors='coerce').fillna(0)

    # Calculate CH4 flux for each facility
    ARP_Raw = (
        ARP_Raw.assign(
            ch4_f=np.select(
                [
                    # For Coal
                    (ARP_Raw['fuel_clean'] == 'Coal') &
                    (ARP_Raw['unit_clean'].isin(['tangentially-fired', 'dry bottom'])),

                    (ARP_Raw['fuel_clean'] == 'Coal') &
                    (ARP_Raw['unit_clean'] == 'wet bottom'),

                    (ARP_Raw['fuel_clean'] == 'Coal') &
                    (ARP_Raw['unit_clean'] == 'cyclone boiler'),

                    (ARP_Raw['fuel_clean'] == 'Coal')
                ],
                [

                    0.7,    # Coal: tangentially-fired or dry bottom
                    0.9,    # Coal: wet bottom
                    0.2,    # Coal: cyclone boiler
                    1,      # Coal: others
                ],
                default=np.nan
            )
            )
    )

    # Calculate CH4 flux for each facility by multiplying heat input by CH4 factor
    ARP_Clean = (
        ARP_Raw.assign(
            ch4_flux=ARP_Raw['Heat Input (mmBtu)'] * ARP_Raw['ch4_f']
            )
        .drop(columns=['Unit ID', 'Heat Input (mmBtu)', 'Primary Fuel Type', 'Unit Type',
                       'unit_clean', 'ch4_f', 'fuel_clean'])
        .rename(columns={'Facility ID': 'facility_id',
                         'Facility Name': 'facility_name',
                         'State': 'state_code',
                         'Year': 'year',
                         'Month': 'month'})
        .groupby(['facility_id', 'facility_name', 'state_code', 'year', 'month'])
        .sum('ch4_flux')
        .sort_values(['facility_id', 'year', 'month'])
        .reset_index()
        )

    # Merge with ARP Facility Data
    ARP_Full = (
        ARP_Clean.merge(ARP_Facility_gdf[['facility_id', 'geometry']],
                        on='facility_id', how='left')
    )

    ARP_Full = gpd.GeoDataFrame(ARP_Full, geometry='geometry', crs=4326)

    ARP_Full.to_parquet(reporting_electric_coal_proxy_output_path)
    return None


########################################################################################
# %% elec_gas_proxy

@mark.persist
@task(id='elec_gas_proxy')
def task_get_reporting_electric_gas_proxy_data(
    facility_path=EPA_ARP_facilities,
    input_path=EPA_ARP_inputfile,
    reporting_electric_gas_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path 
    / 'elec_gas_proxy.parquet'
):
    """
    Relative emissions and location information for reporting facilities are taken from
    the Clean Air Markets Program Data (CAMPD) and the Acid Rain Program (ARP) Data.
    """

    # Read in ARP Facility Data
    ARP_Facility = (
        pd.read_csv(facility_path, index_col=False)
        .filter(items=['Facility ID', 'Facility Name', 'State', 'Latitude', 'Longitude'])
        .drop_duplicates(['Facility ID', 'Latitude', 'Longitude'], keep='last')
        .rename(columns={'Facility ID': 'facility_id'})
        .reset_index(drop=True)
    )

    ARP_Facility_gdf = (
        gpd.GeoDataFrame(
            ARP_Facility,
            geometry=gpd.points_from_xy(
                ARP_Facility['Longitude'],
                ARP_Facility['Latitude'],
                crs=4326
            )
        )
    )

    # Read in Raw ARP Data & Clean Fuel Type
    ARP_Raw = (
        pd.read_csv(input_path, index_col=False)
        .filter(items=['State', 'Facility ID', 'Facility Name', 'Unit ID', 'Year', 'Month',
                       'Heat Input (mmBtu)', 'Primary Fuel Type', 'Unit Type'])
        .query('State not in ["VI", "MP", "GU", "AS", "PR", "AK", "HI"]')
    )

    ARP_Raw = (
        ARP_Raw.assign(
            unit_clean=np.select(
                [
                    ARP_Raw['Unit Type'].str.contains('combustion turbine', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('combined cycle', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('wet bottom', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('dry bottom', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('bubbling', case=False, na=False)
                ],
                [
                    'combustion turbine',
                    'combined cycle',
                    'wet bottom',
                    'dry bottom',
                    'bubbling'
                ],
                default=ARP_Raw['Unit Type'].str.lower()
            ),
            fuel_clean=np.select(
                [
                    ARP_Raw['Primary Fuel Type'].isin(['Pipeline Natural Gas',
                                                       'Natural Gas',
                                                       'Other Gas',
                                                       'Other Gas, Pipeline Natural Gas',
                                                       'Natural Gas, Pipeline Natural Gas',
                                                       'Process Gas']),
                    ARP_Raw['Primary Fuel Type'].isin(['Petroleum Coke',
                                                       'Coal',
                                                       'Coal, Pipeline Natural Gas',
                                                       'Coal, Natural Gas',
                                                       'Coal, Wood',
                                                       'Coal, Coal Refuse',
                                                       'Coal Refuse',
                                                       'Coal, Process Gas',
                                                       'Coal, Diesel Oil',
                                                       'Other Solid Fuel']),
                    ARP_Raw['Primary Fuel Type'].isin(['Other Oil',
                                                       'Diesel Oil',
                                                       'Diesel Oil, Residual Oil',
                                                       'Diesel Oil, Pipeline Natural Gas',
                                                       'Residual Oil',
                                                       'Residual Oil, Pipeline Natural Gas']),
                    ARP_Raw['Primary Fuel Type'].isin(['Wood',
                                                       'Other Solid Fuel, Wood'])
                ],
                [
                    'Gas',
                    'Coal',
                    'Oil',
                    'Wood'
                ],
                default=ARP_Raw['Primary Fuel Type']
            )
        )
        .query('fuel_clean == "Gas"')
    )

    # Coerce Heat Input (mmBtu)
    ARP_Raw['Heat Input (mmBtu)'] = pd.to_numeric(ARP_Raw['Heat Input (mmBtu)'], errors='coerce').fillna(0)

    # Calculate CH4 flux for each facility
    ARP_Raw = (
        ARP_Raw.assign(
            ch4_f=np.select(
                [
                    # For Gas
                    (ARP_Raw['fuel_clean'] == 'Gas') &
                    (ARP_Raw['unit_clean'] == 'combined cycle') |
                    (ARP_Raw['unit_clean'].str.contains('turbine', case=False, na=False)),

                    (ARP_Raw['fuel_clean'] == 'Gas')
                ],
                [
                    3.7,    # Gas: combined cycle or turbine
                    1,     # Gas: others
                ],
                default=np.nan)
        )
        )

    # Calculate CH4 flux for each facility by multiplying heat input by CH4 factor
    ARP_Clean = (
        ARP_Raw.assign(
            ch4_flux=ARP_Raw['Heat Input (mmBtu)'] * ARP_Raw['ch4_f']
            )
        .drop(columns=['Unit ID', 'Heat Input (mmBtu)', 'Primary Fuel Type', 'Unit Type',
                       'unit_clean', 'ch4_f', 'fuel_clean'])
        .rename(columns={'Facility ID': 'facility_id',
                         'Facility Name': 'facility_name',
                         'State': 'state_code',
                         'Year': 'year',
                         'Month': 'month'})
        .groupby(['facility_id', 'facility_name', 'state_code', 'year', 'month'])
        .sum('ch4_flux')
        .sort_values(['facility_id', 'year', 'month'])
        .reset_index()
        )

    # Merge with ARP Facility Data
    ARP_Full = (
        ARP_Clean.merge(ARP_Facility_gdf[['facility_id', 'geometry']],
                        on='facility_id', how='left')
    )

    ARP_Full = gpd.GeoDataFrame(ARP_Full, geometry='geometry', crs=4326)

    ARP_Full.to_parquet(reporting_electric_gas_proxy_output_path)
    return None

########################################################################################
# %% elec_oil_proxy

@mark.persist
@task(id='elec_oil_proxy')
def task_get_reporting_electric_oil_proxy_data(
    facility_path=EPA_ARP_facilities,
    input_path=EPA_ARP_inputfile,
    reporting_electric_oil_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path 
    / 'elec_oil_proxy.parquet'
):
    """
    Relative emissions and location information for reporting facilities are taken from
    the Clean Air Markets Program Data (CAMPD) and the Acid Rain Program (ARP) Data.
    """

    # Read in ARP Facility Data
    ARP_Facility = (
        pd.read_csv(facility_path, index_col=False)
        .filter(items=['Facility ID', 'Facility Name', 'State', 'Latitude', 'Longitude'])
        .drop_duplicates(['Facility ID', 'Latitude', 'Longitude'], keep='last')
        .rename(columns={'Facility ID': 'facility_id'})
        .reset_index(drop=True)
    )

    ARP_Facility_gdf = (
        gpd.GeoDataFrame(
            ARP_Facility,
            geometry=gpd.points_from_xy(
                ARP_Facility['Longitude'],
                ARP_Facility['Latitude'],
                crs=4326
            )
        )
    )

    # Read in Raw ARP Data & Clean Fuel Type
    ARP_Raw = (
        pd.read_csv(input_path, index_col=False)
        .filter(items=['State', 'Facility ID', 'Facility Name', 'Unit ID', 'Year', 'Month',
                       'Heat Input (mmBtu)', 'Primary Fuel Type', 'Unit Type'])
        .query('State not in ["VI", "MP", "GU", "AS", "PR", "AK", "HI"]')
    )

    ARP_Raw = (
        ARP_Raw.assign(
            unit_clean=np.select(
                [
                    ARP_Raw['Unit Type'].str.contains('combustion turbine', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('combined cycle', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('wet bottom', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('dry bottom', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('bubbling', case=False, na=False)
                ],
                [
                    'combustion turbine',
                    'combined cycle',
                    'wet bottom',
                    'dry bottom',
                    'bubbling'
                ],
                default=ARP_Raw['Unit Type'].str.lower()
            ),
            fuel_clean=np.select(
                [
                    ARP_Raw['Primary Fuel Type'].isin(['Pipeline Natural Gas',
                                                       'Natural Gas',
                                                       'Other Gas',
                                                       'Other Gas, Pipeline Natural Gas',
                                                       'Natural Gas, Pipeline Natural Gas',
                                                       'Process Gas']),
                    ARP_Raw['Primary Fuel Type'].isin(['Petroleum Coke',
                                                       'Coal',
                                                       'Coal, Pipeline Natural Gas',
                                                       'Coal, Natural Gas',
                                                       'Coal, Wood',
                                                       'Coal, Coal Refuse',
                                                       'Coal Refuse',
                                                       'Coal, Process Gas',
                                                       'Coal, Diesel Oil',
                                                       'Other Solid Fuel']),
                    ARP_Raw['Primary Fuel Type'].isin(['Other Oil',
                                                       'Diesel Oil',
                                                       'Diesel Oil, Residual Oil',
                                                       'Diesel Oil, Pipeline Natural Gas',
                                                       'Residual Oil',
                                                       'Residual Oil, Pipeline Natural Gas']),
                    ARP_Raw['Primary Fuel Type'].isin(['Wood',
                                                       'Other Solid Fuel, Wood'])
                ],
                [
                    'Gas',
                    'Coal',
                    'Oil',
                    'Wood'
                ],
                default=ARP_Raw['Primary Fuel Type']
            )
        )
        .query('fuel_clean == "Oil"')
    )

    # Coerce Heat Input (mmBtu)
    ARP_Raw['Heat Input (mmBtu)'] = pd.to_numeric(ARP_Raw['Heat Input (mmBtu)'], errors='coerce').fillna(0)

    # Calculate CH4 flux for each facility
    ARP_Raw = (
        ARP_Raw.assign(
            ch4_f=np.select(
                [
                    # For Oil
                    (ARP_Raw['fuel_clean'] == 'Oil') &
                    (ARP_Raw['Primary Fuel Type'].isin(['Residual Oil', 'Residual Oil, Pipeline Natural Gas'])),

                    (ARP_Raw['fuel_clean'] == 'Oil')
                ],
                [
                    0.8,    # Oil: Residual Oil or Pipeline Natural Gas
                    0.9,    # Oil: others
                ],
                default=np.nan
            )
            )
    )

    # Calculate CH4 flux for each facility by multiplying heat input by CH4 factor
    ARP_Clean = (
        ARP_Raw.assign(
            ch4_flux=ARP_Raw['Heat Input (mmBtu)'] * ARP_Raw['ch4_f']
            )
        .drop(columns=['Unit ID', 'Heat Input (mmBtu)', 'Primary Fuel Type', 'Unit Type',
                       'unit_clean', 'ch4_f', 'fuel_clean'])
        .rename(columns={'Facility ID': 'facility_id',
                         'Facility Name': 'facility_name',
                         'State': 'state_code',
                         'Year': 'year',
                         'Month': 'month'})
        .groupby(['facility_id', 'facility_name', 'state_code', 'year', 'month'])
        .sum('ch4_flux')
        .sort_values(['facility_id', 'year', 'month'])
        .reset_index()
        )

    # Merge with ARP Facility Data
    ARP_Full = (
        ARP_Clean.merge(ARP_Facility_gdf[['facility_id', 'geometry']],
                        on='facility_id', how='left')
    )

    ARP_Full = gpd.GeoDataFrame(ARP_Full, geometry='geometry', crs=4326)

    ARP_Full.to_parquet(reporting_electric_oil_proxy_output_path)
    return None


########################################################################################
# %% elec_wood_proxy

@mark.persist
@task(id='elec_wood_proxy')
def task_get_reporting_electric_wood_proxy_data(
    facility_path=EPA_ARP_facilities,
    input_path=EPA_ARP_inputfile,
    reporting_electric_wood_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path 
    / 'elec_wood_proxy.parquet'
):
    """
    Relative emissions and location information for reporting facilities are taken from
    the Clean Air Markets Program Data (CAMPD) and the Acid Rain Program (ARP) Data.
    """

    # Read in ARP Facility Data
    ARP_Facility = (
        pd.read_csv(facility_path, index_col=False)
        .filter(items=['Facility ID', 'Facility Name', 'State', 'Latitude', 'Longitude'])
        .drop_duplicates(['Facility ID', 'Latitude', 'Longitude'], keep='last')
        .rename(columns={'Facility ID': 'facility_id'})
        .reset_index(drop=True)
    )

    ARP_Facility_gdf = (
        gpd.GeoDataFrame(
            ARP_Facility,
            geometry=gpd.points_from_xy(
                ARP_Facility['Longitude'],
                ARP_Facility['Latitude'],
                crs=4326
            )
        )
    )

    # Read in Raw ARP Data & Clean Fuel Type
    ARP_Raw = (
        pd.read_csv(input_path, index_col=False)
        .filter(items=['State', 'Facility ID', 'Facility Name', 'Unit ID', 'Year', 'Month',
                       'Heat Input (mmBtu)', 'Primary Fuel Type', 'Unit Type'])
        .query('State not in ["VI", "MP", "GU", "AS", "PR", "AK", "HI"]')
    )

    ARP_Raw = (
        ARP_Raw.assign(
            unit_clean=np.select(
                [
                    ARP_Raw['Unit Type'].str.contains('combustion turbine', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('combined cycle', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('wet bottom', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('dry bottom', case=False, na=False),
                    ARP_Raw['Unit Type'].str.contains('bubbling', case=False, na=False)
                ],
                [
                    'combustion turbine',
                    'combined cycle',
                    'wet bottom',
                    'dry bottom',
                    'bubbling'
                ],
                default=ARP_Raw['Unit Type'].str.lower()
            ),
            fuel_clean=np.select(
                [
                    ARP_Raw['Primary Fuel Type'].isin(['Pipeline Natural Gas',
                                                       'Natural Gas',
                                                       'Other Gas',
                                                       'Other Gas, Pipeline Natural Gas',
                                                       'Natural Gas, Pipeline Natural Gas',
                                                       'Process Gas']),
                    ARP_Raw['Primary Fuel Type'].isin(['Petroleum Coke',
                                                       'Coal',
                                                       'Coal, Pipeline Natural Gas',
                                                       'Coal, Natural Gas',
                                                       'Coal, Wood',
                                                       'Coal, Coal Refuse',
                                                       'Coal Refuse',
                                                       'Coal, Process Gas',
                                                       'Coal, Diesel Oil',
                                                       'Other Solid Fuel']),
                    ARP_Raw['Primary Fuel Type'].isin(['Other Oil',
                                                       'Diesel Oil',
                                                       'Diesel Oil, Residual Oil',
                                                       'Diesel Oil, Pipeline Natural Gas',
                                                       'Residual Oil',
                                                       'Residual Oil, Pipeline Natural Gas']),
                    ARP_Raw['Primary Fuel Type'].isin(['Wood',
                                                       'Other Solid Fuel, Wood'])
                ],
                [
                    'Gas',
                    'Coal',
                    'Oil',
                    'Wood'
                ],
                default=ARP_Raw['Primary Fuel Type']
            )
        )
        .query('fuel_clean == "Wood"')
    )

    # Coerce Heat Input (mmBtu)
    ARP_Raw['Heat Input (mmBtu)'] = pd.to_numeric(ARP_Raw['Heat Input (mmBtu)'], errors='coerce').fillna(0)

    # Calculate CH4 flux for each facility
    ARP_Raw = (
        ARP_Raw.assign(
            ch4_f=np.select(
                [
                    # For Wood
                    (ARP_Raw['fuel_clean'] == 'Wood'),
                ],
                [
                    1,      # Wood: recover boilers
                ],
                default=np.nan
            )
            )
    )

    # Calculate CH4 flux for each facility by multiplying heat input by CH4 factor
    ARP_Clean = (
        ARP_Raw.assign(
            ch4_flux=ARP_Raw['Heat Input (mmBtu)'] * ARP_Raw['ch4_f']
            )
        .drop(columns=['Unit ID', 'Heat Input (mmBtu)', 'Primary Fuel Type', 'Unit Type',
                       'unit_clean', 'ch4_f', 'fuel_clean'])
        .rename(columns={'Facility ID': 'facility_id',
                         'Facility Name': 'facility_name',
                         'State': 'state_code',
                         'Year': 'year',
                         'Month': 'month'})
        .groupby(['facility_id', 'facility_name', 'state_code', 'year', 'month'])
        .sum('ch4_flux')
        .sort_values(['facility_id', 'year', 'month'])
        .reset_index()
        )

    # Merge with ARP Facility Data
    ARP_Full = (
        ARP_Clean.merge(ARP_Facility_gdf[['facility_id', 'geometry']],
                        on='facility_id', how='left')
    )

    ARP_Full = gpd.GeoDataFrame(ARP_Full, geometry='geometry', crs=4326)

    ARP_Full.to_parquet(reporting_electric_wood_proxy_output_path)
    return None


########################################################################################
# %% indu_proxy

@mark.persist
@task(id='indu_proxy')
def task_get_reporting_indu_proxy_data(
    subpart_C=GHGRP_subC_inputfile,
    subpart_D=GHGRP_subD_inputfile,
    facility_path=GHGRP_subDfacility_loc_inputfile,
    reporting_indu_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path
    / 'indu_proxy.parquet'
):
    """
    Relative emissions and location information for reporting facilities are taken from
    the enviro.epa.gov/query-builder Data.
    """

    # Read in Subpart C
    GHGRP_C = (
        pd.read_csv(GHGRP_subC_inputfile, index_col=False)
        .query('ghg_gas_name == "Methane"')
        .query('reporting_year >= @min_year & reporting_year <= @max_year')
        .drop(columns='ghg_gas_name')
        .reset_index(drop=True)
    )
    # Read in Subpart D
    GHGRP_D = (
        pd.read_csv(GHGRP_subD_inputfile, index_col=False)
        .query('ghg_name == "Methane"')
        .drop_duplicates(subset=['facility_id'])
        .reset_index(drop=True)
    )

    # Get facilities that are in subpart C, but not in subpart D
    # Identify unmatched rows (Anti-join)
    GHGRP_C_Only = (
        GHGRP_C.merge(
            GHGRP_D[['facility_id']],
            on=['facility_id'],
            how='left',
            indicator=True
        )
        .query('_merge == "left_only"')
        .drop(columns='_merge')
    )

    # Read in Facility List
    # Keep most recent unique facility info (no year)
    GHGRP_Facilities = (
        pd.read_csv(GHGRP_subDfacility_loc_inputfile, index_col=False)
        .sort_values(by=['facility_id', 'year'], ascending=[True, False])
        .drop_duplicates(subset=['facility_id'], keep='first')
        .drop(columns='year')
        .reset_index(drop=True)
    )

    # Merge C_Only with Facility Info
    indu_proxy = (
        GHGRP_C_Only.merge(
            GHGRP_Facilities,
            on='facility_id'
        )
        .sort_values(by=['facility_id', 'reporting_year'])
        .query('state not in ["VI", "MP", "GU", "AS", "PR", "AK", "HI"]')
        .rename(columns={
            'state': 'state_code',
            'reporting_year': 'year'
        }
        )
        # Convert Metrtic Tons to KT (1 KT = 1000 Metric Tons)
        .assign(
            rel_emi=lambda df: df["ghg_quantity"] / 1000
        )
        [['facility_id', 'facility_name', 'state_code', 'year', 'rel_emi', 'latitude', 'longitude']]
        .reset_index(drop=True)
    )

    indu_proxy = (
        gpd.GeoDataFrame(
            indu_proxy,
            geometry=gpd.points_from_xy(
                indu_proxy['longitude'],
                indu_proxy['latitude'],
                crs=4326
            )
        )
        .drop(columns=['latitude', 'longitude'])
    )

    indu_proxy.to_parquet(reporting_indu_proxy_output_path)
    return None
