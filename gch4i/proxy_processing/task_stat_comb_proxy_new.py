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
    global_data_dir_path,
    emi_data_dir_path,
    sector_data_dir_path,
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

# State
state_path: Path = global_data_dir_path / "tl_2020_us_state.zip"

# Stationary Combustion Emission Files
elec_coal_emi_path: Path = emi_data_dir_path / "stat_comb_elec_coal_emi.csv"
elec_gas_emi_path: Path = emi_data_dir_path / "stat_comb_elec_gas_emi.csv"
elec_oil_emi_path: Path = emi_data_dir_path / "stat_comb_elec_oil_emi.csv"
elec_wood_emi_path: Path = emi_data_dir_path / "stat_comb_elec_wood_emi.csv"

# EIA 923 Data (data source for electricity generation proxies in v3)
EIA_923_path: Path = sector_data_dir_path / "eia/EIA-923"
EIA_923_plant_locs_path: Path = sector_data_dir_path / "eia/EIA-923/Power_Plants.csv"

# Oregon power plant location for electricity generation proxies
# https://ghgdata.epa.gov/ghgp/service/facilityDetail/2012?id=1007940&ds=E&et=FC_CL&popup=true
#       Latitude: 45.6981567
#       Longitude: -119.7986057

# ARP Data (data source for electricity generation proxies in v2). Work with Vince
# to determine which data source makes the most sense for the v4 proxy.
# EPA_ARP_facilities = GEPA_Stat_Path / "InputData" / "ARP_Data" / "EPA_ARP_2012-2022_Facility_Info.csv"
# EPA_ARP_inputfile = GEPA_Stat_Path / "InputData/ARP_Data/EPA_ARP_2012-2022.csv"

# GHGRP Data (reporting format changed in 2015)
GHGRP_subC_inputfile = GEPA_Stat_Path / "InputData/GHGRP/GHGRP_SubpartCEmissions_2010-2023.csv" #subpart C facility IDs and emissions (locations not available)
GHGRP_subD_inputfile = GEPA_Stat_Path / "InputData/GHGRP/GHGRP_SubpartDEmissions_2010-2023.csv" #subpart D facility IDs and emissions 
GHGRP_subDfacility_loc_inputfile = GEPA_Stat_Path / "InputData/GHGRP/GHGRP_FacilityInfo_2010-2023.csv" #subpart D facility info (for all years, with ID & lat and lons)

########################################################################################
# %% elec_coal_proxy, elec_gas_proxy, elec_oil_proxy, elec_wood_proxy


@mark.persist
@task(id='elec_proxies')
def task_get_electricity_generation_proxy_data(
    state_path=state_path,
    facility_path=EIA_923_plant_locs_path,
    input_path=EIA_923_path,
    elec_coal_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / 'elec_coal_proxy.parquet',
    elec_gas_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / 'elec_gas_proxy.parquet',
    elec_oil_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / 'elec_oil_proxy.parquet',
    elec_wood_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / 'elec_wood_proxy.parquet',
):
    """
    Relative emissions and location information for reporting facilities are taken from
    the EIA Survey 923 Data. In v2, the ARP data was used
    """

    # Read in state geometries of lower 48 and DC
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

    # Read in EIA power plant location data and drop facilities with no lat/lon
    eia_facility_locs = (
        pd.read_csv(facility_path, index_col=False)
        .filter(items=['Plant_Code', 'Plant_Name', 'Utility_ID', 'Utility_Name', 
                       'State', 'tech_desc', 'Longitude', 'Latitude'])
        .rename(columns={'Plant_Code': 'plant_id'})
        .rename(columns=str.lower)
    )

    # Read in EIA-923 data
    eia_923_plants = pd.DataFrame()
    for iyear in years:
        eia_923_file_name = f"EIA923_Schedules_2_3_4_5_M_12_{iyear}_Final_Revision.xlsx"
        eia_923_file_path = os.path.join(input_path, eia_923_file_name)
        data_iyear = (
            pd.read_excel(
                eia_923_file_path,
                sheet_name='Page 1 Generation and Fuel Data',
                # Columns: Plant Id, Plant Name, Operator Name, Operator Id, Plant State
                # Reported Fuel Type Code, AER/MER Fuel Type Code,
                # and Elec_MMBTu {Month} columns under Quantity Consumed For Electricity (MMBtu)											
                usecols="A,D,E,F,G,O,P,BP:CA",
                skiprows=5,)
            .assign(year=iyear))
        data_iyear.columns = ['plant_id', 'plant_name', 'operator_name', 'operator_id',
                              'state_code', 'reported_fuel_type', 'aer_fuel_type',
                              'fuel_qty_01', 'fuel_qty_02', 'fuel_qty_03',
                              'fuel_qty_04', 'fuel_qty_05', 'fuel_qty_06',
                              'fuel_qty_07', 'fuel_qty_08', 'fuel_qty_09',
                              'fuel_qty_10', 'fuel_qty_11', 'fuel_qty_12', 'year']
        eia_923_plants = pd.concat([eia_923_plants, data_iyear]).reset_index(drop=True)

    # Create a power plants df with locations and fuel consumption
    power_plants_df = pd.merge(eia_923_plants, eia_facility_locs, on='plant_id', how='left').dropna(subset=["latitude", "longitude"]).reset_index(drop=True)

    # Assign CH4 factor based on tech description and four main fuel categories based on fuel type
    power_plants_df = (power_plants_df
                       .assign(ch4_f=np.select(
                           [
                               power_plants_df['tech_desc'].str.contains('Natural Gas Fired Combined Cycle', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Natural Gas Fired Combustion Turbine', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Natural Gas Internal Combustion Engine', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Natural Gas Steam Turbine', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Natural Gas with Compressed Air Storage', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Other Natural Gas', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Other Gases', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Coal Integrated Gasification Combined Cycle', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Conventional Steam Coal', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Petroleum Coke', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Petroleum Liquids', case=False, na=False),
                               power_plants_df['tech_desc'].str.contains('Wood/Wood Waste Biomass', case=False, na=False),
                               ],
                           [
                               3.7,  # natural gas combined cycle or turbine
                               3.7,  # natural gas combined cycle or turbine
                               1.0,  # natural gas boilers
                               3.7,  # natural gas combined cycle or turbine
                               1.0,  # natural gas boilers
                               1.0,  # natural gas boilers
                               1.0,  # natural gas boilers
                               0.7,  # weighted average of ARP CH4 factors from v2 GEPA
                               0.7,  # weighted average of ARP CH4 factors from v2 GEPA
                               0.7,  # weighted average of ARP CH4 factors from v2 GEPA
                               0.85,  # oil
                               1.0,  # wood
                           ],
                           default=np.nan))
                       .assign(fuel=np.select(
                           [
                               power_plants_df['aer_fuel_type'].isin(['COL', 'PC', 'WOC']),
                               power_plants_df['aer_fuel_type'].isin(['NG', 'OOG']),
                               power_plants_df['aer_fuel_type'].isin(['DFO', 'RFO', 'WOO']),
                               power_plants_df['aer_fuel_type'].isin(['WWW']),
                               ],
                           [
                               'Coal',
                               'Gas',
                               'Oil',
                               'Wood'
                           ],
                           default=power_plants_df['aer_fuel_type']))
                       .drop(columns=['plant_name_x', 'operator_name', 'operator_id',
                                      'reported_fuel_type', 'aer_fuel_type',
                                      'plant_name_y', 'utility_id', 'utility_name',
                                      'state'])
                       )

    # Clean up data where fuel consumption is "." assign it to 0
    power_plants_df.loc[:, "fuel_qty_01":"fuel_qty_12"] = power_plants_df.loc[:, "fuel_qty_01":"fuel_qty_12"].replace(".", 0)

    # Melt monthly data into a single column and add year_month and month columns
    # Columns we want to melt
    value_vars = ['fuel_qty_01', 'fuel_qty_02', 'fuel_qty_03', 'fuel_qty_04', 'fuel_qty_05', 'fuel_qty_06', 'fuel_qty_07', 'fuel_qty_08', 'fuel_qty_09', 'fuel_qty_10', 'fuel_qty_11', 'fuel_qty_12']
    # Columns we want to maintain in melted df
    id_vars = ['plant_id', 'state_code', 'year', 'latitude', 'longitude', 'fuel', 'tech_desc', 'ch4_f']
    power_plants_df = pd.melt(power_plants_df,
                              id_vars=id_vars,
                              value_vars=value_vars,
                              var_name='fuel_month',
                              value_name='fuel_qty')
    power_plants_df = (power_plants_df
                       .assign(month=lambda df: df['fuel_month'][-2:])
                       .assign(month=power_plants_df.fuel_month.astype(str).str[-2:].astype(int))
                       .assign(year_month=power_plants_df.year.astype(str)+'_'+power_plants_df.fuel_month.astype(str).str[-2:])
                       .query("fuel_qty > 0")
                       .drop(columns="fuel_month")
                       .reset_index(drop=True)
                       )

    # Coerce consumption to be numeric (mmBtu)
    power_plants_df['fuel_qty'] = pd.to_numeric(power_plants_df['fuel_qty'], errors='coerce').fillna(0)

    # Calculate CH4 flux for each facility by multiplying fuel consumption by CH4 factor
    power_plants_df = (power_plants_df
                       .assign(ch4_flux=power_plants_df['fuel_qty'] * power_plants_df['ch4_f'])
                       .drop(columns=['tech_desc', 'ch4_f', 'fuel_qty'])
                       .sort_values(['plant_id', 'year', 'month'])
                       .query("state_code.isin(@state_gdf['state_code'])")
                       .reset_index(drop=True)
                       )

    # Create individual fuel proxies
    coal_proxy = power_plants_df.query("fuel == 'Coal'").reset_index(drop=True)
    gas_proxy = power_plants_df.query("fuel == 'Gas'").reset_index(drop=True)
    oil_proxy = power_plants_df.query("fuel == 'Oil'").reset_index(drop=True)
    wood_proxy = power_plants_df.query("fuel == 'Wood'").reset_index(drop=True)

    # Manually add Oregon coal proxy data - one location and assume emissions are
    # uniformly distributed across the year (rel_emi = 1/12 for each year)
    # Oregon has state-level emissions for 2012-2020 (emissions = 0 for 2021, 2022)
    OR_coal_proxy_imonth = pd.DataFrame({'plant_id': [np.nan], 'state_code': ['OR'], 'latitude': [45.6981567], 'longitude': [-119.7986057], 'fuel': ['Coal'], 'ch4_flux': [1.0]})

    OR_coal_proxy_iyear = pd.DataFrame()
    for imonth in range(1, 13):
        data_temp = OR_coal_proxy_imonth.copy()
        data_temp['month'] = imonth
        data_temp['month_str'] = f"{imonth:02}"  # convert to 2-digit months
        OR_coal_proxy_iyear = pd.concat([OR_coal_proxy_iyear, data_temp]).reset_index(drop=True)

    OR_coal_proxy = pd.DataFrame()
    for iyear in years:
        data_temp = OR_coal_proxy_iyear.copy()
        data_temp['year'] = iyear
        OR_coal_proxy = pd.concat([OR_coal_proxy, data_temp]).reset_index(drop=True)

    OR_coal_proxy = OR_coal_proxy.assign(year_month=OR_coal_proxy.year.astype(str)+'_'+OR_coal_proxy.month_str).drop(columns={"month_str"})

    coal_proxy = pd.concat([coal_proxy, OR_coal_proxy]).reset_index(drop=True)

    # Function to calculate relative emissions for monthly proxy data
    def calc_rel_emi(df):
        # sum of the annual_rel_emi = 1 for each state_code-year combination
        # used to allocate annual emissions to monthly emissions
        df['annual_rel_emi'] = (
            df.groupby(["state_code", "year"])['ch4_flux']
            .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
        )
        # sum of the rel_emi = 1 for each state_code-year_month combination
        # used to allocate monthly emissions to monthly proxy
        df['rel_emi'] = (
            df.groupby(["state_code", "year_month"])['ch4_flux']
            .transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
        )
        df = df.drop(columns='ch4_flux')
        return df

    # Function to format proxy data into geodataframes
    def proxy_df_to_gdf(df):
        gdf = (
            gpd.GeoDataFrame(
                df,
                geometry=gpd.points_from_xy(
                    df["longitude"],
                    df["latitude"],
                    crs=4326
                )
            )
            .drop(columns=["latitude", "longitude"])
        )
        return gdf

    # Create annual and monthly relative emissions and geodataframes
    coal_proxy = proxy_df_to_gdf(calc_rel_emi(coal_proxy))
    gas_proxy = proxy_df_to_gdf(calc_rel_emi(gas_proxy))
    oil_proxy = proxy_df_to_gdf(calc_rel_emi(oil_proxy))
    wood_proxy = proxy_df_to_gdf(calc_rel_emi(wood_proxy))

    # Process to correct for missing proxy data
    # 1. Find missing state_code-year pairs
    # 2. Check to see if proxy data exists for state in another year
    #   2a. If the data exists, use proxy data from the closest year
    #   2b. If the data does not exist, assign emissions uniformly across the state

    # Function to get emissions to grid
    def get_emi_file(emi_path, state_gdf):
        emi_df = (pd.read_csv(emi_path)
                  .query("state_code.isin(@state_gdf['state_code'])")
                  .query("ghgi_ch4_kt > 0.0"))
        return emi_df

    # Function to find missing proxy state-year combinations
    def get_missing_proxy_states(emi_df, proxy_df):
        emi_states = set(emi_df[['state_code', 'year']].itertuples(index=False, name=None))
        proxy_states = set(proxy_df[['state_code', 'year']].itertuples(index=False, name=None))
        missing_states = emi_states.difference(proxy_states)
        return missing_states

    # Function to find the closest year (for step 2a approach)
    # arr is the array of all years with data and target is the year missing data
    def find_closest_year(arr, target):
        arr = np.array(arr)
        idx = (np.abs(arr - target)).argmin()
        return arr[idx]

    # Function to correct for missing proxy data
    def create_alt_proxy(missing_states, original_proxy_df):
        # Add missing states alternative data to grouped_proxy
        if missing_states:
            alt_proxy = gpd.GeoDataFrame()
            # List of states with proxy data in any year
            proxy_unique_states = original_proxy_df['state_code'].unique()
            for istate_year in np.arange(0, len(missing_states)):
                # Missing state
                istate = str(list(missing_states)[istate_year])[2:4]
                # Missing year
                iyear = int(str(list(missing_states)[istate_year])[7:11])
                # If the missing state code-year pair has data for another year, assign
                # the proxy data for the next available previous year
                if istate in proxy_unique_states:
                    # Get proxy data for the state for all years
                    iproxy = (original_proxy_df
                              .query("state_code == @istate")
                              .reset_index(drop=True)
                              )
                    # Get years that have proxy data
                    iproxy_unique_years = iproxy['year'].unique()
                    # Find the closest year to the missing proxy year
                    iyear_closest = find_closest_year(iproxy_unique_years, iyear)
                    # Assign proxy data of the closest year to the missing proxy year
                    iproxy = (iproxy
                              .query("year == @iyear_closest")
                              .assign(year=iyear)
                              .reset_index(drop=True)
                              )
                    # Update year_month column to be the correct year
                    for ifacility in np.arange(0, len(iproxy)):
                        imonth_str = str(iproxy['year_month'][ifacility][5:8])
                        iyear_month_str = str(iyear)+'-'+imonth_str
                        iproxy.loc[ifacility, 'year_month'] = iyear_month_str
                        iproxy.loc[ifacility, 'month'] = int(imonth_str)
                    alt_proxy = gpd.GeoDataFrame(pd.concat([alt_proxy, iproxy], ignore_index=True))
                else:
                    # Create alternative proxy from missing states
                    iproxy = gpd.GeoDataFrame([list(missing_states)[istate_year]])
                    iproxy.columns = ['state_code', 'year']
                    iproxy['annual_rel_emi'] = 1/12  # Assign emissions evenly across the state for a given year
                    iproxy['rel_emi'] = 1.0  # Assign emissions evenly across the state for a given month in the year
                    iproxy = iproxy.merge(
                        state_gdf[['state_code', 'geometry']],
                        on='state_code',
                        how='left')
                    for imonth in range(1, 13):
                        imonth_str = f"{imonth:02}"  # convert to 2-digit months
                        year_month_str = str(iyear)+'-'+imonth_str
                        imonth_proxy = iproxy.copy().assign(year_month=year_month_str).assign(month=imonth)
                        alt_proxy = gpd.GeoDataFrame(pd.concat([alt_proxy, imonth_proxy], ignore_index=True))
            # Add missing proxy to original proxy
            proxy_gdf_final = gpd.GeoDataFrame(pd.concat([original_proxy_df, alt_proxy], ignore_index=True).reset_index(drop=True))
            # Delete unused temp data
            del original_proxy_df
            del alt_proxy
        else:
            proxy_gdf_final = original_proxy_df.copy()
            # Delete unused temp data
            del original_proxy_df

        # Check that annual relative emissions sum to 1.0 each state/year combination
        sums_annual = proxy_gdf_final.groupby(["state_code", "year"])["annual_rel_emi"].sum()  # get sums to check normalization
        assert np.isclose(sums_annual, 1.0, atol=1e-8).all(), f"Annual relative emissions do not sum to 1 for each year and state; {sums_annual}"  # assert that the sums are close to 1

        # Check that monthly relative emissions sum to 1.0 each state/year_month combination
        sums_monthly = proxy_gdf_final.groupby(["state_code", "year_month"])["rel_emi"].sum()  # get sums to check normalization
        assert np.isclose(sums_monthly, 1.0, atol=1e-8).all(), f"Monthly relative emissions do not sum to 1 for each year_month and state; {sums_monthly}"  # assert that the sums are close to 1

        proxy_gdf_final = proxy_gdf_final.reset_index(drop=True)

        return proxy_gdf_final

    # Get missing state-year pairs for each proxy
    missing_states_coal = get_missing_proxy_states(get_emi_file(elec_coal_emi_path, state_gdf), coal_proxy)
    missing_states_gas = get_missing_proxy_states(get_emi_file(elec_gas_emi_path, state_gdf), gas_proxy)
    missing_states_oil = get_missing_proxy_states(get_emi_file(elec_oil_emi_path, state_gdf), oil_proxy)
    missing_states_wood = get_missing_proxy_states(get_emi_file(elec_wood_emi_path, state_gdf), wood_proxy)

    # Correct missing proxy data for coal, gas, oil, and wood proxies
    coal_proxy_final = create_alt_proxy(missing_states_coal, coal_proxy)
    gas_proxy_final = create_alt_proxy(missing_states_gas, gas_proxy)
    oil_proxy_final = create_alt_proxy(missing_states_oil, oil_proxy)
    wood_proxy_final = create_alt_proxy(missing_states_wood, wood_proxy)

    # Re-check for missing states
    missing_states_coal_final = get_missing_proxy_states(get_emi_file(elec_coal_emi_path, state_gdf), coal_proxy_final)
    missing_states_gas_final = get_missing_proxy_states(get_emi_file(elec_gas_emi_path, state_gdf), gas_proxy_final)
    missing_states_oil_final = get_missing_proxy_states(get_emi_file(elec_oil_emi_path, state_gdf), oil_proxy_final)
    missing_states_wood_final = get_missing_proxy_states(get_emi_file(elec_wood_emi_path, state_gdf), wood_proxy_final)

    # Output Proxy Parquet Files
    coal_proxy_final.to_parquet(elec_coal_proxy_output_path)
    gas_proxy_final.to_parquet(elec_gas_proxy_output_path)
    oil_proxy_final.to_parquet(elec_oil_proxy_output_path)
    wood_proxy_final.to_parquet(elec_wood_proxy_output_path)

    return None


########################################################################################
# %% indu_proxy

@mark.persist
@task(id='indu_proxy')
def task_get_reporting_indu_proxy_data(
    subpart_C=GHGRP_subC_inputfile,
    subpart_D=GHGRP_subD_inputfile,
    facility_path=GHGRP_subDfacility_loc_inputfile,
    reporting_indu_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / 'indu_proxy.parquet'
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
