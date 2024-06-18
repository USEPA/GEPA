"""
Name:               5A1_msw_landfills.py
Authors Name:       N. Kruskamp, H. Lohman (RTI International), Erin McDuffie (EPA/OAP)
Date Last Modified: 06/13/2024
Purpose:            Spatially allocates methane emissions for source category 5A1
                    municipal solid waste landfills.
Input Files:
                    - State_MSW_LF_1990-2021.xlsx
                    - Non-Reporting_LF_DB_2020_1.12.2021.xlsx
Output Files:
                    - {FULL_NAME}_ch4_kt_per_year.tif
                    - {FULL_NAME}_ch4_emi_flux.tif
Notes:
TODO: get latest inventory data, get latest list of non-reporting facilities
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------

# for testing/development
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import osgeo  # noqa
import duckdb
import geopandas as gpd
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from shapely import Point, wkb
import geopy
from geopy.geocoders import Nominatim

from gch4i.config import (
    V3_DATA_PATH,
    ghgi_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.utils import (
    QC_emi_raster_sums,
    QC_point_proxy_allocation,
    grid_point_emissions,
    name_formatter,
    plot_annual_raster_data,
    plot_raster_data_difference,
    state_year_point_allocate_emis,
    tg_to_kt,
    write_ncdf_output,
    write_tif_output,
)

t_to_kt = 0.001
year_range = [*range(min_year, 2021+1,1)] #List of emission years
# year_range = [*range(min_year, max_year+1,1)] #List of emission years
year_range_str=[str(i) for i in year_range]
num_years = len(year_range)

# from pytask import Product, task

gpd.options.io_engine = "pyogrio"

def get_reporting_msw_landfills_inventory_data(input_path):
    reporting_emi_df = (
        pd.read_excel(
            input_path,
            sheet_name="InvDB",
            skiprows=15,
            nrows=60,
            usecols="A:AO",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # drop columns we don't need
        .drop(
            columns=[
                "sector",
                "source",
                "subsource",
                "fuel",
                "subref",
                "2nd ref",
                "exclude",
            ]
        )
        # get just methane emissions
        .query("ghg == 'CH4'")
        # remove that column
        .drop(columns="ghg")
        # set the index to state
        .rename(columns={"state": "state_code"})
        .set_index("state_code")
        # covert "NO" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        # drop states that have all nan values
        .dropna(how="all")
        # reset the index state back to a column
        .reset_index()
        # make the table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, 2021)")
        # .query("year.between(@min_year, @max_year)") NOTE: use once we have 2022 data
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
    )
    return reporting_emi_df

def get_nonreporting_msw_landfills_inventory_data(input_path):
    nonreporting_emi_df = pd.DataFrame()
    reporting_emi_df = (
        pd.read_excel(
            input_path,
            sheet_name="InvDB",
            skiprows=15,
            nrows=60,
            usecols="A:AO",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
        # drop columns we don't need
        .drop(
            columns=[
                "sector",
                "source",
                "subsource",
                "fuel",
                "subref",
                "2nd ref",
                "exclude",
            ]
        )
        # get just methane emissions
        .query("ghg == 'CH4'")
        # remove that column
        .drop(columns="ghg")
        # set the index to state
        .rename(columns={"state": "state_code"})
        .set_index("state_code")
        # covert "NO" string to numeric (will become np.nan)
        .apply(pd.to_numeric, errors="coerce")
        # drop states that have all nan values
        .dropna(how="all")
        # reset the index state back to a column
        .reset_index()
        # make the table long by state/year
        .melt(id_vars="state_code", var_name="year", value_name="ch4_tg")
        .assign(ch4_kt=lambda df: df["ch4_tg"] * tg_to_kt)
        .drop(columns=["ch4_tg"])
        # make the columns types correcet
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, 2021)")
        # .query("year.between(@min_year, @max_year)") NOTE: use once we have 2022 data
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
    )

    # Get non-reporting emissions by scaling reporting emissions.
    # Assume emissions are 9% of reporting emissions for 2016 and earlier.
    # Assume emissions are 11% of reporting emissions for 2017 and later.
    emi_09 = reporting_emi_df.query("year <= 2016").assign(ch4_kt=lambda df: df["ch4_kt"] * 0.09)
    emi_11 = reporting_emi_df.query("year >= 2017").assign(ch4_kt=lambda df: df["ch4_kt"] * 0.11)
    nonreporting_emi_df = pd.concat([nonreporting_emi_df, emi_09, emi_11], axis=0)

    return nonreporting_emi_df

def get_reporting_msw_landfills_proxy_data(
    reporting_facility_path,
    state_gdf,
    ):
    # Reporting facilities from Subpart HH
    reporting_facility_df = (
    pd.read_csv(
        reporting_facility_path,
        usecols=("facility_name",
                 "facility_id",
                 "reporting_year",
                 "ghg_quantity",
                 "latitude",
                 "longitude",
                 "state",
                 "zip"))
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"reporting_year": "year", "ghg_quantity": "ch4_t", "state": "state_code"})
    .assign(ch4_kt=lambda df: df["ch4_t"] * t_to_kt)
    .drop(columns=["ch4_t"])
    .drop_duplicates(subset=['facility_id', 'year'], keep='last')
    .astype({"year": int})
    # .query("year.between(@min_year, @max_year)")
    .query("year.between(@min_year, 2021)")
    .reset_index()
    .drop(columns='index')
    )

    reporting_facility_gdf = (
        gpd.GeoDataFrame(
            reporting_facility_df,
            geometry=gpd.points_from_xy(
                reporting_facility_df["longitude"],
                reporting_facility_df["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["latitude", "longitude", "zip"])
        .loc[:, ["facility_id", "facility_name", "state_code", "geometry", "year", "ch4_kt"]]
    )

    return reporting_facility_gdf

def get_nonreporting_msw_landfills_proxy_data(
    nonreporting_facility_path,
    frs_facility_path,
    state_gdf,
    ):

    # Non-reporting facilities from LMOP and WBJ
    nonreporting_facility_df = (
    pd.read_excel(
            nonreporting_facility_path,
            sheet_name="LandfillComp",
            usecols="A:B,D,F,AP,BL:BM,BR:BS",
            skiprows=5,
            nrows=1672)
    .rename(columns={
        'Landfill Name (as listed in Original Instance source)': 'Name',
        'HH Off-Ramped, Last Year Reported': 'Last GHGRP Year',
        'Avg. Est. Total WIP (MT)': 'WIP_MT',
        'LMOP Lat': 'LAT',
        'LMOP Long': 'LON',
    }).astype({"WBJ City": str, "WBJ Location": str})
    )
    nonreporting_facility_df['Full_Address'] = nonreporting_facility_df["WBJ Location"]+' '+nonreporting_facility_df["WBJ City"]+' '+nonreporting_facility_df["State"]
    nonreporting_facility_df['Partial_Address'] = nonreporting_facility_df["WBJ City"]+' '+nonreporting_facility_df["State"]
    nonreporting_facility_df.fillna('NaN', inplace=True)
    nonreporting_facility_df = nonreporting_facility_df[nonreporting_facility_df['WIP_MT'] > 0]
    nonreporting_facility_df.reset_index(inplace=True, drop=True)

    # Separate Landfills with and without location information
    # These are the landfills from the waste business jounral (WBJ) that have limited location information
    
    nonreporting_facility_noloc = nonreporting_facility_df[(nonreporting_facility_df['LAT'] == 0) & (nonreporting_facility_df['LON'] == 0)]
    nonreporting_facility_noloc.reset_index(inplace=True, drop=True)

    nonreporting_facility_loc = nonreporting_facility_df[(nonreporting_facility_df['LAT'] != 0) & (nonreporting_facility_df['LON'] != 0)]
    nonreporting_facility_loc.reset_index(inplace=True, drop=True)

    # Read in FRS data to get location information for landfills with missing location information 
    # This comes from the Facility Registration system, the original file (NATIONAL_SINGLE.csv is > 1.5 Gb)

    FRS_facility_locs = pd.read_csv(frs_facility_path, usecols = [2,3,5,7,8,10,17,20,21,26,27,28,31,32,34,35,36],low_memory=False)
    FRS_facility_locs.fillna(0, inplace=True)
    FRS_facility_locs = FRS_facility_locs[FRS_facility_locs['LATITUDE83'] > 0]
    FRS_facility_locs = FRS_facility_locs[FRS_facility_locs['NAICS_CODES'] != 0]
    FRS_facility_locs.reset_index(inplace=True, drop=True)

    FRS_facility_locs = FRS_facility_locs[(FRS_facility_locs['NAICS_CODES'] == '562212')]
    FRS_facility_locs.reset_index(inplace=True, drop=True)

    FRS_facility_locs['CITY_NAME'] = FRS_facility_locs['CITY_NAME'].replace(0, 'NaN')
    FRS_facility_locs.reset_index(inplace=True, drop=True)

    # Loop through the non-reporting landfills that do not have locations to try to find
    # matches in the FRS dataset.

    nonreporting_facility_noloc.loc[:,'found'] = 0

    for ifacility in np.arange(0,len(nonreporting_facility_noloc)):
        # first try matching by state and exact name of landfill
        state_temp = nonreporting_facility_noloc.loc[ifacility,'State']
        imatch = np.where((FRS_facility_locs['PRIMARY_NAME'].str.contains(nonreporting_facility_noloc.loc[ifacility,'Name'].upper()))\
                        & (FRS_facility_locs['STATE_CODE']==state_temp))[0]
        if len(imatch) == 1:
            nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
            nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
            nonreporting_facility_noloc.loc[ifacility,'found'] = 1
        elif len(imatch) > 1:
            # if name and state match more than one entry, use the one with the higher accuracy
            # or the first entry if the accuracy values are the same
            FRS_temp = FRS_facility_locs.loc[imatch,:].copy()
            new_match = np.where(np.max(FRS_temp['ACCURACY_VALUE']))[0]
            if len(new_match) ==1:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[new_match[0]],'LATITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[new_match[0]],'LONGITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'found'] = 1
            elif len(new_match)<1:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'found'] = 1

        # next try matching based on any of the words in name and state and city
        if nonreporting_facility_noloc.loc[ifacility,'found'] == 0:
            string_temp = [x for x in nonreporting_facility_noloc.loc[ifacility,'Name'].upper().split() \
                        if x not in {'LANDFILL', 'SANITARY','CITY','TOWN','OF'}]
            string_temp = '|'.join(string_temp)
            string_temp = string_temp.replace("(","")
            string_temp = string_temp.replace(")","")
            string_temp = string_temp.replace("&","")
            string_temp = string_temp.replace("/","")

            city_temp = nonreporting_facility_noloc.loc[ifacility,'WBJ City'].upper()
            imatch = np.where((FRS_facility_locs['PRIMARY_NAME'].str.contains(string_temp))\
                        & (FRS_facility_locs['STATE_CODE']==state_temp) & (FRS_facility_locs['CITY_NAME']==city_temp))[0]

            if len(imatch) == 1:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'found'] = 1
            elif len(imatch) > 1:
                # if name and state match more than one entry, use the one with the higher accuracy
                # or the first entry if the accuracy values are the same
                FRS_temp = FRS_facility_locs.loc[imatch,:].copy()
                new_match = np.where(np.max(FRS_temp['ACCURACY_VALUE']))[0]
                if len(new_match) ==1:
                    nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[new_match[0]],'LATITUDE83']
                    nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[new_match[0]],'LONGITUDE83']
                    nonreporting_facility_noloc.loc[ifacility,'found'] = 1
                elif len(new_match)<1:
                    nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
                    nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
                    nonreporting_facility_noloc.loc[ifacility,'found'] = 1
        
        # next try matching based on state and city
        if nonreporting_facility_noloc.loc[ifacility,'found'] == 0:
            string_temp = [x for x in nonreporting_facility_noloc.loc[ifacility,'WBJ Location'].upper().split() \
                        if x not in {'ROAD', 'RD','HWY','HIGHWAY'}]
            # string_temp = EPA_nr_msw_noloc.loc[ifacility,'WBJ Location'].upper().split()
            string_temp = '|'.join(string_temp)
            string_temp = string_temp.replace("(","")
            string_temp = string_temp.replace(")","")
            city_temp = nonreporting_facility_noloc.loc[ifacility,'WBJ City'].upper()
            imatch = np.where((FRS_facility_locs['STATE_CODE']==state_temp) & (FRS_facility_locs['CITY_NAME']==city_temp))[0]
            if len(imatch) == 1:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'found'] = 1
            elif len(imatch) >1:
                # if city and state match more than one entry, use the one that has some matching address
                FRS_temp = FRS_facility_locs.loc[imatch,:].copy()
                new_match = np.where(FRS_temp['LOCATION_ADDRESS'].str.contains(string_temp))[0]
                if len(new_match) >= 1:
                    nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[new_match[0]],'LATITUDE83']
                    nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[new_match[0]],'LONGITUDE83']
                    nonreporting_facility_noloc.loc[ifacility,'found'] = 1
                
        if nonreporting_facility_noloc.loc[ifacility,'found'] == 0:
            # check if state matches and city and name have any matches
            city_temp = nonreporting_facility_noloc.loc[ifacility,'WBJ City'].upper()
            imatch = np.where((FRS_facility_locs['PRIMARY_NAME'].str.contains(city_temp))\
                        & (FRS_facility_locs['STATE_CODE']==state_temp))[0]
            if len(imatch) == 1:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'found'] = 1
            elif len(imatch)>1:
                # no good matches in this case (do nothing)
                continue
        if nonreporting_facility_noloc.loc[ifacility,'found'] == 0:
            # check based on state and any matches between names
            string_temp = [x for x in nonreporting_facility_noloc.loc[ifacility,'Name'].upper().split() \
                        if x not in {'LANDFILL', 'SANITARY','COUNTY','CITY','TOWN','OF','LF','WASTE'}]
            # string_temp = EPA_nr_msw_noloc.loc[ifacility,'Name'].upper().split()[0:2]
            string_temp = '|'.join(string_temp)
            # print(string_temp)
            string_temp = string_temp.replace("(","")
            string_temp = string_temp.replace(")","")
            string_temp = string_temp.replace("&","")
            string_temp = string_temp.replace("/","")
            # print(string_temp, state_temp)
            city_temp = nonreporting_facility_noloc.loc[ifacility,'WBJ City'].upper()
            imatch = np.where((FRS_facility_locs['PRIMARY_NAME'].str.contains(string_temp))\
                        & (FRS_facility_locs['STATE_CODE']==state_temp))[0]
            # print(imatch)
            if len(imatch) == 1:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
                nonreporting_facility_noloc.loc[ifacility,'found'] = 1
            elif len(imatch) > 1:
                # no good matches
                continue
        if nonreporting_facility_noloc.loc[ifacility,'found'] == 0:
            continue
    
    # Find locations by geocoding remaining non-reporting landfills.
    # Try Geocoding to convert facility addresses into lat/lon values. 
    # This uses the free openstreetmaps api (not as good as google maps, but free)
    # only need to get locations for facilities where found = 0
    # if this doesn't work with run using 'run all', try running individually
    print("Geolocating non-reporting facilities")
    geolocator = Nominatim(user_agent="myGeocode")
    geopy.geocoders.options.default_timeout = None
    print(geolocator.timeout)

    for ifacility in np.arange(0,len(nonreporting_facility_noloc)):
        if nonreporting_facility_noloc.loc[ifacility,'found'] ==0:
            location = geolocator.geocode(nonreporting_facility_noloc['Full_Address'][ifacility])
            if location is None:
                continue
            else:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = location.latitude
                nonreporting_facility_noloc.loc[ifacility,'LON'] = location.longitude
                nonreporting_facility_noloc.loc[ifacility,'found']=1
                
    print('First Try - Percentage found:',(np.sum(nonreporting_facility_noloc['found']))/len(EPA_nr_msw_noloc))
    print('Missing Count',len(nonreporting_facility_noloc[nonreporting_facility_noloc['found']==0]))

    for ifacility in np.arange(0,len(nonreporting_facility_noloc)):
        if nonreporting_facility_noloc.loc[ifacility,'found'] ==0:
            #if still no match, remove the address portion and just allocate based on city, state
            location = geolocator.geocode(nonreporting_facility_noloc['Partial_Address'][ifacility])
            if location is None:
                continue
            else:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = location.latitude
                nonreporting_facility_noloc.loc[ifacility,'LON'] = location.longitude
                nonreporting_facility_noloc.loc[ifacility,'found']=1
                
    print('Second Try - Percentage found:',(np.sum(nonreporting_facility_noloc['found']))/len(nonreporting_facility_noloc))
    print('Missing Count',len(nonreporting_facility_noloc[nonreporting_facility_noloc['found']==0]))

    # Recombine non-reporting facilities into a single geodataframe
    nonreporting_facility_df_with_locs = nonreporting_facility_loc.merge(nonreporting_facility_noloc, how='left')

    nonreporting_facility_gdf = (
    gpd.GeoDataFrame(
        nonreporting_facility_df_with_locs,
        geometry=gpd.points_from_xy(
            nonreporting_facility_df_with_locs["LON"],
            nonreporting_facility_df_with_locs["LAT"],
            crs=4326,
            ),
            )
    .drop(columns=["LAT", "LON", "WBJ Location", "WBJ City", "Full_Address", "Partial_Address"])
    .loc[:, ["NR ID", "Name", "State", "Last GHGRP Year", "geometry", "WIP_MT"]]
    )

    # Create individual data rows for non-reporting facilities where
    # "Last GHGRP Year" = 0 to have data for each year.
    nonreporting_facility_gdf = pd.DataFrame()
    nr_year0_facilities_expanded_years = pd.DataFrame()
    nr_otheryears_facilities_expanded_years = pd.DataFrame()

    nr_all_facilities_copy = nonreporting_facility_gdf.copy()
    nr_all_facilities_copy = nr_all_facilities_copy.rename(columns={'Last GHGRP Year': 'Last_GHGRP_Year'})
    nr_all_facilities_copy['Year'] = 0


    nr_year0_facilities = nr_all_facilities_copy.query("Last_GHGRP_Year == 0")
    nr_otheryears_facilities = nr_all_facilities_copy.query("Last_GHGRP_Year != 0")
    for iyear in np.arange(0, num_years):
        year_actual = year_range[iyear]
        print('Year:', year_range[iyear])
        # Assign years to facilities with no GHGRP reporting year (Last_GHGRP_Year == 0)
        nr_year0_facilities['Year'] = year_range[iyear]
        nr_year0_facilities_expanded_years = pd.concat([nr_year0_facilities_expanded_years, nr_year0_facilities])

        # Assign years to facilities that have a year they last reported to GHGRP
        nr_otheryears_facilities = nr_otheryears_facilities.query('Last_GHGRP_Year < @year_actual')
        nr_otheryears_facilities['Year'] = year_range[iyear]
        nr_otheryears_facilities_expanded_years = pd.concat([nr_otheryears_facilities_expanded_years, nr_otheryears_facilities])

    nonreporting_facility_gdf = pd.concat([nonreporting_facility_gdf, nr_year0_facilities_expanded_years, nr_otheryears_facilities_expanded_years]).drop(columns="Last_GHGRP_Year")
    nonreporting_facility_gdf = nonreporting_facility_gdf.rename(columns={"NR ID":"facility_id", "Name": "facility_name", "State": "state_code", "WIP_MT": "wip_mt", "Year": "year"})

    return nonreporting_facility_gdf

# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
IPCC_ID = "5A1"
SECTOR_NAME = "waste"
SOURCE_NAME = "msw landfills"
FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")

netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
netcdf_description = (
    f"Gridded EPA Inventory - {SECTOR_NAME} - "
    f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
)

# PATHS
landfills_dir = ghgi_data_dir_path / "landfills"
# sector_data_dir_path = V3_DATA_PATH / "sector"
sector_data_dir_path = Path(
    "~/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/sector"
    ).expanduser()


# OUTPUT FILES
ch4_kt_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year"
ch4_flux_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux"

# INVENTORY INPUT FILE
inventory_workbook_path = landfills_dir / "State_MSW_LF_1990-2021.xlsx"

# PROXY INPUT FILES
# state vector refence with state_code
state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"
# input 1: reporting facility emissions, Subpart HH
reporting_facility_path = "https://data.epa.gov/efservice/hh_subpart_level_information/pub_dim_facility/ghg_name/=/Methane/CSV"
# input 2: non-reported facility waste-in-place
nonreporting_facility_path = sector_data_dir_path / "landfills" / "Non-Reporting_LF_DB_2020_1.12.2021.xlsx"
# input 3: frs facilities where NAICS code is for composting
# frs_naics_path = global_data_dir_path / "NATIONAL_NAICS_FILE.CSV"
# frs_facility_path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV"
frs_facility_path = Path(
    "~/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/GEPA_Source_Code/Global_Input_Data/FRS/national_single/NATIONAL_SINGLE.csv"
    ).expanduser()

# the NAICS code pulled from the v2 notebook for facilities in the FRS data
# COMPOSTING_FRS_NAICS_CODE = 562219

# %% STEP 1. Load GHGI-Proxy Mapping Files ---------------------------------------------

# %% STEP 2: Read In EPA State GHGI Emissions by Year ----------------------------------

# Get state vectors and state_code for use with inventory and proxy data
state_gdf = (
    gpd.read_file(state_geo_path)
    .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
    .rename(columns=str.lower)
    .rename(columns={"stusps": "state_code", "name": "state_name"})
    .astype({"statefp": int})
    # get only lower 48 + DC
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
    .to_crs(4326)
)

EPA_state_emi_reporting_df = get_reporting_msw_landfills_inventory_data(inventory_workbook_path)
EPA_state_emi_nonreporting_df = get_nonreporting_msw_landfills_inventory_data(inventory_workbook_path)

# plot state inventory data
sns.relplot(
    kind="line",
    data=EPA_state_emi_reporting_df,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)

sns.relplot(
    kind="line",
    data=EPA_state_emi_nonreporting_df,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)

# %% STEP 3: GET AND FORMAT PROXY DATA -------------------------------------------------
reporting_msw_landfills_proxy_gdf = get_reporting_msw_landfills_proxy_data(
    reporting_facility_path,
    state_gdf,
)

nonreporting_msw_landfills_proxy_gdf = get_nonreporting_msw_landfills_proxy_data(
    nonreporting_facility_path,
    frs_facility_path,
    state_gdf,
)

# %% STEP 3.1: QA/QC proxy data --------------------------------------------------------
proxy_count_by_state = (
    state_gdf[["state_code"]]
    .merge(
        composting_proxy_gdf["state_code"].value_counts().rename("proxy_count"),
        how="left",
        left_on="state_code",
        right_index=True,
    )
    .sort_values("proxy_count")
)
display(proxy_count_by_state)
print(f"total number of composting proxies: {composting_proxy_gdf.shape[0]:,}")
print(
    "do all states have at least 1 proxy? "
    f"{proxy_count_by_state['proxy_count'].gt(0).all()}"
)
# %% MAP PROXY DATA --------------------------------------------------------------------
ax = composting_proxy_gdf.drop_duplicates().plot(
    "state_code", categorical=True, cmap="Set2", figsize=(10, 10)
)
state_gdf.boundary.plot(ax=ax, color="xkcd:slate", lw=0.2, zorder=1)

# %% STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO PROXIES ---------------------------
allocated_emis_gdf = state_year_point_allocate_emis(
    composting_proxy_gdf, EPA_state_emi_df, proxy_has_year=False, use_proportional=False
)
allocated_emis_gdf

# %% STEP 4.1: QC PROXY ALLOCATED EMISSIONS BY STATE AND YEAR --------------------------
proxy_qc_result = QC_point_proxy_allocation(allocated_emis_gdf, EPA_state_emi_df)

sns.relplot(
    kind="line",
    data=proxy_qc_result,
    x="year",
    y="allocated_ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)

sns.relplot(
    kind="line",
    data=EPA_state_emi_df,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)
proxy_qc_result

# %% STEP 5: RASTERIZE THE CH4 KT AND FLUX ---------------------------------------------
ch4_kt_result_rasters, ch4_flux_result_rasters = grid_point_emissions(
    allocated_emis_gdf
)

# %% STEP 5.1: QC GRIDDED EMISSIONS BY YEAR --------------------------------------------
# TODO: report QC metrics for flux values compared to V2: descriptive statistics
qc_kt_rasters = QC_emi_raster_sums(ch4_kt_result_rasters, EPA_state_emi_df)
qc_kt_rasters

# %% STEP 6: SAVE THE FILES ------------------------------------------------------------
write_tif_output(ch4_kt_result_rasters, ch4_kt_dst_path)
write_tif_output(ch4_flux_result_rasters, ch4_flux_dst_path)
write_ncdf_output(
    ch4_flux_result_rasters,
    ch4_flux_dst_path,
    netcdf_title,
    netcdf_description,
)

# %% STEP 7: PLOT THE RESULTS AND DIFFERENCE, SAVE FIGURES TO FILES --------------------
plot_annual_raster_data(ch4_flux_result_rasters, SOURCE_NAME)
plot_raster_data_difference(ch4_flux_result_rasters, SOURCE_NAME)

# %%

