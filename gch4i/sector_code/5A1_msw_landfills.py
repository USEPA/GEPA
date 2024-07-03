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

# %% STEP 0. Load packages, configuration files, and local parameters

import calendar
import warnings
from pathlib import Path

import osgeo  # noqa
import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import seaborn as sns
from IPython.display import display
import geopy
from geopy.geocoders import Nominatim

# from pytask import Product, task
from rasterio.features import rasterize

from gch4i.config import (
    data_dir_path,
    ghgi_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
    tmp_data_dir_path,
)
from gch4i.gridding import ARR_SHAPE, GEPA_PROFILE
from gch4i.utils import (
    calc_conversion_factor,
    load_area_matrix,
    name_formatter,
    tg_to_kt,
    write_ncdf_output,
    write_tif_output,
)
t_to_kt = 0.001
year_range = [*range(min_year, 2021+1,1)] #List of emission years
# year_range = [*range(min_year, max_year+1,1)] #List of emission years
year_range_str=[str(i) for i in year_range]
num_years = len(year_range)

# %%

# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
INDUSTRY_NAME = "5A1_waste_msw_landfills"

# State-level inventory data
EPA_inputfile = Path(ghgi_data_dir_path / "landfills" / "State_MSW_LF_1990-2021.xlsx")

# NOTE: this file uses the Envirofacts GHG Query Builder to retrieve
# Supart HH facility-level emissions and locations (latitude, longitude). We need to
# check with the sector leads to determine if other data should be used in addition
# to Subpart HH for facility-level emissions.
# Alternate data if API is not used: hh_scale-up_1990-2022_LA.xlsx, hh_OX_info_90-22.xlsx
EPA_ghgrp_msw_landfills_inputfile = "https://data.epa.gov/efservice/hh_subpart_level_information/pub_dim_facility/ghg_name/=/Methane/CSV"

# Path to v2 data
v2_landfills_data_path = Path(
    "~/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/GEPA_Source_Code/GEPA_Landfills/InputData"
).expanduser()
v2_frs_data_path = Path(
    "~/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/GEPA_Source_Code/Global_InputData/FRS"
).expanduser()

# Facility-level data for non-reporting MSW landfills
# NOTE: this data should be updated if EPA has a more recent file they are working from
EPA_nonreporting_msw_inputfile = Path(v2_landfills_data_path / "Non-Reporting_LF_DB_2020_1.12.2021.xlsx"
)

# FRS data to query locations for non-reporting MSW landfill locations
# NOTE: Do we want to use FRS data or LMOP data for non-reporting facility locations?
FRS_inputfile = Path(v2_frs_data_path / "national_single" / "NATIONAL_SINGLE.csv")

# Alternatively, LMOP data to query locations for non-reporting MSW landfill locations
# Facility-level emissions and location data from the Landfill Methane Outreach
# Program (LMOP). Represents a majority of MSW landfills (more than 2,600 MSW landfills
# that are either accepting MSW or closed in the past few decades).
LMOP_msw_landfills_inputfile = Path(ghgi_data_dir_path / "landfills" / "landfilllmopdata.xlsx")

# reference data paths
state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"

# OUTPUT FILES (TODO: include netCDF flux files)
ch4_kt_dst_path = tmp_data_dir_path / f"{INDUSTRY_NAME}_ch4_kt_per_year.tif"
ch4_flux_dst_path = tmp_data_dir_path / f"{INDUSTRY_NAME}_ch4_emi_flux.tif"

area_matrix = load_area_matrix()

# %% STEP 1: Load GHGI-Proxy Mapping Files

# NOTE: looking at rework of the proxy mapping files into an aggregate flat file
# that would then be formed into a proxy dictionary to retain the existing approach
# but allow for interrogation of the objects when needed.

ghgi_map_path = data_dir_path / "all_ghgi_mappings.csv"
proxy_map_path = data_dir_path / "all_proxy_mappings.csv"
ghgi_map_df = pd.read_csv(ghgi_map_path)
proxy_map_df = pd.read_csv(proxy_map_path)

proxy_map_df.query("GHGI_Emi_Group == 'Emi_Petro'").merge(
    ghgi_map_df.query("GHGI_Emi_Group == 'Emi_Petro'"), on="GHGI_Emi_Group"
)

# %%
# read in the state shapefile to spatial join with facilities, assigning them to
# states for allocation of emissions
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
state_gdf

# %%
# STEP 2: Read In EPA State GHGI Emissions by Year

# read in the ch4_kt values for each state
EPA_msw_landfills_emissions = (
    # read in the data
    pd.read_excel(
        EPA_inputfile,
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
    .rename(columns={"state": "state_code"})
    # get just methane emissions
    .query("ghg == 'CH4'")
    # remove that column
    .drop(columns="ghg")
    # set the index to state
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
    # .query("year.between(@min_year, @max_year)")
    .reset_index(drop=True)
)
EPA_msw_landfills_emissions.head()

# %% 
# QA/QC - check counts of years of state data and plot by state
display(EPA_msw_landfills_emissions["state_code"].value_counts())
display(EPA_msw_landfills_emissions["year"].min(), EPA_msw_landfills_emissions["year"].max())

# a quick plot to verify the values
sns.relplot(
    kind="line",
    data=EPA_msw_landfills_emissions,
    x="year",
    y="ch4_kt",
    hue="state_code",
    # legend=False,
)

# %%
# STEP 3: Read In MSW Landfill Proxy Emissions Data

# STEP 3.1: Read in Proxy Mapping File & Make Proxy Arrays
def task_map_msw_landfills_proxy():
    pass


# %%
# STEP 3.2: Read in MSW Landfill Proxy Emissions Data

# STEP 3.2.1: Read in GHGRP Subpart HH Data (Reporting Landfill Data)

subpart_hh_df = pd.read_csv(
    EPA_ghgrp_msw_landfills_inputfile,
    usecols=("facility_name",
             "facility_id",
             "reporting_year",
             "ghg_quantity",
             "latitude",
             "longitude",
             "state",
             "zip"))
subpart_hh_df

msw_facilities_df = (
    pd.read_csv(
        EPA_ghgrp_msw_landfills_inputfile,
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

msw_facilities_df.head()

msw_facilities_gdf = (
    gpd.GeoDataFrame(
        msw_facilities_df,
        geometry=gpd.points_from_xy(
            msw_facilities_df["longitude"],
            msw_facilities_df["latitude"],
            crs=4326,
        ),
    )
    .drop(columns=["latitude", "longitude"])
    .loc[:, ["facility_id", "facility_name", "state_code", "zip", "geometry", "year", "ch4_kt"]]
)
msw_facilities_gdf.head()

# %% QA/QC
# make sure the merge gave us the number of results we expected.
if not (msw_facilities_gdf.shape[0] == msw_facilities_df.shape[0]):
    print("WARNING the merge shape does not match the original data")

# %%

# STEP 3.2.2: Read in EPA Non-Reporting Landfills Data

# Read in Non-Reporting MSW Landfill Information and Estimate Emissions (based on waste in place)

EPA_nr_msw_df = (
    pd.read_excel(
        EPA_nonreporting_msw_inputfile,
        sheet_name="LandfillComp",
        usecols="A:B,D,F,AP,BL:BM,BR:BS",
        skiprows=5,
        nrows=1672
        )
        )

EPA_nr_msw_df = EPA_nr_msw_df.rename(
    columns={
        'Landfill Name (as listed in Original Instance source)': 'Name',
        'HH Off-Ramped, Last Year Reported': 'Last GHGRP Year',
        'Avg. Est. Total WIP (MT)': 'WIP_MT',
        'LMOP Lat': 'LAT',
        'LMOP Long': 'LON',
    }).astype({"WBJ City": str, "WBJ Location": str})
EPA_nr_msw_df['Full_Address'] = EPA_nr_msw_df["WBJ Location"]+' '+EPA_nr_msw_df["WBJ City"]+' '+EPA_nr_msw_df["State"]
EPA_nr_msw_df['Partial_Address'] = EPA_nr_msw_df["WBJ City"]+' '+EPA_nr_msw_df["State"]
EPA_nr_msw_df.fillna('NaN', inplace=True)
EPA_nr_msw_df = EPA_nr_msw_df[EPA_nr_msw_df['WIP_MT'] > 0]
EPA_nr_msw_df.reset_index(inplace=True, drop=True)

print('Total Non-Reporting Landfills:', len(EPA_nr_msw_df))
# EPA_nr_msw_df.head()

# Separate Landfills with and without location information
# These are the landfills from the waste business jounral (WBJ) that have limited location information
EPA_nr_msw_noloc = EPA_nr_msw_df[(EPA_nr_msw_df['LAT'] == 0) & (EPA_nr_msw_df['LON'] == 0)]
EPA_nr_msw_noloc.reset_index(inplace=True, drop=True)

print('Total Non-Reporting Landfills Without Locations:', len(EPA_nr_msw_noloc))
# display(EPA_nr_msw_noloc)

EPA_nr_msw_loc = EPA_nr_msw_df[(EPA_nr_msw_df['LAT'] != 0) & (EPA_nr_msw_df['LON'] != 0)]
EPA_nr_msw_loc.reset_index(inplace=True, drop=True)

print('Total Non-Reporting Landfills With Locations:', len(EPA_nr_msw_loc))
# display(EPA_nr_msw_loc)

# %%

# Step 3.2.3 Read in FRS Landfills Dataset

# Read in FRS data to get location information for landfills with missing location information 
# This comes from the Facility Registration system, the original file (NATIONAL_SINGLE.csv is > 1.5 Gb)
   
FRS_facility_locs = pd.read_csv(FRS_inputfile, usecols = [2,3,5,7,8,10,17,20,21,26,27,28,31,32,34,35,36],low_memory=False)
FRS_facility_locs.fillna(0, inplace=True)
FRS_facility_locs = FRS_facility_locs[FRS_facility_locs['LATITUDE83'] > 0]
FRS_facility_locs = FRS_facility_locs[FRS_facility_locs['NAICS_CODES'] != 0]
FRS_facility_locs.reset_index(inplace=True, drop=True)

FRS_facility_locs = FRS_facility_locs[(FRS_facility_locs['NAICS_CODES'] == '562212')]
print('Total landfills: ', len(FRS_facility_locs))
FRS_facility_locs.reset_index(inplace=True, drop=True)

FRS_facility_locs['CITY_NAME'] = FRS_facility_locs['CITY_NAME'].replace(0, 'NaN')
FRS_facility_locs.reset_index(inplace=True, drop=True)

print('Landfills selected: ', len(FRS_facility_locs))
FRS_facility_locs.head(5)

# %%

# Step 3.2.4 Find Locations by Matching EPA Non-Reporting Landfills (Without Locations) to FRS

# Loop through the Non-Reporting MSW Landfill records that don't have locations, to 
# try to find matches in the FRS dataset
# Note that there are more landfills in the FRS dataset, then then GHGRP+Non-reporting dataset
# In the previous GEPA, all FRS landfills were used. In the GEPA v2, we only use those
# landfills identified and used to estimate national emissions in the GHGI (i.e., 
# GHGRP + Non-Reporting landfills)

EPA_nr_msw_noloc.loc[:,'found'] = 0

for ifacility in np.arange(0,len(EPA_nr_msw_noloc)):
    # first try matching by state and exact name of landfill
    state_temp = EPA_nr_msw_noloc.loc[ifacility,'State']
    imatch = np.where((FRS_facility_locs['PRIMARY_NAME'].str.contains(EPA_nr_msw_noloc.loc[ifacility,'Name'].upper()))\
                      & (FRS_facility_locs['STATE_CODE']==state_temp))[0]
    if len(imatch) == 1:
        EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
        EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
        EPA_nr_msw_noloc.loc[ifacility,'found'] = 1
    elif len(imatch) > 1:
        # if name and state match more than one entry, use the one with the higher accuracy
        # or the first entry if the accuracy values are the same
        FRS_temp = FRS_facility_locs.loc[imatch,:].copy()
        new_match = np.where(np.max(FRS_temp['ACCURACY_VALUE']))[0]
        if len(new_match) ==1:
            EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[new_match[0]],'LATITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[new_match[0]],'LONGITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'found'] = 1
        elif len(new_match)<1:
            EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'found'] = 1

    # next try matching based on any of the words in name and state and city
    if EPA_nr_msw_noloc.loc[ifacility,'found'] == 0:
        string_temp = [x for x in EPA_nr_msw_noloc.loc[ifacility,'Name'].upper().split() \
                       if x not in {'LANDFILL', 'SANITARY','CITY','TOWN','OF'}]
        # string_temp = EPA_nr_msw_noloc.loc[ifacility,'Name'].upper().split()[0:2]
        string_temp = '|'.join(string_temp)
        string_temp = string_temp.replace("(","")
        string_temp = string_temp.replace(")","")
        string_temp = string_temp.replace("&","")
        string_temp = string_temp.replace("/","")
        # print(string_temp)
        city_temp = EPA_nr_msw_noloc.loc[ifacility,'WBJ City'].upper()
        imatch = np.where((FRS_facility_locs['PRIMARY_NAME'].str.contains(string_temp))\
                      & (FRS_facility_locs['STATE_CODE']==state_temp) & (FRS_facility_locs['CITY_NAME']==city_temp))[0]
        # print(imatch)
        if len(imatch) == 1:
            EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'found'] = 1
        elif len(imatch) > 1:
            # if name and state match more than one entry, use the one with the higher accuracy
            # or the first entry if the accuracy values are the same
            FRS_temp = FRS_facility_locs.loc[imatch,:].copy()
            new_match = np.where(np.max(FRS_temp['ACCURACY_VALUE']))[0]
            if len(new_match) ==1:
                EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[new_match[0]],'LATITUDE83']
                EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[new_match[0]],'LONGITUDE83']
                EPA_nr_msw_noloc.loc[ifacility,'found'] = 1
            elif len(new_match)<1:
                EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
                EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
                EPA_nr_msw_noloc.loc[ifacility,'found'] = 1
    
    # next try matching based on state and city
    if EPA_nr_msw_noloc.loc[ifacility,'found'] == 0:
        string_temp = [x for x in EPA_nr_msw_noloc.loc[ifacility,'WBJ Location'].upper().split() \
                       if x not in {'ROAD', 'RD','HWY','HIGHWAY'}]
        # string_temp = EPA_nr_msw_noloc.loc[ifacility,'WBJ Location'].upper().split()
        string_temp = '|'.join(string_temp)
        string_temp = string_temp.replace("(","")
        string_temp = string_temp.replace(")","")
        city_temp = EPA_nr_msw_noloc.loc[ifacility,'WBJ City'].upper()
        imatch = np.where((FRS_facility_locs['STATE_CODE']==state_temp) & (FRS_facility_locs['CITY_NAME']==city_temp))[0]
        if len(imatch) == 1:
            EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'found'] = 1
        elif len(imatch) >1:
            # if city and state match more than one entry, use the one that has some matching address
            FRS_temp = FRS_facility_locs.loc[imatch,:].copy()
            new_match = np.where(FRS_temp['LOCATION_ADDRESS'].str.contains(string_temp))[0]
            if len(new_match) >= 1:
                EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[new_match[0]],'LATITUDE83']
                EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[new_match[0]],'LONGITUDE83']
                EPA_nr_msw_noloc.loc[ifacility,'found'] = 1
            
    if EPA_nr_msw_noloc.loc[ifacility,'found'] == 0:
        # check if state matches and city and name have any matches
        city_temp = EPA_nr_msw_noloc.loc[ifacility,'WBJ City'].upper()
        imatch = np.where((FRS_facility_locs['PRIMARY_NAME'].str.contains(city_temp))\
                      & (FRS_facility_locs['STATE_CODE']==state_temp))[0]
        if len(imatch) == 1:
            EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'found'] = 1
        elif len(imatch)>1:
            # no good matches in this case (do nothing)
            continue
    if EPA_nr_msw_noloc.loc[ifacility,'found'] == 0:
        # check based on state and any matches between names
        string_temp = [x for x in EPA_nr_msw_noloc.loc[ifacility,'Name'].upper().split() \
                       if x not in {'LANDFILL', 'SANITARY','COUNTY','CITY','TOWN','OF','LF','WASTE'}]
        # string_temp = EPA_nr_msw_noloc.loc[ifacility,'Name'].upper().split()[0:2]
        string_temp = '|'.join(string_temp)
        # print(string_temp)
        string_temp = string_temp.replace("(","")
        string_temp = string_temp.replace(")","")
        string_temp = string_temp.replace("&","")
        string_temp = string_temp.replace("/","")
        # print(string_temp, state_temp)
        city_temp = EPA_nr_msw_noloc.loc[ifacility,'WBJ City'].upper()
        imatch = np.where((FRS_facility_locs['PRIMARY_NAME'].str.contains(string_temp))\
                      & (FRS_facility_locs['STATE_CODE']==state_temp))[0]
        # print(imatch)
        if len(imatch) == 1:
            EPA_nr_msw_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
            EPA_nr_msw_noloc.loc[ifacility,'found'] = 1
        elif len(imatch) > 1:
            # no good matches
            continue
    if EPA_nr_msw_noloc.loc[ifacility,'found'] == 0:
        continue
print('Count', len(EPA_nr_msw_noloc[EPA_nr_msw_noloc['found'] == 0]))

print(EPA_nr_msw_noloc)

# %%

# Step 3.2.5: Find Locations by Geocoding Remaining EPA Non-Reporting Landfills (Without Locations)

# TRY GEOCODING
# Try Geocoding to convert facility addresses into lat/lon values. 
# This uses the free openstreetmaps api (not as good as google maps, but free)
# only need to get locations for facilities where found = 0
# if this doesn't work with run using 'run all', try running individually

# NOTE: This step crashes on the first try, but has been working when running a second try. 

geolocator = Nominatim(user_agent="myGeocode")
geopy.geocoders.options.default_timeout = None
print(geolocator.timeout)

for ifacility in np.arange(0,len(EPA_nr_msw_noloc)):
    if EPA_nr_msw_noloc.loc[ifacility,'found'] ==0:
        location = geolocator.geocode(EPA_nr_msw_noloc['Full_Address'][ifacility])
        if location is None:
            continue
        else:
            EPA_nr_msw_noloc.loc[ifacility,'LAT'] = location.latitude
            EPA_nr_msw_noloc.loc[ifacility,'LON'] = location.longitude
            EPA_nr_msw_noloc.loc[ifacility,'found']=1
            
print('First Try - Percentage found:',(np.sum(EPA_nr_msw_noloc['found']))/len(EPA_nr_msw_noloc))
print('Missing Count',len(EPA_nr_msw_noloc[EPA_nr_msw_noloc['found']==0]))

for ifacility in np.arange(0,len(EPA_nr_msw_noloc)):
    if EPA_nr_msw_noloc.loc[ifacility,'found'] ==0:
        #if still no match, remove the address portion and just allocate based on city, state
        location = geolocator.geocode(EPA_nr_msw_noloc['Partial_Address'][ifacility])
        if location is None:
            continue
        else:
            EPA_nr_msw_noloc.loc[ifacility,'LAT'] = location.latitude
            EPA_nr_msw_noloc.loc[ifacility,'LON'] = location.longitude
            EPA_nr_msw_noloc.loc[ifacility,'found']=1
            
print('Second Try - Percentage found:',(np.sum(EPA_nr_msw_noloc['found']))/len(EPA_nr_msw_noloc))
print('Missing Count',len(EPA_nr_msw_noloc[EPA_nr_msw_noloc['found']==0]))

# %%
# Recombine into a single dataframe
EPA_nr_msw_final = EPA_nr_msw_loc.merge(EPA_nr_msw_noloc, how='left')
EPA_nr_msw_final.head()

nr_msw_facilities_gdf = (
    gpd.GeoDataFrame(
        EPA_nr_msw_final,
        geometry=gpd.points_from_xy(
            EPA_nr_msw_final["LON"],
            EPA_nr_msw_final["LAT"],
            crs=4326,
        ),
    )
    .drop(columns=["LAT", "LON", "WBJ Location", "WBJ City", "Full_Address", "Partial_Address"])
    # remove facilities without lat/lon
    # .query("found == 1")
    .loc[:, ["NR ID", "Name", "State", "Last GHGRP Year", "geometry", "WIP_MT"]]
)
nr_msw_facilities_gdf.head()

# %%
# Create individual data rows for non-reporting facilities where "Last GHGRP Year" = 0
# to have data for each year.

nr_msw_facilities_gdf_expanded_years = pd.DataFrame()
nr_year0_facilities_expanded_years = pd.DataFrame()
nr_otheryears_facilities_expanded_years = pd.DataFrame()

nr_all_facilities_copy = nr_msw_facilities_gdf.copy()
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

nr_msw_facilities_gdf_expanded_years = pd.concat([nr_msw_facilities_gdf_expanded_years, nr_year0_facilities_expanded_years, nr_otheryears_facilities_expanded_years]).drop(columns="Last_GHGRP_Year")
nr_msw_facilities_gdf_expanded_years = nr_msw_facilities_gdf_expanded_years.rename(columns={"NR ID":"facility_id", "Name": "facility_name", "State": "state_code", "WIP_MT": "wip_mt", "Year": "year"})
nr_msw_facilities_gdf_expanded_years.head()


# %%

# STEP 4: ALLOCATION OF STATE / YEAR EMISSIONS TO EACH FACILITY
#         (BY PROXY FRACTION IN EACH GRIDCELL)

# For this source, state-level emissions are spatially allocated using the
# the fraction of facility-level emissions (for reporting facilities) or
# fraction of waste-in-place (for non-reporting facilities)
# within each grid cell in each state, for each year.

# Function of allocate emissions to reporting facilities.
def state_year_allocation_emissions(fac_emissions, inventory_df, non_reported):

    # fac_emissions are the total emissions for the facilities located in that state
    # and year. It can be one or more facilities. Inventory_df EPA state GHGI summary
    # emissions table

    # get the target state and year
    state, year = fac_emissions.name

    # get the total proxy data (e.g., emissions) within that state and year. 
    # It will be a single value. It is assumed 9-11% of emissions were not reported.
    # A 9% and 11% scale-up factor is used to determine the non-reporting emissions.
    if non_reported:
        if year <= 2016:
            emi_scale = 0.09
        else:
            emi_scale = 0.11
    else:
        emi_scale = 1.0

    emi_sum = inventory_df[
        (inventory_df["state_code"] == state) & (inventory_df["year"] == year)
    ]["ch4_kt"].iat[0] * emi_scale

    # allocate the EPA GHGI state emissions to each individual facility based on their
    # proportion emissions (for reporting facilities) or WIP (for non-reporting facilities)
    # (i.e., the fraction of total state-level emissions or WIP occuring at each facility)
    allocated_fac_emissions = ((fac_emissions / fac_emissions.sum()) * emi_sum).fillna(0)
    
    return allocated_fac_emissions


# we create a new column that assigns the allocated summary emissions to each facility
# based on its proportion of emission to the facility totals for that state and year.
# so for each state and year in the summary emissions we apply the function.
msw_facilities_gdf["allocated_ch4_kt"] = msw_facilities_gdf.groupby(
    ["state_code", "year"])["ch4_kt"].transform(
        state_year_allocation_emissions, inventory_df=EPA_msw_landfills_emissions, non_reported=False)

msw_facilities_gdf.head()

nr_msw_facilities_gdf_expanded_years["allocated_ch4_kt"] = nr_msw_facilities_gdf_expanded_years.groupby(
    ["state_code", "year"])["wip_mt"].transform(
        state_year_allocation_emissions, inventory_df=EPA_msw_landfills_emissions, non_reported=True)

nr_msw_facilities_gdf_expanded_years.head()

# %%
# Allocate emissions to non-reporting facilities.
# print('QA/QC: report final landfill values to be placed on CONUS grid')
# for iyear in np.arange(0, num_years):
#     print('Year:', year_range[iyear])
#     # print('Total Landfills (counts):                          ',ghgrp_count+len(EPA_nr_msw_final.iloc[imatch,0]))
#     print('Total Landfill Emissions (kt):                     ',ghgrp_total_emis[iyear]+nonrepoting_total_emis[iyear])
#     print('Total GHGRP Landfills (counts):                    ',ghgrp_count)
#     print('Total GHGRP Landfill Emissions (kt):               ',ghgrp_total_emis[iyear] )
#     print('Total Non-Reporting Landfills w/location (counts): ',len(EPA_nr_msw_final.iloc[imatch,0]))
#     print('Total Non-Reporting Landfills w/location Emis (kt):',np.sum(EPA_nr_msw_final.loc[imatch,'emis_'+year_range_str[iyear]]))
#     print('')


# %%

# Combine reporting and non-reporting facility gdfs.
facilities_gdf = pd.DataFrame()
msw_facilities_gdf = msw_facilities_gdf.drop(columns=["ch4_kt", "zip"])
nr_msw_facilities_gdf_expanded_years = nr_msw_facilities_gdf_expanded_years.drop(
    columns=["wip_mt"])
facilities_gdf = pd.concat([facilities_gdf, msw_facilities_gdf, nr_msw_facilities_gdf_expanded_years]).reset_index(drop=True)
facilities_gdf.head()

# %%


# %% # save a shapefile of the v3 msw facilities for reference
fac_locations = facilities_gdf.dissolve("facility_id")
fac_locations[fac_locations.is_valid].loc[:, ["geometry"]].to_file(
    tmp_data_dir_path / "v3_waste_msw_landfills_facilities.shp.zip", driver="ESRI Shapefile"
)

# %% QA/QC
# We now check that the sum of facility emissions equals the EPA GHGI emissions by state
# and year. The resulting sum_check table shows you where the emissions data DO NOT
# equal and need more investigation.
# NOTE: currently we are missing facilities in states, so we also check below that the
# states that are missing emissions are the ones that are missing facilities.
sum_check = (
    facilities_gdf.groupby(["state_code", "year"])["allocated_ch4_kt"]
    .sum()
    .reset_index()
    .merge(EPA_msw_landfills_emissions, on=["state_code", "year"], how="outer")
    .assign(
        check_diff=lambda df: df.apply(
            lambda x: np.isclose(x["allocated_ch4_kt"], x["ch4_kt"]), axis=1
        )
    )
)

with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(sum_check[~sum_check["check_diff"]])

# NOTE: For now, facilities data are not final / missing. We don't have facilities in
# all the state summaries that are reporting, and we may be missing facilities even
# within states that are represented. If these lists match, we have a good idea of
# what is missing currently due to the preliminary data.


print(
    (
        "states with no facilities in them: "
        f"{EPA_msw_landfills_emissions[~EPA_msw_landfills_emissions['state_code'].isin(facilities_gdf['state_code'])]['state_code'].unique()}"
    )
)

print(
    (
        "states with facilities in them but not accounted in state inventory: "
        f"{facilities_gdf[~facilities_gdf['state_code'].isin(EPA_msw_landfills_emissions['state_code'])]['state_code'].unique()}"
    )
)

print(
    (
        "states with unaccounted emissions: "
        f"{sum_check[~sum_check['check_diff']]['state_code'].unique()}"
    )
)

# %%

# STEP 5: RASTERIZE THE CH4 KT AND FLUX
#         e.g., calculate fluxes and place the facility-level emissions on the CONUS grid

# for each year, grid the adjusted emissions data in kt and do conversion for flux.
ch4_kt_result_rasters = {}
ch4_flux_result_rasters = {}

# NOTE: this warning filter is because we currently have facilities with missing
# geometries.
# TODO: remove this filter when full locations data are available.
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for year, data in facilities_gdf.groupby("year"):

        # same results as summing the month days
        # if calendar.isleap(year):
        #     year_days = 366
        # else:
        #     year_days = 365
        month_days = [calendar.monthrange(year, x)[1] for x in range(1, 13)]
        year_days = np.sum(month_days)

        # TODO: check that when multiple points fall into the same cell, their values
        # are added together.
        ch4_kt_raster = rasterize(
            shapes=[
                (shape, value)
                for shape, value in data[["geometry", "allocated_ch4_kt"]].values
            ],
            out_shape=ARR_SHAPE,
            fill=0,
            transform=GEPA_PROFILE["transform"],
            dtype=np.float64,
            merge_alg=rasterio.enums.MergeAlg.add,
        )

        conversion_factor_annual = calc_conversion_factor(year_days, area_matrix)
        ch4_flux_raster = ch4_kt_raster * conversion_factor_annual

        ch4_kt_result_rasters[year] = ch4_kt_raster
        ch4_flux_result_rasters[year] = ch4_flux_raster
# --------------------------------------------------------------------------


# %%
# STEP 6: SAVE THE FILES

# check the sums all together now...
# TODO: report QC metrics for both kt and flux values

for year, raster in ch4_kt_result_rasters.items():
    raster_sum = raster.sum()
    fac_sum = facilities_gdf.query("year == @year")["allocated_ch4_kt"].sum()
    emi_sum = EPA_msw_landfills_emissions.query("year == @year")["ch4_kt"].sum()
    missing_sum = sum_check[(~sum_check["check_diff"]) & (sum_check["year"] == year)][
        "ch4_kt"
    ].sum()

    print(year)
    print(
        "does the raster sum equal the facility sum: "
        f"{np.isclose(raster_sum, fac_sum)}"
    )
    print(
        "does the raster sum equal the national total: "
        f"{np.isclose(raster_sum, emi_sum)}"
    )
    # this shows we are consistent with our missing emissions where the states with no
    # facilities make up the difference, as we would expect.
    print(
        "do the states with no facilities equal the missing amount: "
        f"{np.isclose((emi_sum - raster_sum), missing_sum)}"
    )
    print()

# %% Write files

# EEM: TODO: add netCDF output
write_tif_output(ch4_kt_result_rasters, ch4_kt_dst_path)
write_tif_output(ch4_flux_result_rasters, ch4_flux_dst_path)
# ------------------------------------------------------------------------
# %%
# STEP 7: PLOT

# TODO: add map of output flux data

#%%

# END
