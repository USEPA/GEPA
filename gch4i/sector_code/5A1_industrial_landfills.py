"""
Name:               5A1_industrial_landfills_pulp_paper.py
Authors Name:       N. Kruskamp, H. Lohman (RTI International), Erin McDuffie (EPA/OAP)
Date Last Modified: 06/19/2024
Purpose:            Spatially allocates methane emissions for source category 5A1
                    industrial landfills - pulp & paper mills.
Input Files:
                    - 
Output Files:
                    - {FULL_NAME}_ch4_kt_per_year.tif
                    - {FULL_NAME}_ch4_emi_flux.tif
Notes:
TODO: get latest inventory data
"""

# %% STEP 0. Load packages, configuration files, and local parameters ------------------

# for testing/development
# %load_ext autoreload
# %autoreload 2

from pathlib import Path
import osgeo  # noqa
import duckdb
import geopandas as gpd
import pandas as pd
import seaborn as sns
from IPython.display import display
from shapely import Point, wkb
import numpy as np

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

gpd.options.io_engine = "pyogrio"

# other utils
t_to_kt = 0.001  # conversion factor, tonnes to kilotonnes
mmt_to_kt = 1000  # conversion factor, million metric tonnes to kilotonnes


year_range = [*range(min_year, 2021+1,1)] #List of emission years
# year_range = [*range(min_year, max_year+1,1)] #List of emission years
year_range_str=[str(i) for i in year_range]
num_years = len(year_range)


def get_pulp_paper_inventory_data(input_path, state_geo_path):
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

    pulp_paper_emi_df = (
        pd.read_excel(
            input_path,
            sheet_name="P&P State Emissions",
            skiprows=5,
            nrows=60,
            usecols="B:AH",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
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
        .melt(id_vars="state_code", var_name="year", value_name="ch4_mmt")
        .assign(ch4_kt=lambda df: df["ch4_mmt"] * mmt_to_kt)
        .drop(columns=["ch4_mmt"])
        # make the columns types correct
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, 2021)")
        # .query("year.between(@min_year, @max_year)") NOTE: use once we have 2022 data
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
    )
    return pulp_paper_emi_df


def get_reporting_pulp_paper_inventory_data():
    # assign GHGRP subpart tt totals to each state-year combination.


def get_nonreporting_pulp_paper_inventory_data():
    # for each state-year combination, compare the total emissions in GHGI against the 
    # sum of emissions of facilities in subpart tt. Assign equally to facilities in 
    # non-reporting list.

    return


def get_food_beverage_inventory_data(input_path, state_geo_path):
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
    
    food_beverage_emi_df = (
        pd.read_excel(
            input_path,
            sheet_name="F&B State Emissions",
            skiprows=5,
            nrows=927,
            usecols="B:AH",
        )
        # name column names lower
        .rename(columns=lambda x: str(x).lower())
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
        .melt(id_vars="state_code", var_name="year", value_name="ch4_mmt")
        .assign(ch4_kt=lambda df: df["ch4_mmt"] * mmt_to_kt)
        .drop(columns=["ch4_mmt"])
        # make the columns types correct
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, 2021)")
        # .query("year.between(@min_year, @max_year)") NOTE: use once we have 2022 data
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
    )
    return food_beverage_emi_df


# %%
def get_reporting_pulp_paper_proxy_data(
    subpart_tt_path,
    state_geo_path,
):
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
    # Reporting facilities from subpart tt
    reporting_pulp_paper_df = (
    pd.read_csv(
        subpart_tt_path,
        usecols=("facility_name",
                    "facility_id",
                    "reporting_year",
                    "ghg_quantity",
                    "latitude",
                    "longitude",
                    "state",
                    "city",
                    "zip",
                    "naics_code"))
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"reporting_year": "year", "ghg_quantity": "ch4_t", "state": "state_code"})
    .assign(ch4_kt=lambda df: df["ch4_t"] * t_to_kt)
    .drop(columns=["ch4_t"])
    .drop_duplicates(subset=['facility_id', 'year'], keep='last')
    .astype({"year": int})
    # .query("year.between(@min_year, @max_year)")
    .query("year.between(@min_year, 2021)")
    .astype({"naics_code": str})
    .query("naics_code.str.startswith('321') | naics_code.str.startswith('322')" )
    .drop(columns=["naics_code"])
    .query("state_code.isin(@state_gdf['state_code'])")
    .reset_index(drop=True)
    .drop(columns='index')
    )

    reporting_pulp_paper_gdf = (
        gpd.GeoDataFrame(
            reporting_pulp_paper_df,
            geometry=gpd.points_from_xy(
                reporting_pulp_paper_df["longitude"],
                reporting_pulp_paper_df["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["latitude", "longitude", "city", "zip"])
        .loc[:, ["facility_id", "facility_name", "state_code", "geometry", "year", "ch4_kt"]]
    )

    return reporting_pulp_paper_gdf


def get_nonreporting_pulp_paper_proxy_data(
    subpart_tt_path,
    mills_online_path,
    frs_facility_path,
    state_geo_path,
):
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
    # Reporting facilities from subpart tt
    reporting_pulp_paper_df = (
    pd.read_csv(
        subpart_tt_path,
        usecols=("facility_name",
                    "facility_id",
                    "reporting_year",
                    "ghg_quantity",
                    "latitude",
                    "longitude",
                    "state_name",
                    "state",
                    "city",
                    "naics_code"))
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"reporting_year": "year", "ghg_quantity": "ch4_t", "state": "state_code"})
    .assign(ch4_kt=lambda df: df["ch4_t"] * t_to_kt)
    .drop(columns=["ch4_t"])
    .drop_duplicates(subset=['facility_id', 'year'], keep='last')
    .astype({"year": int})
    # .query("year.between(@min_year, @max_year)")
    .query("year.between(@min_year, 2021)")
    .astype({"naics_code": str})
    .query("naics_code.str.startswith('321') | naics_code.str.startswith('322')" )
    .drop(columns=["naics_code"])
    .query("state_code.isin(@state_gdf['state_code'])")
    .reset_index(drop=True)
    )
    # list of pulp and paper mills by state, county, and city
    mills_online_df = (
        pd.read_excel(
            mills_online_path,
            skiprows=3,
            nrows=927,
            usecols="A:F",
        )
        .rename(columns=lambda x: str(x).lower())
        .rename(columns={"mill id#": "facility_id", "state": "state_name", "pulp and paper mill": "pulp_and_paper_mill"})
        .query("pulp_and_paper_mill == 'Yes'")
        .drop(columns=["pulp_and_paper_mill"])
        .query("state_name.isin(@state_info_df['state_name'])")
        .reset_index(drop=True)
    )

    # assign state codes to mills online facilities
    mills_locs = mills_online_df.copy()
    num_mills = len(mills_locs)

    for imill in np.arange(0,num_mills):
        state_name = mills_locs['state_name'][imill]
        state_code = state_gdf[state_gdf['state_name'] == state_name]['state_code']
        mills_locs.loc[imill,'state_code'] = state_code.to_string(index=False)

    # match subpart tt reporting facilities to the mills online facility list 
    # and pull out non-reporting facilities
    mills_locs.loc[:,'ghgrp_match'] = 0
    mills_locs.loc[:,'lat'] = 0
    mills_locs.loc[:,'lon'] = 0
    mills_locs.loc[:,'city'] = mills_locs.loc[:,'city'].str.lower()

    reporting_pulp_paper_df.loc[:,'found'] = 0
    reporting_pulp_paper_df.loc[:,'city'] = reporting_pulp_paper_df.loc[:,'city'].str.lower()

    for iyear in np.arange(0, num_years):
        for imill in np.arange(0,num_mills):
            imatch = np.where((reporting_pulp_paper_df['year'] == year_range[iyear]) & \
                            (reporting_pulp_paper_df['state_code'] == mills_locs.loc[imill,'state_code']) & \
                            (reporting_pulp_paper_df['city'] == mills_locs.loc[imill,'city']))[0]
                            # (reporting_pulp_paper_df['city'].str.contains(mills_locs.loc[imill,'city'].upper())))[0]
            if len(imatch) > 0:
                mills_locs.loc[imill,'ghgrp_match'] = 1
                mills_locs.loc[imill,'lat'] = reporting_pulp_paper_df.loc[imatch[0],'latitude']
                mills_locs.loc[imill,'lon'] = reporting_pulp_paper_df.loc[imatch[0],'longitude']
                reporting_pulp_paper_df.loc[imatch[0],'found'] = 1
            else:
                continue

        print('Found (%) Year',year_range[iyear],':',100*np.sum(mills_locs['ghgrp_match']/len(mills_locs)))

    # find locations of remaining mills - by comparing to FRS. NAICS Codes starting with 321 and 322
    # read in FRS data to get location information for landfills with missing location information 
    FRS_facility_locs = pd.read_csv(frs_facility_path,low_memory=False)
    FRS_facility_locs = pd.read_csv(frs_facility_path, usecols = [2,3,5,7,8,10,17,20,21,26,27,28,31,32,34,35,36],low_memory=False)
    FRS_facility_locs.fillna(0, inplace = True)
    FRS_facility_locs = FRS_facility_locs[FRS_facility_locs['LATITUDE83'] > 0]
    FRS_facility_locs = FRS_facility_locs[FRS_facility_locs['NAICS_CODES'] != 0]
    FRS_facility_locs.reset_index(inplace=True, drop=True)

    FRS_facility_locs = FRS_facility_locs[(FRS_facility_locs['NAICS_CODES'].str.startswith('321')) | (FRS_facility_locs['NAICS_CODES'].str.startswith('322'))]

    print('Total FRS Pulp & Paper Facilities: ',len(FRS_facility_locs))
    FRS_facility_locs.reset_index(inplace=True, drop=True)

    FRS_facility_locs['CITY_NAME'] = FRS_facility_locs['CITY_NAME'].replace(0,'NaN')
    FRS_facility_locs.reset_index(inplace=True, drop=True)

    FRS_facility_locs.head()

    #try to match mills database with FRS database, based on state and city - 
    # only need to find locations for where 'ghgrp_match' = 0
    mills_locs['FRS_match'] = 0
    for ifacility in np.arange(0, len(mills_locs)):
        if mills_locs.loc[ifacility,'ghgrp_match'] == 0:
            imatch = np.where((mills_locs.loc[ifacility,'state_code'] == FRS_facility_locs['STATE_CODE']) &\
                            ((mills_locs.loc[ifacility,'city'].upper() == FRS_facility_locs['CITY_NAME'])))[0]
            if len(imatch)==1:
                mills_locs.loc[ifacility,'lat'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
                mills_locs.loc[ifacility,'lon'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
                mills_locs.loc[ifacility,'FRS_match'] = 1
            elif len(imatch)>1:
                FRS_temp = FRS_facility_locs.loc[imatch,:]
                new_match = np.where(np.max(FRS_temp['ACCURACY_VALUE']))[0]
                if len(new_match) >0:
                    mills_locs.loc[ifacility,'lat'] = FRS_facility_locs.loc[imatch[new_match[0]],'LATITUDE83']
                    mills_locs.loc[ifacility,'lon'] = FRS_facility_locs.loc[imatch[new_match[0]],'LONGITUDE83']
                    mills_locs.loc[ifacility,'FRS_match'] = 1
            else:
                continue

    print('Not Found:',len(mills_locs)-(np.sum(mills_locs.loc[:,'FRS_match'])+np.sum(mills_locs.loc[:,'ghgrp_match'])), 'of',len(mills_locs))

    # keep mills that only have FRS locations (mills that are not already covered by
    # subpart tt and mills that are not missing locations)
    mills_locs = mills_locs.query(
        "ghgrp_match == 0 & FRS_match == 1").drop(
            columns=["state_name", "county", "city", "grades", "ghgrp_match", "FRS_match"]).rename(
                columns={"lat": "latitude", "lon": "longitude"})
    
    # add a column to equally allocate unaccounted for GHGI emissions to all non-reporting mills
    mills_locs["ch4_kt"] = 1.0

    nonreporting_pulp_paper_gdf = (
        gpd.GeoDataFrame(
            mills_locs,
            geometry=gpd.points_from_xy(
                mills_locs["longitude"],
                mills_locs["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["latitude", "longitude"])
        .loc[:, ["facility_id", "state_code", "geometry", "ch4_kt"]]
    )

    return nonreporting_pulp_paper_gdf


def get_reporting_food_beverage_proxy_data(
    state_geo_path,
    subpart_tt_path,
):

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
    # Reporting facilities from subpart tt
    reporting_food_beverage_df = (
    pd.read_csv(
        subpart_tt_path,
        usecols=("facility_name",
                    "facility_id",
                    "reporting_year",
                    "ghg_quantity",
                    "latitude",
                    "longitude",
                    "state",
                    "city",
                    "zip",
                    "naics_code"))
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"reporting_year": "year", "ghg_quantity": "ch4_t", "state": "state_code"})
    .assign(ch4_kt=lambda df: df["ch4_t"] * t_to_kt)
    .drop(columns=["ch4_t"])
    .drop_duplicates(subset=['facility_id', 'year'], keep='first')
    .astype({"year": int})
    # .query("year.between(@min_year, @max_year)")
    .query("year.between(@min_year, 2021)")
    .astype({"naics_code": int})
    .query("naics_code == 311612|naics_code == 311421|naics_code == 311513|naics_code == 312140|naics_code == 311611|naics_code == 311615|naics_code == 311225|naics_code == 311613|naics_code == 311710|naics_code == 311221|naics_code == 311224|naics_code == 311314|naics_code == 311313") 
    .drop(columns=["naics_code"])
    .query("state_code.isin(@state_info_df['state_code'])")
    .reset_index(drop=True)
    )

    reporting_food_beverage_gdf = (
        gpd.GeoDataFrame(
            reporting_food_beverage_df,
            geometry=gpd.points_from_xy(
                reporting_food_beverage_df["longitude"],
                reporting_food_beverage_df["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["latitude", "longitude", "city", "zip"])
        .loc[:, ["facility_id", "facility_name", "state_code", "geometry", "year", "ch4_kt"]]
    )

    return reporting_food_beverage_gdf


def get_nonreporting_food_beverage_proxy_data(
    subpart_tt_path,
    food_manufacturers_processors_path,
    frs_facility_path,
    state_geo_path,
):
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
    # Reporting facilities from subpart tt
    reporting_food_beverage_df = (
    pd.read_csv(
        subpart_tt_path,
        usecols=("facility_name",
                    "facility_id",
                    "reporting_year",
                    "ghg_quantity",
                    "latitude",
                    "longitude",
                    "state",
                    "city",
                    "zip",
                    "naics_code"))
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"reporting_year": "year", "ghg_quantity": "ch4_t", "state": "state_code"})
    .assign(ch4_kt=lambda df: df["ch4_t"] * t_to_kt)
    .drop(columns=["ch4_t"])
    .drop_duplicates(subset=['facility_id', 'year'], keep='first')
    .astype({"year": int})
    # .query("year.between(@min_year, @max_year)")
    .query("year.between(@min_year, 2021)")
    .astype({"naics_code": int})
    .query("naics_code == 311612|naics_code == 311421|naics_code == 311513|naics_code == 312140|naics_code == 311611|naics_code == 311615|naics_code == 311225|naics_code == 311613|naics_code == 311710|naics_code == 311221|naics_code == 311224|naics_code == 311314|naics_code == 311313") 
    .drop(columns=["naics_code"])
    .query("state_code.isin(@state_gdf['state_code'])")
    .reset_index(drop=True)
    )

    # list of food and beverage facilities
    food_beverage_facilities_df = (
    pd.read_excel(
        food_manufacturers_processors_path,
        sheet_name='Data',
        nrows=59915,
        usecols="A:K",
    )
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"name": "facility_name", "state": "state_code", "uniqueid": "facility_id"})
    .drop(columns=["naics_code_description"])
    # filter for relevant NAICS codes (based on GHGI)
    .query("naics_code == 311612|naics_code == 311421|naics_code == 311513|naics_code == 312140|naics_code == 311611|naics_code == 311615|naics_code == 311225|naics_code == 311613|naics_code == 311710|naics_code == 311221|naics_code == 311224|naics_code == 311314|naics_code == 311313") 
    .drop(columns=["naics_code"])
    # remove facilities that do not report excess food waste
    .dropna(subset=['excessfood_tonyear_lowest'])
    # get only lower 48 + DC
    # .query("state_code.isin(@state_info_df['state_code'])")
    .fillna({'address': '', 'city': '', 'county': ''})
    # create full address for location matching
    .assign(full_address=lambda df: df["address"] + ' ' + df["city"] + ' ' + df["county"] + ' ' + df["state_code"] + ' ' + df["zip_code"].astype(str))
    # create partial address for location matching
    .assign(partial_address=lambda df: df["city"] + ' ' + df["county"] + ' ' + df["state_code"] + ' ' + df["zip_code"].astype(str))
    # calculate the average food waste (tonnes) at each facility
    .assign(avg_waste_t=lambda df: (df["excessfood_tonyear_lowest"] + df["excessfood_tonyear_highest"]) * 0.5)
    .assign(lat=0)
    .assign(lon=0)
    .reset_index(drop=True)
    )

    # try to match facilities to GHGRP based on county and city

    # match subpart tt reporting facilities to the food and beverage facility list 
    # and pull out non-reporting facilities
    food_beverage_facilities_locs = food_beverage_facilities_df.copy()
    num_facilities = len(food_beverage_facilities_locs)
    food_beverage_facilities_locs.loc[:,'ghgrp_match'] = 0

    reporting_food_beverage_df.loc[:,'found'] = 0
    reporting_food_beverage_df.loc[:,'city'] = reporting_food_beverage_df.loc[:,'city'].str.lower()

    for iyear in np.arange(0, num_years):
        for ifacility in np.arange(0,num_facilities):
            imatch = np.where((reporting_food_beverage_df['year'] == year_range[iyear]) & \
                            (reporting_food_beverage_df['state_code'] == food_beverage_facilities_locs.loc[ifacility,'state_code']) & \
                            (reporting_food_beverage_df['city'] == food_beverage_facilities_locs.loc[ifacility,'city']))[0]
                            # (reporting_pulp_paper_df['city'].str.contains(food_beverage_facilities_locs.loc[ifacility,'city'].upper())))[0]
            if len(imatch) > 0:
                food_beverage_facilities_locs.loc[ifacility,'ghgrp_match'] = 1
                food_beverage_facilities_locs.loc[ifacility,'lat'] = reporting_food_beverage_df.loc[imatch[0],'latitude']
                food_beverage_facilities_locs.loc[ifacility,'lon'] = reporting_food_beverage_df.loc[imatch[0],'longitude']
                reporting_food_beverage_df.loc[imatch[0],'found'] = 1
            else:
                continue

        print('Found (%) Year',year_range[iyear],':',100*np.sum(food_beverage_facilities_locs['ghgrp_match']/len(food_beverage_facilities_locs)))

    # for iyear in np.arange(0, num_years):
    #     for ifacility in np.arange(0,len(food_beverage_facilities_locs)):
    #         imatch = np.where((ghgrp_ind['Year'] == year_range[iyear]) & \
    #                         (ghgrp_ind['STATE'] == food_beverage_facilities_locs.loc[ifacility,'STATE']) & \
    #                         (ghgrp_ind['COUNTY'].str.contains(food_beverage_facilities_locs.loc[ifacility,'COUNTY'].upper())))[0]
    #         if len(imatch)==1:
    #             food_beverage_facilities_locs.loc[ifacility,'ghgrp_match'] = 1
    #             food_beverage_facilities_locs.loc[ifacility,'Lat'] = ghgrp_ind.loc[imatch[0],'LATITUDE']
    #             food_beverage_facilities_locs.loc[ifacility,'Lon'] = ghgrp_ind.loc[imatch[0],'LONGITUDE']
    #             food_beverage_facilities_locs.loc[ifacility,'emi_kt_'+year_range_str[iyear]] = ghgrp_ind.loc[imatch[0],'emis_kt_tot']
    #             ghgrp_ind.loc[imatch[0],'found'] = 1
    #         if len(imatch) > 1:
    #             new_match = np.where((ghgrp_ind['Year'] == year_range[iyear]) & \
    #                                 (ghgrp_ind['STATE'] == food_beverage_facilities_locs.loc[ifacility,'STATE']) &\
    #                                 (ghgrp_ind['COUNTY'].str.contains(food_beverage_facilities_locs.loc[ifacility,'COUNTY'].upper())) &\
    #                             (ghgrp_ind['CITY'].str.contains(food_beverage_facilities_locs.loc[ifacility,'CITY'].upper())))[0]
    #             if len(new_match)>0:
    #                 food_beverage_facilities_locs.loc[ifacility,'ghgrp_match'] = 1
    #                 food_beverage_facilities_locs.loc[ifacility,'Lat'] = ghgrp_ind.loc[new_match[0],'LATITUDE']
    #                 food_beverage_facilities_locs.loc[ifacility,'Lon'] = ghgrp_ind.loc[new_match[0],'LONGITUDE']
    #                 food_beverage_facilities_locs.loc[ifacility,'emi_kt_'+year_range_str[iyear]] = ghgrp_ind.loc[new_match[0],'emis_kt_tot']
    #                 ghgrp_ind.loc[new_match[0],'found'] = 1
                
    #         else:
    #             continue
    #     print('Found (%) Year',year_range[iyear],':',100*np.sum(food_beverage_facilities_locs['ghgrp_match']/len(food_beverage_facilities_locs)))
    

    # find locations of remaining mills - by comparing to FRS. NAICS Codes starting with 321 and 322
    # read in FRS data to get location information for landfills with missing location information 
    FRS_facility_locs = pd.read_csv(frs_facility_path,low_memory=False)
    FRS_facility_locs = pd.read_csv(frs_facility_path, usecols = [2,3,5,7,8,10,17,20,21,26,27,28,31,32,34,35,36],low_memory=False)
    FRS_facility_locs.fillna(0, inplace = True)
    FRS_facility_locs = FRS_facility_locs[FRS_facility_locs['LATITUDE83'] > 0]
    FRS_facility_locs = FRS_facility_locs[FRS_facility_locs['NAICS_CODES'] != 0]
    FRS_facility_locs.reset_index(inplace=True, drop=True)

    FRS_facility_locs = FRS_facility_locs[(FRS_facility_locs['NAICS_CODES'].str.startswith('321')) | (FRS_facility_locs['NAICS_CODES'].str.startswith('322'))]

    print('Total FRS Pulp & Paper Facilities: ',len(FRS_facility_locs))
    FRS_facility_locs.reset_index(inplace=True, drop=True)

    FRS_facility_locs['CITY_NAME'] = FRS_facility_locs['CITY_NAME'].replace(0,'NaN')
    FRS_facility_locs.reset_index(inplace=True, drop=True)

    FRS_facility_locs.head()

    #try to match mills database with FRS database, based on state and city - 
    # only need to find locations for where 'ghgrp_match' = 0
    mills_locs['FRS_match'] = 0
    for ifacility in np.arange(0, len(mills_locs)):
        if mills_locs.loc[ifacility,'ghgrp_match'] == 0:
            imatch = np.where((mills_locs.loc[ifacility,'state_code'] == FRS_facility_locs['STATE_CODE']) &\
                            ((mills_locs.loc[ifacility,'city'].upper() == FRS_facility_locs['CITY_NAME'])))[0]
            if len(imatch)==1:
                mills_locs.loc[ifacility,'lat'] = FRS_facility_locs.loc[imatch[0],'LATITUDE83']
                mills_locs.loc[ifacility,'lon'] = FRS_facility_locs.loc[imatch[0],'LONGITUDE83']
                mills_locs.loc[ifacility,'FRS_match'] = 1
            elif len(imatch)>1:
                FRS_temp = FRS_facility_locs.loc[imatch,:]
                new_match = np.where(np.max(FRS_temp['ACCURACY_VALUE']))[0]
                if len(new_match) >0:
                    mills_locs.loc[ifacility,'lat'] = FRS_facility_locs.loc[imatch[new_match[0]],'LATITUDE83']
                    mills_locs.loc[ifacility,'lon'] = FRS_facility_locs.loc[imatch[new_match[0]],'LONGITUDE83']
                    mills_locs.loc[ifacility,'FRS_match'] = 1
            else:
                continue

    print('Not Found:',len(mills_locs)-(np.sum(mills_locs.loc[:,'FRS_match'])+np.sum(mills_locs.loc[:,'ghgrp_match'])), 'of',len(mills_locs))

    # keep mills that only have FRS locations (mills that are not already covered by
    # subpart tt and mills that are not missing locations)
    mills_locs = mills_locs.query(
        "ghgrp_match == 0 & FRS_match == 1").drop(
            columns=["state_name", "county", "city", "grades", "ghgrp_match", "FRS_match"]).rename(
                columns={"lat": "latitude", "lon": "longitude"})
    
    # add a column to equally allocate unaccounted for GHGI emissions to all non-reporting mills
    mills_locs["ch4_kt"] = 1.0

    nonreporting_pulp_paper_gdf = (
        gpd.GeoDataFrame(
            mills_locs,
            geometry=gpd.points_from_xy(
                mills_locs["longitude"],
                mills_locs["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["latitude", "longitude"])
        .loc[:, ["facility_id", "state_code", "geometry", "ch4_kt"]]
    )

    return nonreporting_pulp_paper_gdf



# https://www.epa.gov/system/files/documents/2024-02/us-ghg-inventory-2024-main-text.pdf
IPCC_ID = "5A1"
SECTOR_NAME = "waste"
SOURCE_NAME = "industrial landfills"
FULL_NAME = "_".join([IPCC_ID, SECTOR_NAME, SOURCE_NAME]).replace(" ", "_")

netcdf_title = f"EPA methane emissions from {SOURCE_NAME}"
netcdf_description = (
    f"Gridded EPA Inventory - {SECTOR_NAME} - "
    f"{SOURCE_NAME} - IPCC Source Category {IPCC_ID}"
)

# PATHS
landfills_dir = ghgi_data_dir_path / "landfills"
sector_data_dir_path = V3_DATA_PATH / "sector"

# OUTPUT FILES
ch4_kt_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_kt_per_year"
ch4_flux_dst_path = tmp_data_dir_path / f"{FULL_NAME}_ch4_emi_flux"

# INVENTORY INPUT FILE
inventory_workbook_path = landfills_dir / "State_IND_LF_1990-2021.xlsx"

# PROXY INPUT FILES
# state vector refence with state_code
state_geo_path = global_data_dir_path / "tl_2020_us_state.zip"
# input 1: subpart tt - industrial waste landfills
subpart_tt_path = "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV"
# input 2: mills online data
mills_online_path = sector_data_dir_path / "landfills" / "Mills_OnLine.xlsx"
# input 3: frs facilities where NAICS code is for pulp & paper
frs_naics_path = global_data_dir_path / "NATIONAL_NAICS_FILE.CSV"
frs_facility_path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV"
# input 4: epa food opportunities map food and beverage manufacturing data
food_manufacturers_processors_path = sector_data_dir_path / "landfills" / "Food Manufacturers and Processors.xlsx"

# the spatial tolerance for removing duplicate facility points
DUPLICATION_TOLERANCE_M = 250
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

EPA_state_emi_df = get_composting_inventory_data(inventory_workbook_path)

# plot state inventory data
sns.relplot(
    kind="line",
    data=EPA_state_emi_df,
    x="year",
    y="ch4_kt",
    hue="state_code",
    palette="tab20",
    legend=False,
)

# %% STEP 3: GET AND FORMAT PROXY DATA -------------------------------------------------
composting_proxy_gdf = get_composting_proxy_data(
    excess_food_op_path,
    frs_facility_path,
    frs_naics_path,
    biocycle_path,
    comp_council_path,
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