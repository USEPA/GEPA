"""
Name:                   task_industrial_landfills_proxy.py
Date Last Modified:     2025-04-02
Authors Name:           H. Lohman (RTI International)
Purpose:                Mapping of industrial landfills reporting and non-reporting
                        food & beverage and pulp & paper proxy emissions
gch4i_name:             5A_industrial_landfills
Input Files:            State Geo: global_data_dir_path / "tl_2020_us_state.zip"
                        GHGRP Subpart TT: "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV"
                        FRS NAICS Codes: global_data_dir_path / "NATIONAL_NAICS_FILE.CSV"
                        FRS Facilities: global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV"
                        Mills OnLine: sector_data_dir_path / "landfills/Mills_OnLine.xlsx"
                        EPA Food Opportunities Map: sector_data_dir_path / "landfills/Food Manufacturers and Processors.xlsx"
Output Files:           - proxy_data_dir_path / "ind_landfills_pp_r_proxy.parquet"
                        - proxy_data_dir_path / "ind_landfills_pp_nr_proxy.parquet"
                        - proxy_data_dir_path / "ind_landfills_fb_r_proxy.parquet"
                        - proxy_data_dir_path / "ind_landfills_fb_nr_proxy.parquet"
"""

# %%
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile
import calendar
import datetime

from pyarrow import parquet
import pandas as pd
import osgeo
import geopandas as gpd
import numpy as np
import seaborn as sns
from pytask import Product, task, mark
import geopy
from geopy.geocoders import Nominatim
import duckdb

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    ghgi_data_dir_path,
    max_year,
    min_year,
)

from gch4i.utils import name_formatter

tg_to_kt = 0.001
year_range = [*range(min_year, max_year+1,1)] #List of emission years
year_range_str=[str(i) for i in year_range]
num_years = len(year_range)

# %%
@mark.persist
@task(id="industrial_landfills_proxy")
def task_get_reporting_industrial_landfills_pulp_paper_proxy_data(
    subpart_tt_path = "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    reporting_pulp_paper_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "ind_landfills_pp_r_proxy.parquet",
):
    """
    Relative emissions and location information for reporting facilities are taken from 
    the Subpart TT database with NAICS codes starting with 321 and 322.
    """

    # Get state vectors and state_code for use with inventory and proxy data
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
        .assign(ch4_kt=lambda df: df["ch4_t"] * tg_to_kt)
        .drop(columns=["ch4_t"])
        .drop_duplicates(subset=['facility_id', 'year'], keep='last')
        .astype({"year": int})
        .query("year.between(@min_year, @max_year)")
        .astype({"naics_code": str})
        .query("naics_code.str.startswith('321') | naics_code.str.startswith('322')" )
        .drop(columns=["naics_code"])
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
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
        .drop(columns=["facility_id", "latitude", "longitude", "city", "zip"])
        .loc[:, ["facility_name", "state_code", "geometry", "year", "ch4_kt"]]
    )

    reporting_pulp_paper_gdf['rel_emi'] = reporting_pulp_paper_gdf.groupby(["state_code", "year"])['ch4_kt'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    reporting_pulp_paper_gdf = reporting_pulp_paper_gdf.drop(columns='ch4_kt')

    reporting_pulp_paper_gdf.to_parquet(reporting_pulp_paper_proxy_output_path)
    return None


def task_get_nonreporting_industrial_landfills_pulp_paper_proxy_data(
            state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
            subpart_tt_path = "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV",
            frs_naics_path = global_data_dir_path / "NATIONAL_NAICS_FILE.CSV",
            frs_facility_path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV",
            mills_online_path: Path = V3_DATA_PATH / "sector/landfills/Mills_OnLine.xlsx",
            nonreporting_pulp_paper_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path
            / "ind_landfills_pp_nr_proxy.parquet",
):
    """
    The Mills OnLine database of facilities is compared against the Subpart HH 
    reporting facilities with NAICS codes starting with 321 and 322 to develop 
    a list of nonreporting facilities (facilities that appear in the Mills OnLine 
    database but not the Subpart HH list).
    
    Facility locations (lat/lon) are determined by matching the nonreporting facility cities and states
    to the facilities in the FRS database.

    Nonreporting emissions are evenly distributed across each facility.
    """

    # Get state vectors and state_code for use with inventory and proxy data
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
    
    # Reporting facilities from subpart tt with NAICS codes that start with 321 and 322
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
        .assign(ch4_kt=lambda df: df["ch4_t"] * tg_to_kt)
        .drop(columns=["ch4_t"])
        .drop_duplicates(subset=['facility_id', 'year'], keep='last')
        .astype({"year": int})
        .query("year.between(@min_year, @max_year)")
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
        .query("state_name.isin(@state_gdf['state_name'])")
        .reset_index(drop=True)
    )

    # assign state codes to mills online facilities
    mills_locs = mills_online_df.copy()
    num_mills = len(mills_locs)

    for imill in np.arange(0,num_mills):
        state_name = mills_locs['state_name'][imill]
        state_code = state_gdf[state_gdf['state_name'] == state_name]['state_code']
        mills_locs.loc[imill, 'state_code'] = state_code.to_string(index=False)

    # match subpart tt reporting facilities to the mills online facility list 
    # and pull out non-reporting facilities
    mills_locs.loc[:, 'ghgrp_match'] = 0
    mills_locs.loc[:, 'lat'] = 0
    mills_locs.loc[:, 'lon'] = 0
    mills_locs.loc[:, 'city'] = mills_locs.loc[:, 'city'].str.lower()

    reporting_pulp_paper_df.loc[:, 'found'] = 0
    reporting_pulp_paper_df.loc[:, 'city'] = reporting_pulp_paper_df.loc[:, 'city'].str.lower()

    # try to match facilities to GHGRP based on county and city
    for iyear in np.arange(0, num_years):
        for ifacility in np.arange(0,num_mills):
            imatch = np.where((reporting_pulp_paper_df['year'] == year_range[iyear]) & \
                            (reporting_pulp_paper_df['state_code'] == mills_locs.loc[ifacility,'state_code']) & \
                            (reporting_pulp_paper_df['city'] == mills_locs.loc[ifacility,'city']))[0]
            if len(imatch) > 0:
                mills_locs.loc[ifacility,'ghgrp_match'] = 1
                mills_locs.loc[ifacility,'lat'] = reporting_pulp_paper_df.loc[imatch[0],'latitude']
                mills_locs.loc[ifacility,'lon'] = reporting_pulp_paper_df.loc[imatch[0],'longitude']
                reporting_pulp_paper_df.loc[imatch[0],'found'] = 1
            else:
                continue

        print('Found (%) Year',year_range[iyear],':',100*np.sum(mills_locs['ghgrp_match']/len(reporting_pulp_paper_df)))
    

    # FRS facilities with NAICS codes that start with 321 and 322
    frs_main = (
        pd.read_csv(
            frs_facility_path,
            usecols=("REGISTRY_ID", "PRIMARY_NAME", "LOCATION_ADDRESS", "CITY_NAME", "STATE_CODE", "ACCURACY_VALUE", "LATITUDE83", "LONGITUDE83"
            ))
            .rename(columns=lambda x: str(x).lower())
            .rename(columns={"primary_name": "name", "latitude83": "latitude", "longitude83": "longitude"})
            .assign(formatted_fac_name=lambda df: name_formatter(df["name"]), source="frs")
    )
    frs_main['city_name'] = frs_main['city_name'].replace('NO LOCALITY NAME TO MIGRATE FROM 2002 NEI V3', 'NaN')

    frs_naics = (
        pd.read_csv(
            frs_naics_path,
            usecols=("REGISTRY_ID", "NAICS_CODE"
            ))
            .rename(columns=lambda x: str(x).lower())
            .astype({"naics_code": str})
    )
    frs_naics = frs_naics[(frs_naics['naics_code'].str.startswith('321')) | (frs_naics['naics_code'].str.startswith('322'))].reset_index(drop=True)

    FRS_facility_locs = frs_naics.merge(frs_main, how='inner', on='registry_id')

    #try to match mills database with FRS database, based on state and city - 
    # only need to find locations for where 'ghgrp_match' = 0
    mills_locs['FRS_match'] = 0
    for ifacility in np.arange(0, len(mills_locs)):
        if mills_locs.loc[ifacility,'ghgrp_match'] == 0:
            imatch = np.where((mills_locs.loc[ifacility,'state_code'] == FRS_facility_locs['state_code']) &\
                            ((mills_locs.loc[ifacility,'city'].upper() == FRS_facility_locs['city_name'])))[0]
            if len(imatch)==1:
                mills_locs.loc[ifacility,'lat'] = FRS_facility_locs.loc[imatch[0],'latitude']
                mills_locs.loc[ifacility,'lon'] = FRS_facility_locs.loc[imatch[0],'longitude']
                mills_locs.loc[ifacility,'FRS_match'] = 1
            elif len(imatch)>1:
                FRS_temp = FRS_facility_locs.loc[imatch,:]
                new_match = np.where(np.max(FRS_temp['accuracy_value']))[0]
                if len(new_match) >0:
                    mills_locs.loc[ifacility,'lat'] = FRS_facility_locs.loc[imatch[new_match[0]],'latitude']
                    mills_locs.loc[ifacility,'lon'] = FRS_facility_locs.loc[imatch[new_match[0]],'longitude']
                    mills_locs.loc[ifacility,'FRS_match'] = 1
            else:
                continue

    print('Not Found:',len(mills_locs)-(np.sum(mills_locs.loc[:,'FRS_match'])+np.sum(mills_locs.loc[:,'ghgrp_match'])), 'of',len(mills_locs))

    # keep mills that only have FRS locations (mills that are not already covered by
    # subpart tt and mills that are not missing locations)
    mills_locs = mills_locs.query(
        "ghgrp_match == 0 & FRS_match == 1").drop(
            columns=["state_name", "county", "city", "grades", "ghgrp_match", "FRS_match"]).rename(
                columns={"lat": "latitude", "lon": "longitude"}).dropna()
    
    # add a column to equally allocate unaccounted for GHGI emissions to all non-reporting mills
    mills_locs["ch4_kt"] = 1.0

    mills_locs = mills_locs.reset_index(drop=True)

    nonreporting_pulp_paper_gdf = (
        gpd.GeoDataFrame(
            mills_locs,
            geometry=gpd.points_from_xy(
                mills_locs["longitude"],
                mills_locs["latitude"],
                crs=4326,
            ),
        )
        .drop(columns=["facility_id", "latitude", "longitude"])
        .loc[:, ["state_code", "geometry", "ch4_kt"]]
    )

    nonreporting_pulp_paper_gdf['rel_emi'] = nonreporting_pulp_paper_gdf.groupby(["state_code"])['ch4_kt'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    nonreporting_pulp_paper_gdf = nonreporting_pulp_paper_gdf.drop(columns='ch4_kt')

    nonreporting_pulp_paper_gdf.to_parquet(nonreporting_pulp_paper_proxy_output_path)
    return None


def task_get_reporting_industrial_landfills_food_beverage_proxy_data(
    subpart_tt_path = "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    reporting_food_beverage_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "ind_landfills_fb_r_proxy.parquet",
):
    """
    Relative emissions and location information for reporting facilities are taken from 
    the Subpart TT database with NAICS codes: 311612, 311421, 311513, 312140, 311611,
    311615, 311225, 311613, 311710, 311221, 311224, 311314, 311313.
    """

    # Get state vectors and state_code for use with inventory and proxy data
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
    .assign(ch4_kt=lambda df: df["ch4_t"] * tg_to_kt)
    .drop(columns=["ch4_t"])
    .drop_duplicates(subset=['facility_id', 'year'], keep='first')
    .astype({"year": int})
    .query("year.between(@min_year, @max_year)")
    .astype({"naics_code": int})
    .query("naics_code == 311612|naics_code == 311421|naics_code == 311513|naics_code == 312140|naics_code == 311611|naics_code == 311615|naics_code == 311225|naics_code == 311613|naics_code == 311710|naics_code == 311221|naics_code == 311224|naics_code == 311314|naics_code == 311313") 
    .drop(columns=["naics_code"])
    .query("state_code.isin(@state_gdf['state_code'])")
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

    reporting_food_beverage_gdf['rel_emi'] = reporting_food_beverage_gdf.groupby(["state_code"])['ch4_kt'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    reporting_food_beverage_gdf = reporting_food_beverage_gdf.drop(columns='ch4_kt')
    
    reporting_food_beverage_gdf.to_parquet(reporting_food_beverage_proxy_output_path)
    return None


def task_get_nonreporting_industrial_landfills_food_beverage_proxy_data(
            state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
            subpart_tt_path = "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV",
            frs_naics_path = global_data_dir_path / "NATIONAL_NAICS_FILE.CSV",
            frs_facility_path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV",
            food_manufacturers_processors_path = V3_DATA_PATH / "sector/landfills/Food Manufacturers and Processors.xlsx",
            nonreporting_food_beverage_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path
            / "ind_landfills_fb_nr_proxy.parquet",
):
    """

    Facilities in the EPA Food Opportunities Map are assumed to be the food and
    beverage facilities contributing to industrial waste emissions. Facilities are
    filtered to include NAICS codes: 311612, 311421, 311513, 312140, 311611,
    311615, 311225, 311613, 311710, 311221, 311224, 311314, 311313. Waste-in-place
    at each facility is assumed to be the proxy for methane emissions.

    The EPA Food Opportunities Map facilities are compared against the Subpart HH 
    reporting facilities to develop a list of nonreporting facilities 
    (facilities that appear in the EPA Food Opportunities Map but not the 
    Subpart HH list).
    
    Facility locations (lat/lon) are determined by matching the nonreporting facility 
    cities and states to the facilities in the FRS database.

    Location information is geocoded for facilities without an FRS match.
    """

    # Get state vectors and state_code for use with inventory and proxy data
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
    .assign(ch4_kt=lambda df: df["ch4_t"] * tg_to_kt)
    .drop(columns=["ch4_t"])
    .drop_duplicates(subset=['facility_id', 'year'], keep='first')
    .astype({"year": int})
    .query("year.between(@min_year, @max_year)")
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
    food_beverage_facilities_locs.loc[:,'city'] = food_beverage_facilities_locs.loc[:,'city'].str.lower()

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

        print('Found (%) Year',year_range[iyear],':',100*np.sum(food_beverage_facilities_locs['ghgrp_match']/len(reporting_food_beverage_df)))
    
    # Read in FRS data to get location information for landfills with missing location 
    # information. FRS facilities with NAICS codes: 311612, 311421, 311513, 312140, 311611,
    # 311615, 311225, 311613, 311710, 311221, 311224, 311314, 311313.
    frs_main = (
        pd.read_csv(
            frs_facility_path,
            usecols=("REGISTRY_ID", "PRIMARY_NAME", "LOCATION_ADDRESS", "CITY_NAME", "STATE_CODE", "ACCURACY_VALUE", "LATITUDE83", "LONGITUDE83"
            ))
            .rename(columns=lambda x: str(x).lower())
            .rename(columns={"primary_name": "name", "latitude83": "latitude", "longitude83": "longitude"})
            .assign(formatted_fac_name=lambda df: name_formatter(df["name"]), source="frs")
    )
    frs_main['city_name'] = frs_main['city_name'].replace('NO LOCALITY NAME TO MIGRATE FROM 2002 NEI V3', 'NaN')

    frs_naics = (
        pd.read_csv(
            frs_naics_path,
            usecols=("REGISTRY_ID", "NAICS_CODE"
            ))
            .rename(columns=lambda x: str(x).lower())
            .astype({"naics_code": str})
    )
    frs_naics = frs_naics[(frs_naics['naics_code'] == '311612') | 
                        (frs_naics['naics_code'] == '311421') | 
                        (frs_naics['naics_code'] == '311513') | 
                        (frs_naics['naics_code'] == '312140') |
                        (frs_naics['naics_code'] == '311611') | 
                        (frs_naics['naics_code'] == '311615') | 
                        (frs_naics['naics_code'] == '311225') | 
                        (frs_naics['naics_code'] == '311613') | 
                        (frs_naics['naics_code'] == '311710') | 
                        (frs_naics['naics_code'] == '311221') | 
                        (frs_naics['naics_code'] == '311224') | 
                        (frs_naics['naics_code'] == '311314') | 
                        (frs_naics['naics_code'] == '311313')].reset_index(drop=True)
    FRS_facility_locs = frs_naics.merge(frs_main, how='inner', on='registry_id')

    FRS_facility_locs = FRS_facility_locs.query('latitude > 0').reset_index(drop=True)

    #try to match nonreporting facilities with FRS database, based on state and city - 
    # only need to find locations for where 'ghgrp_match' = 0
    food_beverage_facilities_locs['FRS_match'] = 0
    for ifacility in np.arange(0, len(food_beverage_facilities_locs)):
        if food_beverage_facilities_locs.loc[ifacility,'ghgrp_match'] == 0:
            imatch = np.where((food_beverage_facilities_locs.loc[ifacility,'state_code'] == FRS_facility_locs['state_code']) &\
                            ((food_beverage_facilities_locs.loc[ifacility,'city'].upper() == FRS_facility_locs['city_name'])))[0]
            if len(imatch)==1:
                food_beverage_facilities_locs.loc[ifacility,'lat'] = FRS_facility_locs.loc[imatch[0],'latitude']
                food_beverage_facilities_locs.loc[ifacility,'lon'] = FRS_facility_locs.loc[imatch[0],'longitude']
                food_beverage_facilities_locs.loc[ifacility,'FRS_match'] = 1
            elif len(imatch)>1:
                FRS_temp = FRS_facility_locs.loc[imatch,:]
                new_match = np.where(np.max(FRS_temp['accuracy_value']))[0]
                if len(new_match) >0:
                    food_beverage_facilities_locs.loc[ifacility,'lat'] = FRS_facility_locs.loc[imatch[new_match[0]],'latitude']
                    food_beverage_facilities_locs.loc[ifacility,'lon'] = FRS_facility_locs.loc[imatch[new_match[0]],'longitude']
                    food_beverage_facilities_locs.loc[ifacility,'FRS_match'] = 1
            else:
                continue

    print('Not Found:',len(food_beverage_facilities_locs)-(np.sum(food_beverage_facilities_locs.loc[:,'FRS_match'])+np.sum(food_beverage_facilities_locs.loc[:,'ghgrp_match'])), 'of',len(food_beverage_facilities_locs))

    # Find missing locations by Geocoding addresses
    geolocator = Nominatim(user_agent="myGeocode")
    geopy.geocoders.options.default_timeout = None
    print(geolocator.timeout)

    food_beverage_facilities_locs['geo_match'] = 0
    for ifacility in np.arange(0,len(food_beverage_facilities_locs)):
        if food_beverage_facilities_locs.loc[ifacility,'FRS_match'] ==0 and food_beverage_facilities_locs.loc[ifacility,'ghgrp_match'] == 0:
            location = geolocator.geocode(food_beverage_facilities_locs['full_address'][ifacility])
            if location is None:
                continue
            else:
                food_beverage_facilities_locs.loc[ifacility,'lat'] = location.latitude
                food_beverage_facilities_locs.loc[ifacility,'lon'] = location.longitude
                food_beverage_facilities_locs.loc[ifacility,'geo_match']=1
                
    print('First Try - Percentage found:',(np.sum(food_beverage_facilities_locs['ghgrp_match'])+\
                                            np.sum(food_beverage_facilities_locs['FRS_match'])+\
                                            np.sum(food_beverage_facilities_locs['geo_match']))/len(food_beverage_facilities_locs))

    for ifacility in np.arange(0,len(food_beverage_facilities_locs)):
        if food_beverage_facilities_locs.loc[ifacility,'FRS_match'] ==0 and food_beverage_facilities_locs.loc[ifacility,'ghgrp_match'] == 0 and \
            food_beverage_facilities_locs.loc[ifacility,'geo_match']==0:
            #remove suite information from the address and try again 
            address_temp = food_beverage_facilities_locs['full_address'][ifacility]
            address_temp = address_temp.replace('Ste','')
            address_temp = address_temp.replace('Apt','')
            address_temp = address_temp.replace('Unit','')
            address_temp = address_temp.replace('Bldg','')
            location = geolocator.geocode(address_temp)
            if location is None:
                #if still no match, remove the address portion and just allocate based on city, state, county, zip
                #address_temp = food_beverage_facilities_locs.loc[ifacility,"Partial_Address"]
                address_temp = food_beverage_facilities_locs.loc[ifacility,"city"] +' '+\
                                            food_beverage_facilities_locs.loc[ifacility,"county"]+' '+\
                                            food_beverage_facilities_locs.loc[ifacility,"state_code"]+' '+\
                                            food_beverage_facilities_locs.loc[ifacility,"zip_code"].astype(str)
                location2 = geolocator.geocode(address_temp)
                if location2 is None:
                    #print(ifacility,address_temp)
                    continue
                else:
                    #count -= 1
                    food_beverage_facilities_locs.loc[ifacility,'lat'] = location2.latitude
                    food_beverage_facilities_locs.loc[ifacility,'lon'] = location2.longitude
                    food_beverage_facilities_locs.loc[ifacility,'geo_match']=1
            else:
                #count -= 1
                food_beverage_facilities_locs.loc[ifacility,'lat'] = location.latitude
                food_beverage_facilities_locs.loc[ifacility,'lon'] = location.longitude
                food_beverage_facilities_locs.loc[ifacility,'geo_match']=1
                
    print('Second Try - Percentage found:',(np.sum(food_beverage_facilities_locs['ghgrp_match'])+\
                                            np.sum(food_beverage_facilities_locs['FRS_match'])+\
                                            np.sum(food_beverage_facilities_locs['geo_match']))/len(food_beverage_facilities_locs))
    
    nonreporting_food_beverage_df = food_beverage_facilities_locs.query('FRS_match == 1 | geo_match == 1')

    nonreporting_food_beverage_gdf = (
        gpd.GeoDataFrame(
            nonreporting_food_beverage_df,
            geometry=gpd.points_from_xy(
                nonreporting_food_beverage_df["lon"],
                nonreporting_food_beverage_df["lat"],
                crs=4326,
            ),
        )
        .drop(columns=["facility_name", "address", "city", "county", "zip_code", 
                       "excessfood_tonyear_lowest", "excessfood_tonyear_highest", 
                       "full_address", "partial_address", "lat", "lon", 
                       "ghgrp_match", "FRS_match", "geo_match"])
        .loc[:, ["facility_id", "state_code", "geometry", "avg_waste_t"]]
    )
    nonreporting_food_beverage_gdf['rel_emi'] = nonreporting_food_beverage_gdf.groupby(["state_code", "year"])['avg_waste_t'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    nonreporting_food_beverage_gdf = nonreporting_food_beverage_gdf.drop(columns='avg_waste_t')

    nonreporting_food_beverage_gdf.to_parquet(nonreporting_food_beverage_proxy_output_path)
    return None
