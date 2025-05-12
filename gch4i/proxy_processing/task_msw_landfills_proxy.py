"""
Name:                   task_msw_landfills_proxy.py
Date Last Modified:     2025-04-02
Authors Name:           H. Lohman (RTI International)
Purpose:                Mapping of municipal solid waste (MSW) landfills reporting and
                        non-reporting proxy emissions
gch4i_name:             5A_msw_landfills
Input Files:            State Geo: global_data_dir_path / "tl_2020_us_state.zip"
                        GHGRP Subpart HH: "https://data.epa.gov/efservice/hh_subpart_level_information/pub_dim_facility/ghg_name/=/Methane/CSV"
                        FRS NAICS Codes: global_data_dir_path / "NATIONAL_NAICS_FILE.CSV"
                        FRS Facilities: global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV"
                        Non-Reporting Facilities: sector_data_dir_path / "landfills/Non-Reporting_LF_DB_2020_1.12.2021.xlsx"
Output Files:           - proxy_data_dir_path / "msw_landfills_r_proxy.parquet"
                        - proxy_data_dir_path / "msw_landfills_nr_proxy.parquet"
"""
# %% STEP 0. Load packages, configuration files, and local parameters ------------------
from pathlib import Path
from typing import Annotated

import pandas as pd
import geopandas as gpd
import numpy as np
from pytask import Product, task, mark
import geopy
from geopy.geocoders import Nominatim
import duckdb

from gch4i.config import (
    V3_DATA_PATH,
    emi_data_dir_path,
    proxy_data_dir_path,
    global_data_dir_path,
    max_year,
    min_year,
)

from gch4i.utils import name_formatter

#rename variable to 'mt_to_kt'
tg_to_kt = 0.001
year_range = [*range(min_year, max_year + 1, 1)]  # List of emission years
year_range_str = [str(i) for i in year_range]
num_years = len(year_range)

# %%
@mark.persist
@task(id="msw_landfills_proxy")
def task_get_reporting_msw_landfills_proxy_data(
    subpart_hh_path = "https://data.epa.gov/efservice/hh_subpart_level_information/pub_dim_facility/ghg_name/=/Methane/CSV",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    reporting_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / "msw_landfills_r_proxy.parquet",
):
    """
    Relative emissions and location information for reporting facilities are taken from 
    the Subpart HH database.
    """

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

    # Reporting facilities from Subpart HH
    reporting_facility_df = (
        pd.read_csv(
            subpart_hh_path,
            usecols=("facility_name",
                     "facility_id",
                     "reporting_year",
                     "ghg_quantity",
                     "latitude",
                     "longitude",
                     "state",
                     "zip")
                     )
                     .rename(columns=lambda x: str(x).lower())
                     .rename(columns={"reporting_year": "year", "ghg_quantity": "ch4_t", "state": "state_code"})
                     .assign(ch4_kt=lambda df: df["ch4_t"] * tg_to_kt)
                     .drop(columns=["ch4_t"])
                     .drop_duplicates(subset=['facility_id', 'year'], keep='last')
                     .astype({"year": int})
                     .query("year.between(@min_year, @max_year)")
                     .query("state_code.isin(@state_gdf['state_code'])")
                     .reset_index(drop=True)
                     )

    reporting_facility_df['rel_emi'] = reporting_facility_df.groupby(["state_code", "year"])['ch4_kt'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    reporting_facility_df = reporting_facility_df.drop(columns='ch4_kt')

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
        .loc[:, ["year", "state_code", "geometry", "rel_emi"]]
        .dropna(subset=['geometry'])
    )

    reporting_facility_gdf.to_parquet(reporting_proxy_output_path)
    return None


def get_nonreporting_msw_landfills_proxy_data(
    nonreporting_facility_path: Path = V3_DATA_PATH / "sector/landfills/Non-Reporting_LF_DB_2020_1.12.2021.xlsx",
    frs_naics_path = global_data_dir_path / "NATIONAL_NAICS_FILE.CSV",
    frs_facility_path = global_data_dir_path / "NATIONAL_FACILITY_FILE.CSV",
    # the NAICS code pulled from the v2 notebook for facilities in the FRS data
    MSW_FRS_NAICS_CODE = 562219,
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    nonreporting_proxy_output_path: Annotated[Path, Product] = proxy_data_dir_path / "msw_landfills_nr_proxy.parquet",
    ):

    # Non-reporting facilities from LMOP and WBJ
    nonreporting_facility_df = (
        pd.read_excel(
            nonreporting_facility_path,
            sheet_name="LandfillComp",
            usecols="A:B,D,F,AP,BL:BM,BR:BS",
            skiprows=5,
            nrows=1672
            )
            .rename(columns={
                'Landfill Name (as listed in Original Instance source)': 'Name',
                'HH Off-Ramped, Last Year Reported': 'Last GHGRP Year',
                'Avg. Est. Total WIP (MT)': 'WIP_MT',
                'LMOP Lat': 'LAT',
                'LMOP Long': 'LON'
                }
            )
            .astype({"WBJ City": str, "WBJ Location": str})
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

    FRS_facility_locs = (
        duckdb.execute(
            (
                "SELECT frs_main.primary_name as name, frs_main.location_address as location_address, frs_main.city_name as city_name, frs_main.state_code as state_code, frs_main.accuracy_value as accuracy_value, frs_main.latitude83 as latitude, frs_main.longitude83 as longitude "
                f"FROM (SELECT registry_id, primary_name, location_address, city_name, state_code, accuracy_value, latitude83, longitude83 FROM '{frs_facility_path}') as frs_main "
                f"JOIN (SELECT registry_id, naics_code FROM '{frs_naics_path}') AS frs_naics "
                "ON frs_main.registry_id = frs_naics.registry_id "
                f"WHERE naics_code == {MSW_FRS_NAICS_CODE}"
            )
        )
        .df()
        .assign(formatted_fac_name=lambda df: name_formatter(df["name"]), source="frs")
    )
    FRS_facility_locs['city_name'] = FRS_facility_locs['city_name'].replace('NO LOCALITY NAME TO MIGRATE FROM 2002 NEI V3', 'NaN')
    FRS_facility_locs.head()

    # Loop through the non-reporting landfills that do not have locations to try to find
    # matches in the FRS dataset.

    nonreporting_facility_noloc.loc[:, 'found'] = 0

    for ifacility in np.arange(0, len(nonreporting_facility_noloc)):
        # first try matching by state and exact name of landfill
        state_temp = nonreporting_facility_noloc.loc[ifacility, 'State']
        imatch = np.where((FRS_facility_locs['name'].str.contains(nonreporting_facility_noloc.loc[ifacility, 'Name'].upper()))\
                        & (FRS_facility_locs['state_code']==state_temp))[0]
        if len(imatch) == 1:
            nonreporting_facility_noloc.loc[ifacility, 'LAT'] = FRS_facility_locs.loc[imatch[0], 'latitude']
            nonreporting_facility_noloc.loc[ifacility, 'LON'] = FRS_facility_locs.loc[imatch[0], 'longitude']
            nonreporting_facility_noloc.loc[ifacility, 'found'] = 1
        elif len(imatch) > 1:
            # if name and state match more than one entry, use the one with the higher accuracy
            # or the first entry if the accuracy values are the same
            FRS_temp = FRS_facility_locs.loc[imatch, :].copy()
            # use the location of the first occurance of the facility with the highest accuracy value
            new_match = np.argmax(np.max(FRS_temp['accuracy_value']))
            nonreporting_facility_noloc.loc[ifacility, 'LAT'] = FRS_facility_locs.loc[imatch[new_match], 'latitude']
            nonreporting_facility_noloc.loc[ifacility, 'LON'] = FRS_facility_locs.loc[imatch[new_match], 'longitude']
            nonreporting_facility_noloc.loc[ifacility, 'found'] = 1

        # next try matching based on any of the words in name and state and city
        if nonreporting_facility_noloc.loc[ifacility, 'found'] == 0:
            string_temp = [x for x in nonreporting_facility_noloc.loc[ifacility,'Name'].upper().split() \
                        if x not in {'LANDFILL', 'SANITARY','CITY','TOWN','OF'}]
            string_temp = '|'.join(string_temp)
            string_temp = string_temp.replace("(","")
            string_temp = string_temp.replace(")","")
            string_temp = string_temp.replace("&","")
            string_temp = string_temp.replace("/","")

            city_temp = nonreporting_facility_noloc.loc[ifacility, 'WBJ City'].upper()
            imatch = np.where((FRS_facility_locs['name'].str.contains(string_temp))\
                        & (FRS_facility_locs['state_code']==state_temp) & (FRS_facility_locs['city_name'] == city_temp))[0]

            if len(imatch) == 1:
                nonreporting_facility_noloc.loc[ifacility, 'LAT'] = FRS_facility_locs.loc[imatch[0], 'latitude']
                nonreporting_facility_noloc.loc[ifacility, 'LON'] = FRS_facility_locs.loc[imatch[0], 'longitude']
                nonreporting_facility_noloc.loc[ifacility, 'found'] = 1
            elif len(imatch) > 1:
                # if name and state match more than one entry, use the one with the higher accuracy
                # or the first entry if the accuracy values are the same
                FRS_temp = FRS_facility_locs.loc[imatch,:].copy()
                # use the location of the first occurance of the facility with the highest accuracy value
                new_match = np.argmax(np.max(FRS_temp['accuracy_value']))
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[new_match],'latitude']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[new_match],'longitude']
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
            imatch = np.where((FRS_facility_locs['state_code']==state_temp) & (FRS_facility_locs['city_name']==city_temp))[0]
            if len(imatch) == 1:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'latitude']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'longitude']
                nonreporting_facility_noloc.loc[ifacility,'found'] = 1
            elif len(imatch) >1:
                # if city and state match more than one entry, use the one that has some matching address
                FRS_temp = FRS_facility_locs.loc[imatch,:].copy()
                new_match = np.where(FRS_temp['location_address'].str.contains(string_temp))[0]
                if len(new_match) >= 1:
                    nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[new_match[0]],'latitude']
                    nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[new_match[0]],'longitude']
                    nonreporting_facility_noloc.loc[ifacility,'found'] = 1
                
        if nonreporting_facility_noloc.loc[ifacility,'found'] == 0:
            # check if state matches and city and name have any matches
            city_temp = nonreporting_facility_noloc.loc[ifacility,'WBJ City'].upper()
            imatch = np.where((FRS_facility_locs['name'].str.contains(city_temp))\
                        & (FRS_facility_locs['state_code']==state_temp))[0]
            if len(imatch) == 1:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'latitude']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'longitude']
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
            imatch = np.where((FRS_facility_locs['name'].str.contains(string_temp))\
                        & (FRS_facility_locs['state_code']==state_temp))[0]
            # print(imatch)
            if len(imatch) == 1:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = FRS_facility_locs.loc[imatch[0],'latitude']
                nonreporting_facility_noloc.loc[ifacility,'LON'] = FRS_facility_locs.loc[imatch[0],'longitude']
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
    # If this portion of the code fails due to timeout error (1), just re-run and it will work.
    # This section takes ~15 minutes to run.
    print("Geolocating non-reporting facilities")
    geolocator = Nominatim(user_agent="myGeocode")
    geopy.geocoders.options.default_timeout = None
    print(geolocator.timeout)

    for ifacility in np.arange(0, len(nonreporting_facility_noloc)):
        if nonreporting_facility_noloc.loc[ifacility, 'found'] == 0:
            location = geolocator.geocode(nonreporting_facility_noloc['Full_Address'][ifacility])
            if location is None:
                continue
            else:
                nonreporting_facility_noloc.loc[ifacility, 'LAT'] = location.latitude
                nonreporting_facility_noloc.loc[ifacility, 'LON'] = location.longitude
                nonreporting_facility_noloc.loc[ifacility, 'found'] = 1
                
    print('First Try - Percentage found:',(np.sum(nonreporting_facility_noloc['found']))/len(nonreporting_facility_noloc))
    print('Missing Count',len(nonreporting_facility_noloc[nonreporting_facility_noloc['found'] == 0]))

    for ifacility in np.arange(0,len(nonreporting_facility_noloc)):
        if nonreporting_facility_noloc.loc[ifacility,'found'] ==0:
            #if still no match, remove the address portion and just allocate based on city, state
            location = geolocator.geocode(nonreporting_facility_noloc['Partial_Address'][ifacility])
            if location is None:
                continue
            else:
                nonreporting_facility_noloc.loc[ifacility,'LAT'] = location.latitude
                nonreporting_facility_noloc.loc[ifacility,'LON'] = location.longitude
                nonreporting_facility_noloc.loc[ifacility,'found'] = 1
                
    print('Second Try - Percentage found:',(np.sum(nonreporting_facility_noloc['found']))/len(nonreporting_facility_noloc))
    print('Missing Count',len(nonreporting_facility_noloc[nonreporting_facility_noloc['found']==0]))

    # Recombine non-reporting facilities into a single dataframe
    nonreporting_facility_df_with_locs = nonreporting_facility_loc.merge(nonreporting_facility_noloc, how='left')

    # Create individual data rows for non-reporting facilities where
    # "Last GHGRP Year" = 0 to have data for each year.
    nonreporting_facility_all_years_df = pd.DataFrame()
    nr_year0_facilities_expanded_years = pd.DataFrame()
    nr_otheryears_facilities_expanded_years = pd.DataFrame()

    nr_all_facilities_copy = nonreporting_facility_df_with_locs.copy()
    nr_all_facilities_copy = nr_all_facilities_copy.rename(columns={'Last GHGRP Year': 'Last_GHGRP_Year'})
    nr_all_facilities_copy['Year'] = 0

    nr_year0_facilities = nr_all_facilities_copy.query("Last_GHGRP_Year == 0")
    nr_otheryears_facilities = nr_all_facilities_copy.query("Last_GHGRP_Year != 0")
    for iyear in np.arange(0, num_years):
        year_actual = year_range[iyear]
        # Assign years to facilities with no GHGRP reporting year (Last_GHGRP_Year == 0)
        nr_year0_facilities['Year'] = year_range[iyear]
        nr_year0_facilities_expanded_years = pd.concat([nr_year0_facilities_expanded_years, nr_year0_facilities])

        # Assign years to facilities that have a year they last reported to GHGRP
        nr_otheryears_facilities = nr_otheryears_facilities.query('Last_GHGRP_Year < @year_actual')
        nr_otheryears_facilities['Year'] = year_range[iyear]
        nr_otheryears_facilities_expanded_years = pd.concat([nr_otheryears_facilities_expanded_years, nr_otheryears_facilities])

    nonreporting_facility_all_years_df = pd.concat([nonreporting_facility_all_years_df, nr_year0_facilities_expanded_years, nr_otheryears_facilities_expanded_years]).drop(columns="Last_GHGRP_Year")
    nonreporting_facility_all_years_df = nonreporting_facility_all_years_df.rename(columns={"NR ID":"facility_id", "Name": "facility_name", "State": "state_code", "WIP_MT": "wip_mt", "Year": "year"})

    nonreporting_facility_all_years_df['rel_emi'] = nonreporting_facility_all_years_df.groupby(["state_code", "year"])['wip_mt'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
    nonreporting_facility_all_years_df = nonreporting_facility_all_years_df.drop(columns='wip_mt')

    nonreporting_facility_gdf = (
    gpd.GeoDataFrame(
        nonreporting_facility_all_years_df,
        geometry=gpd.points_from_xy(
            nonreporting_facility_all_years_df["LON"],
            nonreporting_facility_all_years_df["LAT"],
            crs=4326,
            ),
            )
    .drop(columns=["facility_id", "LAT", "LON", "WBJ Location", "WBJ City", "Full_Address", "Partial_Address", "found"])
    .loc[:, ["year", "state_code", "geometry", "rel_emi"]]
    )

    nonreporting_facility_gdf.to_parquet(nonreporting_proxy_output_path)
    return None


# Correct proxy data if missing states compared to emi files:
# If proxy data exists for a state in the other landfill proxy, assign emissions to 
# those locations, otherwise assign the emissions uniformly across the entire state.
# For example, if the proxy data is missing for reporting landfills in RI, check to see
# if the data exists for nonreporting landfills in RI. If the data exists, assign 
# reporting emissions to these locations, otherwise assign the emissions uniformly 
# across the state of RI.
def add_missing_landfills_proxy_data(
        state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
        r_emi_path: Annotated[Path, Product] = emi_data_dir_path / "msw_landfills_r_emi.csv",
        r_proxy_path: Annotated[Path, Product] = proxy_data_dir_path / "msw_landfills_r_proxy.parquet",
        nr_emi_path: Annotated[Path, Product] = emi_data_dir_path / "msw_landfills_nr_emi.csv",
        nr_proxy_path: Annotated[Path, Product] = proxy_data_dir_path / "msw_landfills_nr_proxy.parquet",
        ):

    # Read in State Geo Data
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

    # Read in reporting and non-reporting emission and proxy files
    r_emi_df = pd.read_csv(r_emi_path).query("ghgi_ch4_kt > 0")
    r_proxy_gdf = gpd.read_parquet(r_proxy_path)
    nr_emi_df = pd.read_csv(nr_emi_path).query("ghgi_ch4_kt > 0")
    nr_proxy_gdf = gpd.read_parquet(nr_proxy_path)

    def get_alt_proxy_data(emi_to_update, proxy_to_update, proxy_to_compare):
        # Get lists of state code-year pairs for all emi and proxy files
        emi_to_update_states = set(emi_to_update[['state_code', 'year']].itertuples(index=False, name=None))
        proxy_to_update_states = set(proxy_to_update[['state_code', 'year']].itertuples(index=False, name=None))
        proxy_to_compare_states = set(proxy_to_compare[['state_code', 'year']].itertuples(index=False, name=None))
        # Get unique state code-year pairs without proxy data
        missing_states = emi_to_update_states.difference(proxy_to_update_states)
        if missing_states:
            alt_proxy = gpd.GeoDataFrame()
            for istate_year in np.arange(0, len(missing_states)):
                istate = str(list(missing_states)[istate_year])[2:4]
                iyear = int(str(list(missing_states)[istate_year])[7:11])
                # If the missing state code-year pair is in the non-reporting facility list,
                # assign emissions to those facility locations.
                if list(missing_states)[istate_year] in list(proxy_to_compare_states):
                    iproxy_data = (proxy_to_compare
                                   .query("state_code == @istate")
                                   .query("year == @iyear")
                                   )
                    alt_proxy = gpd.GeoDataFrame(pd.concat([alt_proxy, iproxy_data], ignore_index=True))
                # Otherwise, assign emissions evenly across the state.
                else:
                    iproxy_data = (gpd.GeoDataFrame()
                                   .assign(year=iyear,
                                           state_code=istate,
                                           rel_emi=1.0)
                                           .merge(
                                               state_gdf[['state_code', 'geometry']],
                                               on='state_code',
                                               how='left')
                                  )
                    alt_proxy = gpd.GeoDataFrame(pd.concat([alt_proxy, iproxy_data], ignore_index=True))
            # Add missing proxy to original proxy
            proxy_gdf_updated = gpd.GeoDataFrame(pd.concat([proxy_to_update, alt_proxy], ignore_index=True).reset_index(drop=True))
        else:
            proxy_gdf_updated = proxy_to_update
        
        return proxy_gdf_updated
            
    r_proxy_gdf_updated = get_alt_proxy_data(r_emi_df, r_proxy_gdf, nr_proxy_gdf)
    nr_proxy_gdf_updated = get_alt_proxy_data(nr_emi_df, nr_proxy_gdf, r_proxy_gdf)

    # Save new proxy data
    r_proxy_gdf_updated.to_parquet(r_proxy_path)
    nr_proxy_gdf_updated.to_parquet(nr_proxy_path)

    return None
