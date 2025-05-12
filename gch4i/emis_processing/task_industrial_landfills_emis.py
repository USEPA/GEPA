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
    emi_data_dir_path,
    max_year,
    min_year,
    years,
)

from gch4i.utils import name_formatter

# change this variable name to 'mt_to_kt'
tg_to_kt = 0.001  # conversion factor, metric tonnes to kilotonnes
mmtg_to_kt = 1000  # conversion factor, million metric tonnes to kilotonnes


# %%
@mark.persist
@task(id="industrial_landfills_emi")
def task_get_industrial_landfills_pulp_paper_inv_data(
    inventory_workbook_path: Path = ghgi_data_dir_path / "landfills/State_IND_LF_1990-2022_LA.xlsx",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    subpart_tt_path = "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV",
    mills_online_path: Path = V3_DATA_PATH / "sector/landfills/Mills_OnLine.xlsx",
    reporting_pp_emis_output_path: Annotated[Path, Product] = emi_data_dir_path / "ind_landfills_pp_r_emi.csv",
    nonreporting_pp_emis_output_path: Annotated[Path, Product] = emi_data_dir_path / "ind_landfills_pp_nr_emi.csv",
) -> None:
    """read in the ghgi_ch4_kt values for each state"""

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

    # State-level inventory emissions for pulp and paper (reporting + non-reporting)
    state_inventory_pp_emi_df = (
        pd.read_excel(
            inventory_workbook_path,
            sheet_name="P&P State Emissions",
            skiprows=5,
            nrows=60,
            usecols="B:AI",
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
        .assign(ch4_kt=lambda df: df["ch4_mmt"] * mmtg_to_kt)
        .drop(columns=["ch4_mmt"])
        # make the columns types correct
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
    )
    # National-level inventory emissions for pulp and paper (reporting + non-reporting)
    national_inventory_pp_emi_df = state_inventory_pp_emi_df.drop(columns=["state_code"]).groupby(['year']).sum().reset_index()

    # Reporting facilities from subpart tt with NAICS codes that start with 321 and 322
    subpart_tt_pp_emi_df = (
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
        .drop(columns=["facility_id", "facility_name", "latitude", "longitude", "zip", "naics_code"])
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
    )
    # State-level Subpart TT emissions for pulp and paper (reporting)
    state_subpart_tt_pp_emi_df = subpart_tt_pp_emi_df.groupby(['year', 'state_code']).sum().reset_index()
    # list of unique states in subpart tt
    subpart_tt_pp_states = state_subpart_tt_pp_emi_df['state_code'].unique()
    # National-level Subpart TT emissions for pulp and paper (reporting)
    national_subpart_tt_pp_emi_df = state_subpart_tt_pp_emi_df.drop(columns=["state_code"]).groupby(['year']).sum().reset_index()

    # List of pulp and paper mills by state, county, and city
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
    mills_locs.loc[:, 'city'] = mills_locs.loc[:, 'city'].str.lower()
    subpart_tt_pp_emi_df.loc[:, 'city'] = subpart_tt_pp_emi_df.loc[:, 'city'].str.lower()
    # try to match facilities to GHGRP based on county and city
    for iyear in years:
        for ifacility in np.arange(0,num_mills):
            imatch = np.where((subpart_tt_pp_emi_df['year'] == iyear) & \
                            (subpart_tt_pp_emi_df['state_code'] == mills_locs.loc[ifacility,'state_code']) & \
                            (subpart_tt_pp_emi_df['city'] == mills_locs.loc[ifacility,'city']))[0]
            if len(imatch) > 0:
                mills_locs.loc[ifacility,'ghgrp_match'] = 1
            else:
                continue
    # drop facilities already in reporting dataframe
    mills_locs = mills_locs.query('ghgrp_match == 0')
    # list of unique states in non-reporting dataframe
    nr_pp_states = mills_locs['state_code'].unique()
    # remove city from subpart tt dataframe
    subpart_tt_pp_emi_df = subpart_tt_pp_emi_df.drop(columns=["city"])

    # Calculate reporting and non-reporting emissions
    # dataframe to hold state-level reporting and nonreporting emissions
    corrected_pp_emi_df = pd.DataFrame()

    for istate in subpart_tt_pp_states:
        # subpart tt data for the specific state of interest
        subpart_tt_pp_istate = state_subpart_tt_pp_emi_df.copy()
        subpart_tt_pp_istate = subpart_tt_pp_istate.query('state_code == @istate')
        # list of reporting years for the specific state of interest
        # (not all states have data for all reporting years e.g., FL and OK)
        subpart_tt_pp_istate_years = subpart_tt_pp_istate['year'].unique()
        # inventory emissions for the specific state of interest
        state_inventory_pp_istate = state_inventory_pp_emi_df.copy()
        state_inventory_pp_istate = state_inventory_pp_istate.query('state_code == @istate')
        
        for iyear in subpart_tt_pp_istate_years:
            # national inventory emission value for the specific year
            inventory_national_emi = national_inventory_pp_emi_df.loc[national_inventory_pp_emi_df['year'] == iyear,'ch4_kt'].values[0]
            # national subpart tt emission value for the specific year
            subpart_tt_national_emi = national_subpart_tt_pp_emi_df.loc[national_subpart_tt_pp_emi_df['year'] == iyear, 'ch4_kt'].values[0]
            # recalculated reporting emission for the specific state-year combination
            # national subpart tt emission value weighted by the ratio of the specific
            # state's inventory contribution to the total inventory.
            if istate in nr_pp_states:
                reporting_state_emi = state_inventory_pp_istate.query('year == @iyear').assign(r_ch4_kt=lambda df: df["ch4_kt"] * subpart_tt_national_emi / inventory_national_emi).drop(columns=["ch4_kt"])
            else:
                reporting_state_emi = state_inventory_pp_istate.query('year == @iyear').assign(r_ch4_kt=lambda df: df["ch4_kt"]).drop(columns=["ch4_kt"])
            corrected_pp_emi_df = pd.concat([corrected_pp_emi_df, reporting_state_emi])
    corrected_pp_emi_df = (corrected_pp_emi_df
                                .merge(state_inventory_pp_emi_df, left_on=['state_code', 'year'], right_on=['state_code', 'year'], how='outer')
                                .rename(columns={"ch4_kt": "tot_ch4_kt"})
                                .fillna(0)
                                .assign(nr_ch4_kt=lambda df: df['tot_ch4_kt']-df['r_ch4_kt'])
                                .replace(0, np.nan)                                   
                                )
    
    reporting_pp_emi_df = (corrected_pp_emi_df
                                   .drop(columns=["tot_ch4_kt", "nr_ch4_kt"])
                                   .rename(columns={"r_ch4_kt": "ghgi_ch4_kt"})
                                   .dropna()
                                   .reset_index(drop=True)
                                   )

    nonreporting_pp_emi_df = (corrected_pp_emi_df
                                   .drop(columns=["tot_ch4_kt", "r_ch4_kt"])
                                   .rename(columns={"nr_ch4_kt": "ghgi_ch4_kt"})
                                   .dropna()
                                   .reset_index(drop=True)
                                   )

    reporting_pp_emi_df.to_csv(reporting_pp_emis_output_path, index=False)
    nonreporting_pp_emi_df.to_csv(nonreporting_pp_emis_output_path, index=False)

    return None


def task_get_industrial_landfills_food_beverage_inv_data(
    inventory_workbook_path: Path = ghgi_data_dir_path / "landfills/State_IND_LF_1990-2022_LA.xlsx",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    subpart_tt_path = "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV",
    food_manufacturers_processors_path = V3_DATA_PATH / "sector/landfills/Food Manufacturers and Processors.xlsx",
    reporting_fb_emis_output_path: Annotated[Path, Product] = emi_data_dir_path / "ind_landfills_fb_r_emi.csv",
    nonreporting_fb_emis_output_path: Annotated[Path, Product] = emi_data_dir_path / "ind_landfills_fb_nr_emi.csv",
) -> None:
    """read in the ghgi_ch4_kt values for each state"""

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
        
    # State-level inventory emissions for pulp and paper (reporting + non-reporting)
    state_inventory_fb_emi_df = (
        pd.read_excel(
            inventory_workbook_path,
            sheet_name="F&B State Emissions",
            skiprows=5,
            nrows=927,
            usecols="B:AI",
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
        .assign(ch4_kt=lambda df: df["ch4_mmt"] * mmtg_to_kt)
        .drop(columns=["ch4_mmt"])
        # make the columns types correct
        .astype({"year": int, "ch4_kt": float})
        .fillna({"ch4_kt": 0})
        # get only the years we need
        .query("year.between(@min_year, @max_year)")
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
    )
    # National-level inventory emissions for pulp and paper (reporting + non-reporting)
    national_inventory_fb_emi_df = state_inventory_fb_emi_df.drop(columns=["state_code"]).groupby(['year']).sum().reset_index()

    # Reporting facilities from subpart tt with NAICS codes that start with 321 and 322
    subpart_tt_fb_emi_df = (
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
    .drop(columns=["facility_id", "facility_name", "latitude", "longitude", "zip", "naics_code"])
    .query("state_code.isin(@state_gdf['state_code'])")
    .reset_index(drop=True)
    )
    # State-level Subpart TT emissions for pulp and paper (reporting)
    state_subpart_tt_fb_emi_df = subpart_tt_fb_emi_df.groupby(['year', 'state_code']).sum().reset_index()
    # list of unique states in subpart tt
    subpart_tt_fb_states = state_subpart_tt_fb_emi_df['state_code'].unique()
    # National-level Subpart TT emissions for pulp and paper (reporting)
    national_subpart_tt_fb_emi_df = state_subpart_tt_fb_emi_df.drop(columns=["state_code"]).groupby(['year']).sum().reset_index()

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
    .reset_index(drop=True)
    )

    # try to match facilities to GHGRP based on county and city
    food_beverage_facilities_locs = food_beverage_facilities_df.copy()
    num_facilities = len(food_beverage_facilities_locs)
    food_beverage_facilities_locs.loc[:,'ghgrp_match'] = 0
    food_beverage_facilities_locs.loc[:,'city'] = food_beverage_facilities_locs.loc[:,'city'].str.lower()

    subpart_tt_fb_emi_df.loc[:,'city'] = subpart_tt_fb_emi_df.loc[:,'city'].str.lower()

    for iyear in years:
        for ifacility in np.arange(0,num_facilities):
            imatch = np.where((subpart_tt_fb_emi_df['year'] == iyear) & \
                            (subpart_tt_fb_emi_df['state_code'] == food_beverage_facilities_locs.loc[ifacility,'state_code']) & \
                            (subpart_tt_fb_emi_df['city'] == food_beverage_facilities_locs.loc[ifacility,'city']))[0]
            if len(imatch) > 0:
                food_beverage_facilities_locs.loc[ifacility,'ghgrp_match'] = 1
            else:
                continue
    
    # drop facilities already in the reporting dataframe
    food_beverage_facilities_locs = food_beverage_facilities_locs.query('ghgrp_match == 0')
    # list of unique states in non-reporting dataframe
    nr_fb_states = food_beverage_facilities_locs['state_code'].unique()
    # remove city from subpart tt dataframe
    subpart_tt_fb_emi_df = subpart_tt_fb_emi_df.drop(columns=["city"])

    # Calcute reporting and non-reporting emissions
    # dataframe to hold state-level reporting and nonreporting emissions
    corrected_fb_emi_df = pd.DataFrame()

    for istate in subpart_tt_fb_states:
        # subpart tt data for the specific state of interest
        subpart_tt_fb_istate = state_subpart_tt_fb_emi_df.copy()
        subpart_tt_fb_istate = subpart_tt_fb_istate.query('state_code == @istate')
        # list of reporting years for the specific state of interest
        # (not all states have data for all reporting years e.g., FL and OK)
        subpart_tt_fb_istate_years = subpart_tt_fb_istate['year'].unique()
        # inventory emissions for the specific state of interest
        state_inventory_fb_istate = state_inventory_fb_emi_df.copy()
        state_inventory_fb_istate = state_inventory_fb_istate.query('state_code == @istate')
        
        for iyear in subpart_tt_fb_istate_years:
            # national inventory emission value for the specific year
            inventory_national_emi = national_inventory_fb_emi_df.loc[national_inventory_fb_emi_df['year'] == iyear,'ch4_kt'].values[0]
            # national subpart tt emission value for the specific year
            subpart_tt_national_emi = national_subpart_tt_fb_emi_df.loc[national_subpart_tt_fb_emi_df['year'] == iyear, 'ch4_kt'].values[0]
            # recalculated reporting emission for the specific state-year combination
            # national subpart tt emission value weighted by the ratio of the specific
            # state's inventory contribution to the total inventory.
            if istate in nr_fb_states:
                reporting_state_emi = state_inventory_fb_istate.query('year == @iyear').assign(r_ch4_kt=lambda df: df["ch4_kt"] * subpart_tt_national_emi / inventory_national_emi).drop(columns=["ch4_kt"])
            else:
                reporting_state_emi = state_inventory_fb_istate.query('year == @iyear').assign(r_ch4_kt=lambda df: df["ch4_kt"]).drop(columns=["ch4_kt"])
            corrected_fb_emi_df = pd.concat([corrected_fb_emi_df, reporting_state_emi])
    corrected_fb_emi_df = (corrected_fb_emi_df
                                .merge(state_inventory_fb_emi_df, left_on=['state_code', 'year'], right_on=['state_code', 'year'], how='outer')
                                .rename(columns={"ch4_kt": "tot_ch4_kt"})
                                .fillna(0)
                                .assign(nr_ch4_kt=lambda df: df['tot_ch4_kt']-df['r_ch4_kt'])
                                .replace(0, np.nan)                                   
                                )
    
    reporting_fb_emi_df = (corrected_fb_emi_df
                                   .drop(columns=["tot_ch4_kt", "nr_ch4_kt"])
                                   .rename(columns={"r_ch4_kt": "ghgi_ch4_kt"})
                                   .dropna()
                                   .query("ghgi_ch4_kt > 0")
                                   .reset_index(drop=True)
                                   )

    nonreporting_fb_emi_df = (corrected_fb_emi_df
                                   .drop(columns=["tot_ch4_kt", "r_ch4_kt"])
                                   .rename(columns={"nr_ch4_kt": "ghgi_ch4_kt"})
                                   .dropna()
                                   .query("ghgi_ch4_kt > 0")
                                   .reset_index(drop=True)
                                   )

    reporting_fb_emi_df.to_csv(reporting_fb_emis_output_path, index=False)
    nonreporting_fb_emi_df.to_csv(nonreporting_fb_emis_output_path, index=False)
    
    return None
