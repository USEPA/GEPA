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

t_to_kt = 0.001  # conversion factor, metric tonnes to kilotonnes
mmt_to_kt = 1000  # conversion factor, million metric tonnes to kilotonnes
year_range = [*range(min_year, max_year+1,1)] #List of emission years
year_range_str=[str(i) for i in year_range]
num_years = len(year_range)

# %%
@mark.persist
@task(id="industrial_landfills_emi")
def task_get_industrial_landfills_pulp_paper_inv_data(
    inventory_workbook_path: Path = ghgi_data_dir_path / "landfills/State_IND_LF_1990-2022_LA.xlsx",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    subpart_tt_path = "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV",
    reporting_pulp_paper_emis_output_path: Annotated[Path, Product] = emi_data_dir_path / "ind_landfills_pp_r_emi.csv",
    nonreporting_pulp_paper_emis_output_path: Annotated[Path, Product] = emi_data_dir_path / "ind_landfills_pp_nr_emi.csv",
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
    state_inventory_pulp_paper_emi_df = (
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
        .assign(ch4_kt=lambda df: df["ch4_mmt"] * mmt_to_kt)
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
    national_inventory_pulp_paper_emi_df = state_inventory_pulp_paper_emi_df.drop(columns=["state_code"]).groupby(['year']).sum().reset_index()

    # Reporting facilities from subpart tt with NAICS codes that start with 321 and 322
    subpart_tt_pulp_paper_emi_df = (
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
        .query("year.between(@min_year, @max_year)")
        .astype({"naics_code": str})
        .query("naics_code.str.startswith('321') | naics_code.str.startswith('322')" )
        .drop(columns=["facility_id", "facility_name", "latitude", "longitude", "city", "zip", "naics_code"])
        .query("state_code.isin(@state_gdf['state_code'])")
        .reset_index(drop=True)
    )

    # State-level Subpart TT emissions for pulp and paper (reporting)
    state_subpart_tt_pulp_paper_emi_df = subpart_tt_pulp_paper_emi_df.groupby(['year', 'state_code']).sum().reset_index()

    # National-level Subpart TT emissions for pulp and paper (reporting)
    national_subpart_tt_pulp_paper_emi_df = state_subpart_tt_pulp_paper_emi_df.drop(columns=["state_code"]).groupby(['year']).sum().reset_index()

    # National-level reporting and non-reporting emissions by year
    national_reporting_pulp_paper_emi_df = national_subpart_tt_pulp_paper_emi_df.copy()

    # State-level reporting and non-reporting emissions by year
    corrected_pulp_paper_emi_df = pd.DataFrame()  # dataframe to hold state-level reporting and nonreporting emissions
    state_inventory_pulp_paper_emi_df_reporting_states = state_inventory_pulp_paper_emi_df.query("state_code.isin(@state_subpart_tt_pulp_paper_emi_df['state_code'])").reset_index(drop=True)
    for iyear in np.arange(0, len(national_reporting_pulp_paper_emi_df)):
        year_actual = years[iyear]
        ghgi_national_emi = national_inventory_pulp_paper_emi_df.loc[iyear,'ch4_kt']
        reporting_national_emi = national_reporting_pulp_paper_emi_df.loc[iyear, 'ch4_kt']
        reporting_state_emi = state_inventory_pulp_paper_emi_df_reporting_states.query('year == @year_actual').assign(ghgi_ch4_kt=lambda df: df["ch4_kt"] * reporting_national_emi / ghgi_national_emi)
        corrected_pulp_paper_emi_df = pd.concat([corrected_pulp_paper_emi_df, reporting_state_emi])
    corrected_pulp_paper_emi_df = (corrected_pulp_paper_emi_df
                                   .merge(state_inventory_pulp_paper_emi_df, left_on=['state_code', 'year'], right_on=['state_code', 'year'], how='outer')
                                   .rename(columns={"ghgi_ch4_kt": "rep_ghgi_ch4_kt", "ch4_kt_y": "tot_ghgi_ch4_kt"})
                                   .fillna(0)
                                   .assign(nonrep_ghgi_ch4_kt=lambda df: df['tot_ghgi_ch4_kt']-df['rep_ghgi_ch4_kt'])
                                   .drop(columns=["ch4_kt_x"])                                   
                                   )
    
    reporting_pulp_paper_emi_df = (corrected_pulp_paper_emi_df
                                   .drop(columns=["tot_ghgi_ch4_kt", "nonrep_ghgi_ch4_kt"])
                                   .rename(columns={"rep_ghgi_ch4_kt": "ghgi_ch4_kt"}))

    nonreporting_pulp_paper_emi_df = (corrected_pulp_paper_emi_df
                                   .drop(columns=["tot_ghgi_ch4_kt", "rep_ghgi_ch4_kt"])
                                   .rename(columns={"nonrep_ghgi_ch4_kt": "ghgi_ch4_kt"}))

    reporting_pulp_paper_emi_df.to_csv(reporting_pulp_paper_emis_output_path, index=False)
    nonreporting_pulp_paper_emi_df.to_csv(nonreporting_pulp_paper_emis_output_path, index=False)
    return None


def task_get_industrial_landfills_food_beverage_inv_data(
    inventory_workbook_path: Path = ghgi_data_dir_path / "landfills/State_IND_LF_1990-2022_LA.xlsx",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    subpart_tt_path = "https://data.epa.gov/efservice/tt_subpart_ghg_info/pub_dim_facility/ghg_name/=/Methane/CSV",
    reporting_food_beverage_emis_output_path: Annotated[Path, Product] = emi_data_dir_path / "ind_landfills_fb_r_emi.csv",
    nonreporting_food_beverage_emis_output_path: Annotated[Path, Product] = emi_data_dir_path / "ind_landfills_fb_nr_emi.csv",
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
    state_inventory_food_beverage_emi_df = (
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
        .assign(ch4_kt=lambda df: df["ch4_mmt"] * mmt_to_kt)
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
    national_inventory_food_beverage_emi_df = state_inventory_food_beverage_emi_df.drop(columns=["state_code"]).groupby(['year']).sum().reset_index()

    # Reporting facilities from subpart tt with NAICS codes that start with 321 and 322
    subpart_tt_food_beverage_emi_df = (
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
    .query("year.between(@min_year, @max_year)")
    .astype({"naics_code": int})
    .query("naics_code == 311612|naics_code == 311421|naics_code == 311513|naics_code == 312140|naics_code == 311611|naics_code == 311615|naics_code == 311225|naics_code == 311613|naics_code == 311710|naics_code == 311221|naics_code == 311224|naics_code == 311314|naics_code == 311313") 
    .drop(columns=["facility_id", "facility_name", "latitude", "longitude", "city", "zip", "naics_code"])
    .query("state_code.isin(@state_gdf['state_code'])")
    .reset_index(drop=True)
    )

    # State-level Subpart TT emissions for food and beverage (reporting)
    state_subpart_tt_food_beverage_emi_df = subpart_tt_food_beverage_emi_df.groupby(['year', 'state_code']).sum().reset_index()

    # National-level Subpart TT emissions for food and beverage (reporting)
    national_subpart_tt_food_beverage_emi_df = state_subpart_tt_food_beverage_emi_df.drop(columns=["state_code"]).groupby(['year']).sum().reset_index()

    # National-level reporting and non-reporting emissions by year
    national_reporting_food_beverage_emi_df = national_subpart_tt_food_beverage_emi_df.copy()

    # State-level reporting and non-reporting emissions by year
    corrected_food_beverage_emi_df = pd.DataFrame()  # dataframe to hold state-level reporting and nonreporting emissions
    state_inventory_food_beverage_emi_df_reporting_states = state_inventory_food_beverage_emi_df.query("state_code.isin(@state_subpart_tt_food_beverage_emi_df['state_code'])").reset_index(drop=True)
    for iyear in np.arange(0, len(national_reporting_food_beverage_emi_df)):
        year_actual = years[iyear]
        ghgi_national_emi = national_inventory_food_beverage_emi_df.loc[iyear,'ch4_kt']
        reporting_national_emi = national_reporting_food_beverage_emi_df.loc[iyear, 'ch4_kt']
        reporting_state_emi = state_inventory_food_beverage_emi_df_reporting_states.query('year == @year_actual').assign(ghgi_ch4_kt=lambda df: df["ch4_kt"] * reporting_national_emi / ghgi_national_emi)
        corrected_food_beverage_emi_df = pd.concat([corrected_food_beverage_emi_df, reporting_state_emi])
    corrected_food_beverage_emi_df = (corrected_food_beverage_emi_df
                                   .merge(state_inventory_food_beverage_emi_df, left_on=['state_code', 'year'], right_on=['state_code', 'year'], how='outer')
                                   .rename(columns={"ghgi_ch4_kt": "rep_ghgi_ch4_kt", "ch4_kt_y": "tot_ghgi_ch4_kt"})
                                   .fillna(0.0)
                                   .assign(nonrep_ghgi_ch4_kt=lambda df: df['tot_ghgi_ch4_kt']-df['rep_ghgi_ch4_kt'])
                                   .drop(columns=["ch4_kt_x"])
                                   )
    
    reporting_food_beverage_emi_df = (corrected_food_beverage_emi_df
                                   .drop(columns=["tot_ghgi_ch4_kt", "nonrep_ghgi_ch4_kt"])
                                   .rename(columns={"rep_ghgi_ch4_kt": "ghgi_ch4_kt"}))

    nonreporting_food_beverage_emi_df = (corrected_food_beverage_emi_df
                                   .drop(columns=["tot_ghgi_ch4_kt", "rep_ghgi_ch4_kt"])
                                   .rename(columns={"nonrep_ghgi_ch4_kt": "ghgi_ch4_kt"}))

    reporting_food_beverage_emi_df.to_csv(reporting_food_beverage_emis_output_path, index=False)
    nonreporting_food_beverage_emi_df.to_csv(nonreporting_food_beverage_emis_output_path, index=False)
    return None
