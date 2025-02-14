# %%
from pathlib import Path
import os
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
import pyodbc

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    sector_data_dir_path,
    emi_data_dir_path,
    max_year,
    min_year,
    years,
)

from gch4i.proxy_processing.ng_oil_production_utils import (
    create_alt_proxy,
)

# %%
@mark.persist
@task(id="ng_oil_federal_gom_offshore_proxy")
def task_get_ng_oil_federal_gom_offshore_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    GOADS_11_path: Path = sector_data_dir_path / "boem" / "2011_Gulfwide_Platform_Inventory.accdb",
    GOADS_14_path: Path = sector_data_dir_path / "boem" / "2014_Gulfwide_Platform_Inventory.accdb",
    GOADS_17_path: Path = sector_data_dir_path / "boem" / "2017_Gulfwide_Platform_Inventory.accdb",
    ERG_GOADSEmissions_path: Path = sector_data_dir_path / "boem" / "BOEM GEI Emissions Data_EmissionSource_2020-03-11.xlsx",
    federal_gom_offshore_emi_path: Path = emi_data_dir_path / "federal_gom_offshore_emi.csv",
    oil_gom_federal_emi_path: Path = emi_data_dir_path / "oil_gom_federal_emi.csv",
    ng_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_federal_gom_offshore_proxy.parquet",
    oil_output_path: Annotated[Path, Product] = proxy_data_dir_path / "oil_federal_gom_offshore_proxy.parquet",
):
    """
    # TODO:
    """

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

    # Get and format BOEM GOM data for 2011, 2014, and 2017

    # GOADS data year assignments
        # 2011 data: 2012
        # 2014 data: 2013, 2014, 2015
        # 2017 data: 2016-2022
        # 2021 data: NOT USED BY GHGI TEAM YET - CHECK FOR V4

    federal_gom_offshore_data_years = pd.DataFrame(
        {'year': [2012, 2013, 2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022],
         'goads_data': [2011, 2014, 2014, 2014, 2017, 2017, 2017, 2017, 2017, 2017, 2017]
         })

    # Use ERG Preprocessed data to determine if oil or gas
    ERG_complex_crosswalk = (pd.read_excel(
        ERG_GOADSEmissions_path,
        sheet_name = "Complex Emissions by Source",
        usecols = "AJ:AM",
        nrows = 11143)
        .rename(columns={"Year.2": "year",
                         "BOEM COMPLEX ID.2": "boem_complex_id",
                         "Oil Gas Defn FINAL.1": "oil_gas_defn",
                         "Major / Minor.1": "major_minor"})
        .query("year == 2011 | year == 2014 | year == 2017")
        .astype({"boem_complex_id": int})
        .drop(columns="major_minor")  # no longer separating major vs. minor in v3
        .replace('', np.nan)
        .dropna()
        .reset_index(drop=True)
        )

    # 2011 GOADS Data
    # Read In and Format 2011 BEOM Data
    GOADS_11_inputfile = str(GOADS_11_path)
    driver_str = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+GOADS_11_inputfile+';'''
    conn = pyodbc.connect(driver_str)
    GOADS_locations = pd.read_sql("SELECT * FROM tblPointER", conn)
    GOADS_emissions = pd.read_sql("SELECT * FROM tblPointEM", conn)
    conn.close()

    # Format Location Data
    GOADS_locations = GOADS_locations[["strStateFacilityIdentifier","strEmissionReleasePointID","dblXCoordinate","dblYCoordinate"]]
    # Create platform-by-platform file
    GOADS_locations_Unique = pd.DataFrame({'strStateFacilityIdentifier':GOADS_locations['strStateFacilityIdentifier'].unique()})
    GOADS_locations_Unique['lon'] = 0.0
    GOADS_locations_Unique['lat'] = 0.0
    GOADS_locations_Unique['strEmissionReleasePointID'] = ''

    for iplatform in np.arange(len(GOADS_locations_Unique)):
        match_platform = np.where(GOADS_locations['strStateFacilityIdentifier'] == GOADS_locations_Unique['strStateFacilityIdentifier'][iplatform])[0][0]
        GOADS_locations_Unique.loc[iplatform,'lon',] = GOADS_locations['dblXCoordinate'][match_platform]
        GOADS_locations_Unique.loc[iplatform,'lat',] = GOADS_locations['dblYCoordinate'][match_platform]
        GOADS_locations_Unique.loc[iplatform,'strEmissionReleasePointID'] = GOADS_locations['strEmissionReleasePointID'][match_platform][:3]

    GOADS_locations_Unique = (GOADS_locations_Unique
                              .drop(columns='strEmissionReleasePointID')
                              .replace('', np.nan)
                              .dropna()
                              .reset_index(drop=True))

    # Format Emissions Data (clean lease data string)
    GOADS_emissions = GOADS_emissions[["strStateFacilityIdentifier","strPollutantCode",
                                       "dblEmissionNumericValue","BOEM-MONTH",
                                       "BOEM-COMPLEX_ID"]]
    GOADS_emissions = (GOADS_emissions
                       .query("strPollutantCode == 'CH4'")
                       .assign(Emis_tg = 0.0)
                       .assign(Emis_tg = lambda df: 9.0718474E-7 * df['dblEmissionNumericValue']) #convert short tons to Tg
                       .rename(columns={"BOEM-COMPLEX_ID": "boem_complex_id"})
                       .astype({"boem_complex_id": int})
                       .drop(columns={"strPollutantCode", "dblEmissionNumericValue"})
                       .replace('', np.nan)
                       .dropna()
                       .reset_index(drop=True)
                       )

    # Select 2011 data from ERG complex crosswalk
    ERG_complex_crosswalk_2011 = ERG_complex_crosswalk.copy().query('year == 2011').reset_index(drop=True)

    # Join locations, emissions, and complex types together
    federal_gom_offshore_2011 = (GOADS_emissions
                                 .set_index("boem_complex_id")
                                 .join(ERG_complex_crosswalk_2011.set_index("boem_complex_id"))
                                 .reset_index()
                                 .set_index("strStateFacilityIdentifier")
                                 .join(GOADS_locations_Unique.set_index("strStateFacilityIdentifier"))
                                 .reset_index()
                                 .astype({"BOEM-MONTH": str})
                                 .assign(month=lambda df: df['BOEM-MONTH'].astype(str).str.zfill(2))
                                 .assign(state_code='FO')
                                 .drop(columns={'strStateFacilityIdentifier', 'BOEM-MONTH'})
                                 )
    federal_gom_offshore_2011_gdf = (
        gpd.GeoDataFrame(
            federal_gom_offshore_2011,
            geometry=gpd.points_from_xy(
                federal_gom_offshore_2011["lon"],
                federal_gom_offshore_2011["lat"],
                crs=4326
            )
        )
        .drop(columns=["lat", "lon"])
        .loc[:, ["boem_complex_id", "year", "month", "state_code", "Emis_tg", "geometry", "oil_gas_defn"]]
    )

    # Separate out ng and oil
    ng_federal_gom_offshore_2011_gdf = (federal_gom_offshore_2011_gdf
                                        .query("oil_gas_defn == 'Gas'")
                                        .assign(rel_emi=lambda df: df.groupby(["state_code", "year"])['Emis_tg'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0))
                                        .drop(columns={'Emis_tg', 'oil_gas_defn'})
                                        .reset_index(drop=True)
                                        )
    oil_federal_gom_offshore_2011_gdf = (federal_gom_offshore_2011_gdf
                                         .query("oil_gas_defn == 'Oil'")
                                         .assign(rel_emi=lambda df: df.groupby(["state_code", "year"])['Emis_tg'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0))
                                         .drop(columns={'Emis_tg', 'oil_gas_defn'})
                                         .reset_index(drop=True)
                                         )

    # 2014 GOADS Data
    # Read In and Format 2014 BEOM Data
    GOADS_14_inputfile = str(GOADS_14_path)
    driver_str = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+GOADS_14_inputfile+';'''
    conn = pyodbc.connect(driver_str)
    GOADS_emissions = pd.read_sql("SELECT * FROM 2014_Gulfwide_Platform_20161102", conn)
    conn.close()

    # Format Emissions Data (clean lease data string)
    GOADS_emissions = GOADS_emissions[["X_COORDINATE", "Y_COORDINATE", "POLLUTANT_CODE",
                                       "EMISSIONS_VALUE", "MONTH", "COMPLEX_ID"]]
    GOADS_emissions = (GOADS_emissions
                       .query("POLLUTANT_CODE == 'CH4'")
                       .assign(Emis_tg = 0.0)
                       .assign(Emis_tg = lambda df: 9.0718474E-7 * df['EMISSIONS_VALUE']) #convert short tons to Tg
                       .rename(columns={"COMPLEX_ID": "boem_complex_id"})
                       .astype({"boem_complex_id": int})
                       .drop(columns={"POLLUTANT_CODE", "EMISSIONS_VALUE"})
                       .replace('', np.nan)
                       .dropna()
                       .reset_index(drop=True)
                       )

    # Select 2014 data from ERG complex crosswalk
    ERG_complex_crosswalk_2014 = ERG_complex_crosswalk.copy().query('year == 2014').reset_index(drop=True)

    # Join locations, emissions, and complex types together
    federal_gom_offshore_2014 = (GOADS_emissions
                                 .set_index("boem_complex_id")
                                 .join(ERG_complex_crosswalk_2014.set_index("boem_complex_id"))
                                 .reset_index()
                                 .astype({"MONTH": str})
                                 .assign(state_code='FO')
                                 .rename(columns={'X_COORDINATE': 'lon', 'Y_COORDINATE': 'lat', 'MONTH': 'month'})
                                 )
    
    # Correct months to be numeric digits
    month_to_mm_df = pd.DataFrame(
        {'month': ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December'],
         'mm': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
         })
    federal_gom_offshore_2014 = (federal_gom_offshore_2014
                                 .merge(month_to_mm_df, how='left')
                                 .drop(columns='month')
                                 .rename(columns={'mm': 'month'})
                                 )

    federal_gom_offshore_2014_gdf = (
        gpd.GeoDataFrame(
            federal_gom_offshore_2014,
            geometry=gpd.points_from_xy(
                federal_gom_offshore_2014["lon"],
                federal_gom_offshore_2014["lat"],
                crs=4326
            )
        )
        .drop(columns=["lat", "lon"])
        .loc[:, ["boem_complex_id", "year", "month", "state_code", "Emis_tg", "geometry", "oil_gas_defn"]]
    )

    # Separate out ng and oil
    ng_federal_gom_offshore_2014_gdf = (federal_gom_offshore_2014_gdf
                                        .query("oil_gas_defn == 'Gas'")
                                        .assign(rel_emi=lambda df: df.groupby(["state_code", "year"])['Emis_tg'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0))
                                        .drop(columns={'Emis_tg', 'oil_gas_defn'})
                                        .reset_index(drop=True)
                                        )
    oil_federal_gom_offshore_2014_gdf = (federal_gom_offshore_2014_gdf
                                         .query("oil_gas_defn == 'Oil'")
                                         .assign(rel_emi=lambda df: df.groupby(["state_code", "year"])['Emis_tg'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0))
                                         .drop(columns={'Emis_tg', 'oil_gas_defn'})
                                         .reset_index(drop=True)
                                         )

    # 2017 GOADS Data
    # Read In and Format 2017 BEOM Data
    GOADS_17_inputfile = str(GOADS_17_path)
    driver_str = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+GOADS_17_inputfile+';'''
    conn = pyodbc.connect(driver_str)
    GOADS_emissions = pd.read_sql("SELECT * FROM 2017_Gulfwide_Platform_20190705_CAP_GHG", conn)
    conn.close()

    # Format Emissions Data (clean lease data string)
    GOADS_emissions = GOADS_emissions[["X_COORDINATE", "Y_COORDINATE", "POLLUTANT_CODE",
                                       "EMISSIONS_VALUE", "Month", "COMPLEX_ID"]]
    GOADS_emissions = (GOADS_emissions
                       .query("POLLUTANT_CODE == 'CH4'")
                       .assign(Emis_tg = 0.0)
                       .assign(Emis_tg = lambda df: 9.0718474E-7 * df['EMISSIONS_VALUE']) #convert short tons to Tg
                       .rename(columns={"COMPLEX_ID": "boem_complex_id"})
                       .astype({"boem_complex_id": int})
                       .drop(columns={"POLLUTANT_CODE", "EMISSIONS_VALUE"})
                       .replace('', np.nan)
                       .dropna()
                       .reset_index(drop=True)
                       )

    # Select 2017 data from ERG complex crosswalk
    ERG_complex_crosswalk_2017 = ERG_complex_crosswalk.copy().query('year == 2017').reset_index(drop=True)

    # Join locations, emissions, and complex types together
    federal_gom_offshore_2017 = (GOADS_emissions
                                 .set_index("boem_complex_id")
                                 .join(ERG_complex_crosswalk_2017.set_index("boem_complex_id"))
                                 .reset_index()
                                 .astype({"Month": str})
                                 .assign(state_code='FO')
                                 .rename(columns={'X_COORDINATE': 'lon', 'Y_COORDINATE': 'lat', 'Month': 'month'})
                                 )
    
    # Correct months to be numeric digits
    month_to_mm_df = pd.DataFrame(
        {'month': ['January', 'February', 'March', 'April', 'May', 'June', 'July',
                   'August', 'September', 'October', 'November', 'December'],
         'mm': ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
         })
    federal_gom_offshore_2017 = (federal_gom_offshore_2017
                                 .merge(month_to_mm_df, how='left')
                                 .drop(columns='month')
                                 .rename(columns={'mm': 'month'})
                                 )

    federal_gom_offshore_2017_gdf = (
        gpd.GeoDataFrame(
            federal_gom_offshore_2017,
            geometry=gpd.points_from_xy(
                federal_gom_offshore_2017["lon"],
                federal_gom_offshore_2017["lat"],
                crs=4326
            )
        )
        .drop(columns=["lat", "lon"])
        .loc[:, ["boem_complex_id", "year", "month", "state_code", "Emis_tg", "geometry", "oil_gas_defn"]]
    )

    # Separate out ng and oil
    ng_federal_gom_offshore_2017_gdf = (federal_gom_offshore_2017_gdf
                                        .query("oil_gas_defn == 'Gas'")
                                        .assign(rel_emi=lambda df: df.groupby(["state_code", "year"])['Emis_tg'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0))
                                        .drop(columns={'Emis_tg', 'oil_gas_defn'})
                                        .reset_index(drop=True)
                                        )
    oil_federal_gom_offshore_2017_gdf = (federal_gom_offshore_2017_gdf
                                         .query("oil_gas_defn == 'Oil'")
                                         .assign(rel_emi=lambda df: df.groupby(["state_code", "year"])['Emis_tg'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0))
                                         .drop(columns={'Emis_tg', 'oil_gas_defn'})
                                         .reset_index(drop=True)
                                         )

    # Build complete proxy (2012-2022)
    ng_federal_gom_offshore_gdf = gpd.GeoDataFrame()
    oil_federal_gom_offshore_gdf = gpd.GeoDataFrame()
    for iyear in years:
        data_year = federal_gom_offshore_data_years[federal_gom_offshore_data_years['year'] == iyear]['goads_data'].values[0]
        if data_year == 2011:
            ng_temp_data = (ng_federal_gom_offshore_2011_gdf
                            .copy()
                            .assign(year = iyear)
                            .assign(year_month=lambda df: df['year'].astype(str)+'_'+df['month'])
                            )
            oil_temp_data = (oil_federal_gom_offshore_2011_gdf
                             .copy()
                             .assign(year = iyear)
                             .assign(year_month=lambda df: df['year'].astype(str)+'_'+df['month'])
                             )
        if data_year == 2014:
            ng_temp_data = (ng_federal_gom_offshore_2014_gdf
                            .copy()
                            .assign(year = iyear)
                            .assign(year_month=lambda df: df['year'].astype(str)+'_'+df['month'])
                            )
            oil_temp_data = (oil_federal_gom_offshore_2014_gdf
                             .copy()
                             .assign(year = iyear)
                             
                             )
        if data_year == 2017:
            ng_temp_data = (ng_federal_gom_offshore_2017_gdf
                            .copy()
                            .assign(year = iyear)
                            .assign(year_month=lambda df: df['year'].astype(str)+'_'+df['month'])
                            )
            oil_temp_data = (oil_federal_gom_offshore_2017_gdf
                             .copy()
                             .assign(year = iyear)
                             .assign(year_month=lambda df: df['year'].astype(str)+'_'+df['month'])
                             )
        ng_federal_gom_offshore_gdf = pd.concat([ng_federal_gom_offshore_gdf, ng_temp_data])
        oil_federal_gom_offshore_gdf = pd.concat([oil_federal_gom_offshore_gdf, oil_temp_data])
    
    ng_federal_gom_offshore_gdf = (ng_federal_gom_offshore_gdf
                                   .loc[:, ["boem_complex_id", "year", "month", 
                                            "year_month", "state_code", "geometry", 
                                            "rel_emi"]]
                                   .astype({'year': int})
                                   .reset_index(drop=True)
                                   )
    oil_federal_gom_offshore_gdf = (oil_federal_gom_offshore_gdf
                                   .loc[:, ["boem_complex_id", "year", "month", 
                                            "year_month", "state_code", "geometry", 
                                            "rel_emi"]]
                                    .astype({'year': int})                                   
                                    .reset_index(drop=True)
                                    )

    # Correct for missing proxy data
    # 1. Find missing state_code-year pairs
    # 2. Check to see if proxy data exists for state in another year
    #   2a. If the data exists, use proxy data from the closest year
    #   2b. If the data does not exist, assign emissions uniformly across the state

    # Read in emissions data and drop states with 0 emissions
    ng_emi_df = (pd.read_csv(federal_gom_offshore_emi_path)
                          .query("state_code.isin(@state_gdf['state_code'])")
                          .query("ghgi_ch4_kt > 0.0")
                          )
    oil_emi_df = (pd.read_csv(oil_gom_federal_emi_path)
                          .query("state_code.isin(@state_gdf['state_code'])")
                          .query("ghgi_ch4_kt > 0.0")
                          )

    # Retrieve unique state codes for emissions without proxy data
    # This step is necessary, as not all emissions data excludes emission-less states
    ng_emi_states = set(ng_emi_df[['state_code', 'year']].itertuples(index=False, name=None))
    ng_proxy_states = set(ng_federal_gom_offshore_gdf[['state_code', 'year']].itertuples(index=False, name=None))
    oil_emi_states = set(oil_emi_df[['state_code', 'year']].itertuples(index=False, name=None))
    oil_proxy_states = set(oil_federal_gom_offshore_gdf[['state_code', 'year']].itertuples(index=False, name=None))

    # Find missing states
    ng_missing_states = ng_emi_states.difference(ng_proxy_states)
    oil_missing_states = oil_emi_states.difference(oil_proxy_states)

    # Add missing states alternative data to grouped_proxy
    ng_proxy_gdf_final = create_alt_proxy(ng_missing_states, ng_federal_gom_offshore_gdf)
    oil_proxy_gdf_final = create_alt_proxy(oil_missing_states, oil_federal_gom_offshore_gdf)

    ng_proxy_gdf_final.to_parquet(ng_output_path)
    oil_proxy_gdf_final.to_parquet(oil_output_path)

    return None
