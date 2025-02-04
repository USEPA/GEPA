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
import shapefile as shp
from pytask import Product, task, mark

from gch4i.config import (
    V3_DATA_PATH,
    proxy_data_dir_path,
    global_data_dir_path,
    sector_data_dir_path,
    max_year,
    min_year,
    years,
)

from gch4i.utils import us_state_to_abbrev
from gch4i.proxy_processing.ng_oil_production_utils import (
    calc_enverus_rel_emi,
    enverus_df_to_gdf,
    nei_data_years,
    get_nei_file_name,
    ng_water_prod_file_names,
    get_raw_NEI_data,
)

# %%
@mark.persist
@task(id="ng_water_prod_proxy")
def task_get_ng_water_prod_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    enverus_production_path: Path = sector_data_dir_path / "enverus/production",
    intermediate_outputs_path: Path = enverus_production_path / "intermediate_outputs",
    nei_path: Path = sector_data_dir_path / "nei_og",
    water_prod_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_water_prod_proxy.parquet",
    ):
    """
    Data come from Enverus, both Drilling Info and Prism
    The reason 2 datasets are used is because Prism does not include all states
    So remaining states, or those with more DI coverage are taken from DI

    DI: KS, MD, MI, MO, OK, TN

    Prism: AK, AL, AR, AZ, CA, CAO (California Offshore), CO, FL, KY, LA, MS, MT, ND,
    NE, NGOM (federal offshore waters in the Gulf of Mexico), NM, NV, NY, OH, OR, PA,
    SD, TX, UT, VA, WV, WY

    States with no Enverus Data: CT, DE, DC, GA, HI, ID, IL*, IN*, IA, ME, MA, MN, NH,
    NJ, NC, RI, SC, VT, WA, WI, US territories. These states are assumed to have no oil
    and gas production with an exception for IL and IN.

    *IL and IN do not report to Enverus, but do have oil and gas production. Production
    data is taken from the Energy Information Administration (EIA).

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

    # Make Annual gridded arrays (maps) of well data (a well will be counted every month if there is any production that year)
    # Includes NA Gas Wells and Production onshore in the CONUS region
    # source emissions are related to the presence of a well and its production status (no emission if no production)
    # Details: ERG does not include a well in the national count if there is no (cummulative) oil or gas production from that well.
    # Wells are not considered active for a given year if there is no production data that year
    # This may cause wells that are coadmpleted but not yet producing to be dropped from the national count. 
    # ERG has developed their own logic to determine if a well is an HF well or not and that result is included in the 
    # HF variable in this dataset. This method does not rely on the Enverus well 'Producing Status'
    # Well Type (e.g., non-associated gas well) is determined based on annual production GOR at that well (CUM OIL/ CUM GAS), 
    # but the presence of a well will only be included in maps in months where monthly gas prod > 0

    # Proxy Data Dataframes:
    water_prod_df = pd.DataFrame()

    ## Enverus DI and Prism Data: 
    # Read in and query formatted and corrrected Enverus data to create dictionaries of 
    # proxy data (Enverus data is from task_enverus_di_prism_data_processing.py)
    for iyear in years:
        enverus_file_name_iyear = f"formatted_raw_enverus_tempoutput_{iyear}.csv"
        enverus_file_path_iyear = os.path.join(intermediate_outputs_path, enverus_file_name_iyear)
        ng_data_temp = (pd.read_csv(enverus_file_path_iyear, dtype={3:'str', 'spud_year': str, 'first_prod_year': str})
                        .query("STATE_CODE.isin(@state_gdf['state_code'])")
                        .query("OFFSHORE == 'N'")
                        .query("CUM_GAS > 0")
                        .assign(gas_to_oil_ratio=lambda df: df['CUM_GAS']/df['CUM_OIL'])
                        .assign(year=str(iyear))
                        .replace(np.inf, 0)
                        .astype({"spud_year": str, "first_prod_year": str})
                        .query("gas_to_oil_ratio > 100 | GOR_QUAL == 'Gas only'")
                        )

        # Include wells in map only for months where there is gas production (emissions ~ when production is occuring)
        for imonth in range(1,13):
            imonth_str = f"{imonth:02}"  # convert to 2-digit months
            year_month_str = str(iyear)+'-'+imonth_str
            gas_prod_str = 'GASPROD_'+imonth_str
            water_prod_str = 'WATERPROD_'+imonth_str
            # Onshore data for imonth
            ng_data_imonth_temp = (ng_data_temp
                                   .query(f"{gas_prod_str} > 0")
                                   .assign(year_month=str(iyear)+'-'+imonth_str)
                                   )
            ng_data_imonth_temp = (ng_data_imonth_temp[[
                'year', 'year_month','STATE_CODE','AAPG_CODE_ERG','LATITUDE','LONGITUDE',
                'HF','WELL_COUNT',gas_prod_str,water_prod_str,
                'comp_year_month','spud_year','first_prod_year']]
                )
            # Water Production
            # Data Source by state defined in Enverus DrillingInfo Processing - Produced
            # Water_2023-11-14_forGridding.xlsx file.
            if iyear < 2016:  # WV uses NEI data
                water_prod_enverus_states = ['AK','AL','AR','AZ','CA','CO','FL','LA',
                                             'MI','MO','MS','MT','ND','NE','NM','NV',
                                             'NY','OH','SD','TX','UT','VA','WY'
                                             ]
                # States using NEI for reference: ['IL','IN','KS','OK','PA','WV']
            else:  # 2016 and beyond; WV uses Enverus data
                water_prod_enverus_states = ['AK','AL','AR','AZ','CA','CO','FL','LA',
                                             'MI','MO','MS','MT','ND','NE','NM','NV',
                                             'NY','OH','SD','TX','UT','VA','WY','WV'
                                             ]  #WV uses Enverus
                # States using NEI for reference: ['IL','IN','KS','OK','PA']
            # Enverus water production for applicable states (NEI water producted will
            # be added in the NEI section of the code below)
            water_prod_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE',water_prod_str]]
                                .query("STATE_CODE.isin(@water_prod_enverus_states)")
                                .assign(proxy_data=lambda df: df[water_prod_str])
                                .drop(columns=[water_prod_str])
                                .rename(columns=lambda x: str(x).lower())
                                .reset_index(drop=True)
                                )
            water_prod_df = pd.concat([water_prod_df,water_prod_imonth])

    # Delete unused temp data
    del ng_data_temp
    del ng_data_imonth_temp
    del water_prod_imonth

    # Calculate relative emissions and convert to a geodataframe   
    water_prod_df = calc_enverus_rel_emi(water_prod_df)    
    water_prod_df = enverus_df_to_gdf(water_prod_df)

    # NEI Data:
    nei_df = pd.DataFrame()

    for iyear in years:
        nei_data_year = nei_data_years[nei_data_years['year'] == iyear]['nei_data'].values[0]
        # Gas Production
        ifile_name = get_nei_file_name(nei_data_year, ng_water_prod_file_names)
        nei_iyear = get_raw_NEI_data(iyear, nei_data_year, ifile_name)
        nei_df = pd.concat([nei_df, nei_iyear])
    
    # Convert NEI Data to GDF and polygon to centroid point
    nei_df = gpd.GeoDataFrame(nei_df, crs=4326)
    nei_df = nei_df.to_crs(3857)  # projected CRS for centroid calculation
    nei_df.loc[:, 'geometry'] = nei_df.loc[:, 'geometry'].centroid
    nei_df = nei_df.to_crs(4326)
    
    # Add NEI Data to Enverus Data
    water_prod_df = pd.concat([water_prod_df, nei_df]).reset_index(drop=True)

    # Delete unused temp data
    del nei_iyear
    del nei_df

    # Check that relative emissions sum to 1.0 each state/year combination
    sums = water_prod_df.groupby(["state_code", "year"])["rel_emi"].sum()  # get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"  # assert that the sums are close to 1

    # Output Proxy Parquet Files
    water_prod_df = water_prod_df.astype({'year':str})
    water_prod_df.to_parquet(water_prod_output_path)

    return None
