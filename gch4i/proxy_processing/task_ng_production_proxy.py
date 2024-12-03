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

# %%
@mark.persist
@task(id="ng_production_proxy")
def task_get_ng_production_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    nems_region_dict_path: Path = sector_data_dir_path / "enverus/NEMS_Region_Dictionary.xlsx",
    enverus_production_path: Path = sector_data_dir_path / "enverus/production",
    enverus_well_counts_path: Path = sector_data_dir_path / "enverus/production/temp_data_v2/Enverus DrillingInfo Processing - Well Counts_2021-03-17.xlsx",
    nei_path: Path = sector_data_dir_path / "nei_og",
    all_well_count_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_all_well_count_proxy.parquet",
    conv_well_count_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_conv_well_count_proxy.parquet",
    hf_well_count_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_hf_well_count_proxy.parquet",
    all_well_prod_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_all_well_prod_proxy.parquet",
    basin_220_prod_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_basin_220_prod_proxy.parquet",
    basin_395_prod_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_basin_395_prod_proxy.parquet",
    basin_430_prod_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_basin_430_prod_proxy.parquet",
    basin_other_prod_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_basin_other_prod_proxy.parquet",
    water_prod_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_water_prod_proxy.parquet",
    conv_well_comp_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_conv_well_comp_proxy.parquet",
    hf_well_comp_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_hf_well_comp_proxy.parquet",
    drilled_well_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_drilled_well_proxy.parquet",
    state_gom_offshore_well_count_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_state_gom_offshore_well_count_proxy.parquet",
    state_gom_offshore_well_prod_output_path: Annotated[Path, Product] = proxy_data_dir_path / "ng_state_gom_offshore_well_prod_proxy.parquet",
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

    TODO: Update enverus_well_counts_path with v3 data (currently using v2 data)
    """

    # Functions:
    # Define safe devide to set result to zero if denominator is zero
    def safe_div(x,y):
        if y == 0:
            return 0
        return x / y

    # STEP 1: Load in State ANSI data and NEMS definitions

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

    # Make NEMS State classifications
    # Treat NM and TX separately since these states cover multiple NEMS regions

    # 0 = NE, 1 = MC, 2 = RM, 3 = SW, 4 = WC, 5 = GC, 6 = offshore
    NEMS_State = pd.read_excel(nems_region_dict_path)
    NEMS_State = NEMS_State.fillna(0)
    NM_idx = NEMS_State.index[NEMS_State['State_Name'].str.contains('New Mexico')].tolist()
    TX_idx = NEMS_State.index[NEMS_State['State_Name'].str.contains('Texas')].tolist()
    idx = NM_idx+TX_idx
    NEMS_State= NEMS_State.drop(NEMS_State.index[idx])
    NEMS_State.reset_index(drop=True,inplace=True)

    NEMS_dict = {'North East':0, 'Midcontinent':1,'Rocky Mountain':2,'South West':3,'West Coast':4,'Gulf Coast':5}

    # STEP 2: Read-in and Format Proxy Data

    # STEP 2.1: State Condensate Data

    # TODO: state condensate data code

    # STEP 2.2: GOADS Emissions Data

    # TODO: GOADS emissions data code

    # STEP 2.3: Well and Production Data (from Enverus)
    
    # STEP 2.3.1: Read In & Combine Each Year of Prism & DI Monthly Data (from Enverus)

    # Data come from Enverus, both Drilling Info and Prism
    # The reason 2 datasets are used is because Prism does not include all states
    # So remaining states, or those with more DI coverage are taken from DI

    # Read In and Format the Prism and DI data
    # 1. Read Data
    # 2. Drop unsed columns, rename columns to match between DI and Prism
    # 3. Combine DI and Prism into one data array
    # 4. Calculate annual cummulate production totals
    # 5. Save the data as a year-specific variable

    # Based on ERGs logic, active wells are determined based on their production levels and not producing status
    Enverus_data_dict = {}
    DI_data_dict = {}
    Prism_data_dict = {}
    for iyear in years:
        #DI data
        DI_file_name = f"didsk_monthly_{iyear}.csv"
        DI_file_path = os.path.join(enverus_production_path, DI_file_name)
        DI_data = (pd.read_csv(
            DI_file_path,
            usecols=['WELL_COUNT_ID','STATE','COUNTY','BASIN','AAPG_CODE_ERG',
                     'NEMS_REGION_ERG','LATITUDE','LONGITUDE','STATUS','COMPDATE',
                     'SPUDDATE','FIRSTPRODDATE','HF','OFFSHORE','GOR',
                     'GOR_QUAL','PROD_FLAG','PRODYEAR',
                     'LIQ_01','GAS_01','WTR_01','LIQ_02','GAS_02','WTR_02',
                     'LIQ_03','GAS_03','WTR_03','LIQ_04','GAS_04','WTR_04',
                     'LIQ_05','GAS_05','WTR_05','LIQ_06','GAS_06','WTR_06',
                     'LIQ_07','GAS_07','WTR_07','LIQ_08','GAS_08','WTR_08',
                     'LIQ_09','GAS_09','WTR_09','LIQ_10','GAS_10','WTR_10',
                     'LIQ_11','GAS_11','WTR_11','LIQ_12','GAS_12','WTR_12',],
            dtype={7:'str'})
            .rename(columns={'WELL_COUNT_ID':'WELL_COUNT','STATE':'STATE_CODE',
                             'NEMS_REGION_ERG':'NEMS_REGION', 'STATUS':'PRODUCING_STATUS',
                             'LIQ_01':'OILPROD_01','GAS_01':'GASPROD_01','WTR_01':'WATERPROD_01',
                             'LIQ_02':'OILPROD_02','GAS_02':'GASPROD_02','WTR_02':'WATERPROD_02',
                             'LIQ_03':'OILPROD_03','GAS_03':'GASPROD_03','WTR_03':'WATERPROD_03',
                             'LIQ_04':'OILPROD_04','GAS_04':'GASPROD_04','WTR_04':'WATERPROD_04',
                             'LIQ_05':'OILPROD_05','GAS_05':'GASPROD_05','WTR_05':'WATERPROD_05',
                             'LIQ_06':'OILPROD_06','GAS_06':'GASPROD_06','WTR_06':'WATERPROD_06',
                             'LIQ_07':'OILPROD_07','GAS_07':'GASPROD_07','WTR_07':'WATERPROD_07',
                             'LIQ_08':'OILPROD_08','GAS_08':'GASPROD_08','WTR_08':'WATERPROD_08',
                             'LIQ_09':'OILPROD_09','GAS_09':'GASPROD_09','WTR_09':'WATERPROD_09',
                             'LIQ_10':'OILPROD_10','GAS_10':'GASPROD_10','WTR_10':'WATERPROD_10',
                             'LIQ_11':'OILPROD_11','GAS_11':'GASPROD_11','WTR_11':'WATERPROD_11',
                             'LIQ_12':'OILPROD_12','GAS_12':'GASPROD_12','WTR_12':'WATERPROD_12',})
            .assign(WELL_COUNT=1)  # TODO: Check to see if this should actually be set to 1
            )
        # Format completion date (YYYY-MM)
        for iwell in range(0,len(DI_data)):
            comp_date = str(DI_data.loc[iwell, 'COMPDATE'])
            if comp_date == 'NaN':
                comp_year_month = 'NaN'
            elif comp_date == 'nan':
                comp_year_month = 'NaN'
            else:  # date format M/DD/YYYY
                comp_month = f"{int(comp_date.split('/')[0]):02}"
                comp_year = f"{int(comp_date.split('/')[2])}"
                comp_year_month = str(comp_year)+'-'+str(comp_month)
            DI_data.loc[iwell, 'comp_year_month'] = comp_year_month
        # Format spud date (YYYY)
        for iwell in range(0,len(DI_data)):
            spud_date = str(DI_data.loc[iwell, 'SPUDDATE'])
            if spud_date == 'NaN':
                spud_year = 'NaN'
            elif spud_date == 'nan':
                spud_year = 'NaN'
            else:  # date format M/DD/YYYY
                spud_year = f"{int(spud_date.split('/')[2])}"
                spud_year = str(spud_year)
            DI_data.loc[iwell, 'spud_year'] = spud_year
        # Format first production date (YYYY)
        for iwell in range(0,len(DI_data)):
            first_prod_date = str(DI_data.loc[iwell, 'FIRSTPRODDATE'])
            if first_prod_date == 'NaN':
                first_prod_year = 'NaN'
            elif first_prod_date == 'nan':
                first_prod_year = 'NaN'
            else:  # date format M/DD/YYYY
                first_prod_year = f"{int(first_prod_date.split('/')[2])}"
                first_prod_year = str(first_prod_year)
            DI_data.loc[iwell, 'first_prod_year'] = first_prod_year
        DI_data_dict[f'{iyear}'] = DI_data

        # Prism Data
        Prism_file_name = f"prism_monthly_{iyear}.csv"
        Prism_file_path = os.path.join(enverus_production_path, Prism_file_name)
        Prism_data = (pd.read_csv(
            Prism_file_path,
            usecols=['STATE','COUNTY','ENVBASIN','AAPG_CODE_ERG',
                     'NEMS_REGION_ERG','LATITUDE','LONGITUDE','ENVWELLSTATUS','COMPLETIONDATE',
                     'SPUDDATE','FIRSTPRODDATE','HF','OFFSHORE','GOR',
                     'GOR_QUAL','PROD_FLAG','PRODYEAR',
                     'LIQUIDSPROD_BBL_01','GASPROD_MCF_01','WATERPROD_BBL_01',
                     'LIQUIDSPROD_BBL_02','GASPROD_MCF_02','WATERPROD_BBL_02',
                     'LIQUIDSPROD_BBL_03','GASPROD_MCF_03','WATERPROD_BBL_03',
                     'LIQUIDSPROD_BBL_04','GASPROD_MCF_04','WATERPROD_BBL_04',
                     'LIQUIDSPROD_BBL_05','GASPROD_MCF_05','WATERPROD_BBL_05',
                     'LIQUIDSPROD_BBL_06','GASPROD_MCF_06','WATERPROD_BBL_06',
                     'LIQUIDSPROD_BBL_07','GASPROD_MCF_07','WATERPROD_BBL_07',
                     'LIQUIDSPROD_BBL_08','GASPROD_MCF_08','WATERPROD_BBL_08',
                     'LIQUIDSPROD_BBL_09','GASPROD_MCF_09','WATERPROD_BBL_09',
                     'LIQUIDSPROD_BBL_10','GASPROD_MCF_10','WATERPROD_BBL_10',
                     'LIQUIDSPROD_BBL_11','GASPROD_MCF_11','WATERPROD_BBL_11',
                     'LIQUIDSPROD_BBL_12','GASPROD_MCF_12','WATERPROD_BBL_12',],
            dtype={7:'str'})
            .rename(columns={'STATE':'STATE_CODE', 'ENVBASIN':'BASIN',
                             'NEMS_REGION_ERG':'NEMS_REGION', 'ENVWELLSTATUS':'PRODUCING_STATUS',
                             'COMPLETIONDATE':'COMPDATE',
                             'LIQUIDSPROD_BBL_01':'OILPROD_01','GASPROD_MCF_01':'GASPROD_01','WATERPROD_BBL_01':'WATERPROD_01',
                             'LIQUIDSPROD_BBL_02':'OILPROD_02','GASPROD_MCF_02':'GASPROD_02','WATERPROD_BBL_02':'WATERPROD_02',
                             'LIQUIDSPROD_BBL_03':'OILPROD_03','GASPROD_MCF_03':'GASPROD_03','WATERPROD_BBL_03':'WATERPROD_03',
                             'LIQUIDSPROD_BBL_04':'OILPROD_04','GASPROD_MCF_04':'GASPROD_04','WATERPROD_BBL_04':'WATERPROD_04',
                             'LIQUIDSPROD_BBL_05':'OILPROD_05','GASPROD_MCF_05':'GASPROD_05','WATERPROD_BBL_05':'WATERPROD_05',
                             'LIQUIDSPROD_BBL_06':'OILPROD_06','GASPROD_MCF_06':'GASPROD_06','WATERPROD_BBL_06':'WATERPROD_06',
                             'LIQUIDSPROD_BBL_07':'OILPROD_07','GASPROD_MCF_07':'GASPROD_07','WATERPROD_BBL_07':'WATERPROD_07',
                             'LIQUIDSPROD_BBL_08':'OILPROD_08','GASPROD_MCF_08':'GASPROD_08','WATERPROD_BBL_08':'WATERPROD_08',
                             'LIQUIDSPROD_BBL_09':'OILPROD_09','GASPROD_MCF_09':'GASPROD_09','WATERPROD_BBL_09':'WATERPROD_09',
                             'LIQUIDSPROD_BBL_10':'OILPROD_10','GASPROD_MCF_10':'GASPROD_10','WATERPROD_BBL_10':'WATERPROD_10',
                             'LIQUIDSPROD_BBL_11':'OILPROD_11','GASPROD_MCF_11':'GASPROD_11','WATERPROD_BBL_11':'WATERPROD_11',
                             'LIQUIDSPROD_BBL_12':'OILPROD_12','GASPROD_MCF_12':'GASPROD_12','WATERPROD_BBL_12':'WATERPROD_12',})
            .assign(WELL_COUNT=1)
            )
        # Format completion date (YYYY-MM)
        for iwell in range(0,len(Prism_data)):
            comp_date = str(Prism_data.loc[iwell, 'COMPDATE'])
            if comp_date == 'NaN':
                comp_year_month = 'NaN'
            elif comp_date == 'nan':
                comp_year_month = 'NaN'
            else:  # date format YYYY-MM-DD
                comp_month = f"{int(comp_date.split('-')[1]):02}"
                comp_year = f"{int(comp_date.split('-')[0])}"
                comp_year_month = str(comp_year)+'-'+str(comp_month)
            Prism_data.loc[iwell, 'comp_year_month'] = comp_year_month
        # Format spud date (YYYY)
        for iwell in range(0,len(Prism_data)):
            spud_date = str(Prism_data.loc[iwell, 'SPUDDATE'])
            if spud_date == 'NaN':
                spud_year = 'NaN'
            elif spud_date == 'nan':
                spud_year = 'NaN'
            else:  # date format YYYY-MM-DD
                spud_year = f"{int(spud_date.split('-')[0])}"
                spud_year = str(spud_year)
            Prism_data.loc[iwell, 'spud_year'] = spud_year
        # Format first production date (YYYY)
        for iwell in range(0,len(Prism_data)):
            first_prod_date = str(Prism_data.loc[iwell, 'FIRSTPRODDATE'])
            if first_prod_date == 'NaN':
                first_prod_year = 'NaN'
            elif first_prod_date == 'nan':
                first_prod_year = 'NaN'
            else:  # date format YYYY-MM-DD
                first_prod_year = f"{int(first_prod_date.split('-')[0])}"
                first_prod_year = str(first_prod_year)
            Prism_data.loc[iwell, 'first_prod_year'] = first_prod_year
        Prism_data_dict[f'{iyear}'] = Prism_data
        
        # Combine into one array with common column names, replace nans with zeros, and sum annual production
        Enverus_data = pd.concat([DI_data, Prism_data], ignore_index=True)
        Enverus_data.loc[:, Enverus_data.columns.str.contains('GASPROD_')] = Enverus_data.loc[:, Enverus_data.columns.str.contains('GASPROD_')].fillna(0)
        Enverus_data.loc[:, Enverus_data.columns.str.contains('OILPROD_')] = Enverus_data.loc[:, Enverus_data.columns.str.contains('OILPROD_')].fillna(0)
        Enverus_data.loc[:, Enverus_data.columns.str.contains('WATERPROD_')] = Enverus_data.loc[:, Enverus_data.columns.str.contains('WATERPROD_')].fillna(0)

        # Calculate cummulative annual production totals for Gas, Oil, Water
        Enverus_data['CUM_GAS'] = Enverus_data.loc[:,Enverus_data.columns.str.contains('GASPROD_')].sum(1)
        Enverus_data['CUM_OIL'] = Enverus_data.loc[:,Enverus_data.columns.str.contains('OILPROD_')].sum(1)
        Enverus_data['CUM_WATER'] = Enverus_data.loc[:,Enverus_data.columns.str.contains('WATERPROD_')].sum(1)
        
        Enverus_data['NEMS_CODE'] = Enverus_data['NEMS_REGION'].map(NEMS_dict)
        
        # Save out the data for that year
        Enverus_data_dict[f'{iyear}'] = Enverus_data

        del Prism_data
        del DI_data #save memory space 
        
        #define default values for a new row in this table (to be used later during data corrections)
        default = {'WELL_COUNT': 0, 'STATE_CODE':'','COUNTY':'','NEMS_REGION':'UNK',
                   'AAPG_CODE_ERG':'UNK','LATITUDE':0,'LONGITUDE':0,
                   'PRODUCING_STATUS':'','BASIN':'','SPUDDATE':'','COMPDATE':'',
                   'FIRSTPRODDATE':'','HF':'', 'OFFSHORE':'','GOR':-99,
                   'GOR_QUAL':'','PROD_FLAG':'','PRODYEAR':'',
                   'OILPROD_01':0, 'GASPROD_01':0, 'WATERPROD_01':0,'OILPROD_02':0, 'GASPROD_02':0, 'WATERPROD_02':0,
                   'OILPROD_03':0, 'GASPROD_03':0, 'WATERPROD_03':0,'OILPROD_04':0, 'GASPROD_04':0, 'WATERPROD_04':0,\
                   'OILPROD_05':0, 'GASPROD_05':0, 'WATERPROD_05':0,'OILPROD_06':0, 'GASPROD_06':0, 'WATERPROD_06':0,\
                   'OILPROD_07':0, 'GASPROD_07':0, 'WATERPROD_07':0,'OILPROD_08':0, 'GASPROD_08':0, 'WATERPROD_08':0,\
                   'OILPROD_09':0, 'GASPROD_09':0, 'WATERPROD_09':0,'OILPROD_10':0, 'GASPROD_10':0, 'WATERPROD_10':0,\
                   'OILPROD_11':0, 'GASPROD_11':0, 'WATERPROD_11':0,'OILPROD_12':0, 'GASPROD_12':0, 'WATERPROD_12':0,
                   'CUM_GAS':0, 'CUM_OIL':0, 'CUM_WATER':0, 'NEMS_CODE':99}

    # Correct the NEMS Code for missing NEMS_REGIONS
    # Note OFFSHORE regions will have NaN as NEMS_Code
    for iyear in years:
        enverus_data_temp = Enverus_data_dict[f'{iyear}']
        list_well = enverus_data_temp.index[pd.isna(enverus_data_temp.loc[:,'NEMS_REGION'])].tolist()
        if np.size(list_well) > 0:
            for irow in list_well: 
                match_state = np.where(NEMS_State['State_Code']==enverus_data_temp['STATE_CODE'][irow])[0][0]
                enverus_data_temp.loc[irow,'NEMS_CODE'] = NEMS_State['NEMS'][match_state].astype(int)
        Enverus_data_dict[f'{iyear}'] = enverus_data_temp.copy()

    # STEP 2.3.2: Correct Enverus Data for Select States

    # 1) Read In Coverage Table from State Well Counts File from ERG
    # (specifies the first year with bad data and which years need to be corrected; 
    # all years including and after the first bad year of data need to be corrected)

    ERG_StateWellCounts_LastGoodDataYear = (pd.read_excel(
        enverus_well_counts_path,
        sheet_name = "2021 - Coverage",
        usecols = {"State","Last Good Year"},
        skiprows = 2,
        nrows = 40)
        )

    # 2) Loops through the each state and year in Enverus to determine if the data for that particualar year needs to 
    # be corrected. At the moment, the only corrections ERG makes to the data is to use the prior year of data if there
    # is no new Enverus data reportd for that state. If a particular state is not included for any years in the Enverus
    # dataset, then a row of zeros is added to the Enverus table for that year.
         
    for istate in np.arange(0,len(state_gdf)):
        correctdata =0
        istate_code = state_gdf['state_code'][istate]
        lastgoodyear = ERG_StateWellCounts_LastGoodDataYear['Last Good Year'][ERG_StateWellCounts_LastGoodDataYear['State'] == istate_code].values
        if lastgoodyear  == max_year:
            lastgoodyear = max_year+5 #if state isn't included in correction list, don't correct any data
        
        for iyear in years:
            enverus_data_temp= Enverus_data_dict[f'{iyear}'].copy()
            state_list = np.unique(enverus_data_temp['STATE_CODE'])
            if istate_code in state_list:
                inlist =1
            else:
                inlist = 0
            if inlist ==1 or correctdata==1: #if the state is included in Enverus data, or had data for at least one good year
                #if first year, correctdata will be zero, but inlist will also be zero if no Enverus data
                #check to see whether corrections are necessary for the given year/state
                if iyear == (lastgoodyear):
                    print(istate_code,iyear,'last good year')
                    # This is the last year of good data. Do not correct the data but save
                    # but so that this data can be used for all following years for that state
                    temp_data = enverus_data_temp[enverus_data_temp['STATE_CODE'] == istate_code]
                    correctdata=1
                elif iyear > lastgoodyear: 
                    print(istate_code,iyear)
                    #correct data for all years equal to and after the first bad year (remove old data first if necessary)
                    if inlist == 1:
                        enverus_data_temp = enverus_data_temp[enverus_data_temp['STATE_CODE'] != istate_code]
                    enverus_data_temp = pd.concat([enverus_data_temp,temp_data],ignore_index=True)
                    print(istate_code +' data for ' +str(iyear) +' were corrected with '+str(lastgoodyear)+' data')
                else:
                    # year_range[iyear] < firstbadyear:
                    #no data corrections if the current year is before the first bad year
                    #print('no corrections')
                    #print(state_str,year_range[iyear])
                    no_corrections =1
                    
            if inlist==0 and correctdata==0:
            #if there is no Enverus data for a given state, and there was no good data, add a row with default values
                # temp_row = {'STATE':istate_code}
                # enverus_data_temp = enverus_data_temp.append({**default,**temp_row}, ignore_index=True)
                print(istate_code +' has no Enverus data in the year ' +str(iyear))
                
            #resave that year of Enverus data
            enverus_data_temp.reset_index(drop=True,inplace=True)
            Enverus_data_dict[f'{iyear}'] = enverus_data_temp.copy()

    # STEP 2.4: Calculate Fractional Monthly Condensate Arrays
    # (EIA condensate production (bbl) relative to producing Enverus gas wells by month
    # in each state and region)

    # TODO: fractional monthly condensate array code

    # STEP 2.5: Convert Enverus Well Production Arrays and Condensate Array into Gridded
    # Location Arrays

    # clear variables
    # del ERG_StateWellCounts_FirstBadDataYear
    # del Prism_data
    # del colnames
    # del names
    # del state_condensates
    # del temp_data

    # Make Annual gridded arrays (maps) of well data (a well will be counted every month if there is any production that year)
    # Includes NA Gas Wells and Production onshore in the CONUS region
    # source emissions are related to the presence of a well and its production status (no emission if no production)
    # Details: ERG does not include a well in the national count if there is no (cummulative) oil or gas production from that well.
    # Wells are not considered active for a given year if there is no production data that year
    # This may cause wells that are completed but not yet producing to be dropped from the national count. 
    # ERG has developed their own logic to determine if a well is an HF well or not and that result is included in the 
    # HF variable in this dataset. This method does not rely on the Enverus well 'Producing Status'
    # Well Type (e.g., non-associated gas well) is determined based on annual production GOR at that well (CUM OIL/ CUM GAS), 
    # but the presence of a well will only be included in maps in months where monthly gas prod > 0

    # Proxy Data Dataframes:
    # Well Counts
    all_well_count_df = pd.DataFrame()  # Active gas well (conventional + HF) counts in a given month
    conv_well_count_df = pd.DataFrame()  # Active conventional gas well counts in a given month
    hf_well_count_df = pd.DataFrame()  # Active HF gas well counts in a given month
    # Well-Level Production Volumes
    all_well_prod_df = pd.DataFrame()  # Active gas well (conventional + HF) gas production in a given month
    basin_220_prod_df = pd.DataFrame()  # Gas well gas production in Basin 220 in a given month
    basin_395_prod_df = pd.DataFrame()  # Gas well gas production in Basin 395 in a given month
    basin_430_prod_df = pd.DataFrame()  # Gas well gas production in Basin 430 in a given month
    basin_other_prod_df = pd.DataFrame()  # Gas well gas production in Other Basins in a given month
    # Water Production Volumes
    water_prod_df = pd.DataFrame()
    # Well Completions
    conv_well_comp_df = pd.DataFrame()  # Conventional gas well completions
    hf_well_comp_df = pd.DataFrame()  # HF gas well completions
    # Drilled Gas Wells
    drilled_well_df = pd.DataFrame()  # Gas wells drilled
    # Offshore Well Counts and Production Volumes in State Waters in the Gulf of Mexico
    state_gom_offshore_well_count_df = pd.DataFrame()  # Offshore state GOM gas well counts
    state_gom_offshore_well_prod_df = pd.DataFrame()  # Offshore state GOM gas production


    # Query Enverus data to create dictionaries of proxy data
    for iyear in years:
        enverus_data_temp = Enverus_data_dict[f'{iyear}'].copy()
        
        # Onshore Natural Gas
        ng_data_temp = (enverus_data_temp
                        .query("STATE_CODE.isin(@state_gdf['state_code'])")
                        .query("OFFSHORE == 'N'")
                        .query("CUM_GAS > 0")
                        .assign(gas_to_oil_ratio=lambda df: df['CUM_GAS']/df['CUM_OIL'])
                        .assign(year=str(iyear))
                        .replace(np.inf, 0)
                        .query("gas_to_oil_ratio > 100 | GOR_QUAL == 'Gas only'")
                        )
        # Offshore Natural Gas Wells
        ng_offshore_data_temp = (enverus_data_temp
                                 .query("STATE_CODE.isin(@state_gdf['state_code'])")
                                 .query("OFFSHORE == 'Y'")
                                 .query("CUM_GAS > 0")
                                 .assign(gas_to_oil_ratio=lambda df: df['CUM_GAS']/df['CUM_OIL'])
                                 .assign(year=str(iyear))
                                 .replace(np.inf, 0)
                                 .query("gas_to_oil_ratio > 100 | GOR_QUAL == 'Gas only'")
                                 )
        
        # Include wells in map only for months where there is gas production (emissions ~ when production is occuring)
        for imonth in range(1,13):
            imonth_str = f"{imonth:02}"  # convert to 2-digit months
            year_month_str = str(iyear)+'-'+imonth_str
            gas_prod_str = 'GASPROD_'+imonth_str
            water_prod_str = 'WATERPROD_'+imonth_str
            # onshore data for imonth
            ng_data_imonth_temp = (ng_data_temp
                                   .query(f"{prod_str} > 0")
                                   .assign(year_month=str(iyear)+'-'+imonth_str)
                                   )
            ng_data_imonth_temp = (ng_data_imonth_temp[[
                'year', 'year_month','STATE_CODE','AAPG_CODE_ERG','LATITUDE','LONGITUDE',
                'HF','WELL_COUNT',gas_prod_str,water_prod_str,
                'comp_year_month','spud_year','first_prod_year']]
                )
            # offshore data for imonth
            ng_offshore_data_imonth_temp = (ng_offshore_data_temp
                                            .query(f"{prod_str} > 0")
                                            .assign(year_month=str(iyear)+'-'+imonth_str)
                                            )
            ng_data_imonth_temp = (ng_offshore_data_imonth_temp[[
                'year','year_month','STATE_CODE','AAPG_CODE_ERG','LATITUDE','LONGITUDE',
                'HF','WELL_COUNT',gas_prod_str,water_prod_str,
                'comp_year_month','spud_year','first_prod_year']]
                )
            # Well Counts
            # All Gas Well Count
            all_well_count_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','WELL_COUNT']]
                                    .rename(columns=lambda x: str(x).lower())
                                    .rename(columns={"well_count":"proxy_data"})
                                    .reset_index(drop=True)
                                    )
            all_well_count_df = pd.concat([all_well_count_df,all_well_count_imonth])
            # Conventional Gas Well Count
            conv_well_count_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','WELL_COUNT','HF']]
                                     .query("HF != 'Y'")
                                     .drop(columns=["HF"])
                                     .rename(columns=lambda x: str(x).lower())
                                     .rename(columns={"well_count":"proxy_data"})
                                     .reset_index(drop=True)
                                     )
            conv_well_count_df = pd.concat([conv_well_count_df,conv_well_count_imonth])
            # HF Gas Well Count
            hf_well_count_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','WELL_COUNT','HF']]
                                   .query("HF == 'Y'")
                                   .drop(columns=["HF"])
                                   .rename(columns=lambda x: str(x).lower())
                                   .rename(columns={"well_count":"proxy_data"})
                                   .reset_index(drop=True)
                                   )
            hf_well_count_df = pd.concat([hf_well_count_df,hf_well_count_imonth])

            # Gas Production
            # All Gas Well Gas Production
            all_well_prod_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE',gas_prod_str]]
                                   .assign(proxy_data=lambda df: df[gas_prod_str])
                                   .drop(columns=[gas_prod_str])
                                   .rename(columns=lambda x: str(x).lower())
                                   .reset_index(drop=True)
                                   )
            all_well_prod_df = pd.concat([all_well_prod_df,all_well_prod_imonth])
            # Basin 220 Gas Well Gas Production
            basin_220_prod_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','AAPG_CODE_ERG',gas_prod_str]]
                                    .query("AAPG_CODE_ERG == '220'")
                                    .assign(proxy_data=lambda df: df[gas_prod_str])
                                    .drop(columns=[gas_prod_str, 'AAPG_CODE_ERG'])
                                    .rename(columns=lambda x: str(x).lower())
                                    .reset_index(drop=True)
                                    )
            basin_220_prod_df = pd.concat([basin_220_prod_df,basin_220_prod_imonth])
            # Basin 395 Gas Well Gas Production
            basin_395_prod_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','AAPG_CODE_ERG',gas_prod_str]]
                                    .query("AAPG_CODE_ERG == '395'")
                                    .assign(proxy_data=lambda df: df[gas_prod_str])
                                    .drop(columns=[gas_prod_str, 'AAPG_CODE_ERG'])
                                    .rename(columns=lambda x: str(x).lower())
                                    .reset_index(drop=True)
                                    )
            basin_395_prod_df = pd.concat([basin_395_prod_df,basin_395_prod_imonth])
            # Basin 430 Gas Well Gas Production
            basin_430_prod_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','AAPG_CODE_ERG',gas_prod_str]]
                                    .query("AAPG_CODE_ERG == '430'")
                                    .assign(proxy_data=lambda df: df[gas_prod_str])
                                    .drop(columns=[gas_prod_str, 'AAPG_CODE_ERG'])
                                    .rename(columns=lambda x: str(x).lower())
                                    .reset_index(drop=True)
                                    )
            basin_430_prod_df = pd.concat([basin_430_prod_df,basin_430_prod_imonth])
            # Other Basins Gas Well Gas Production
            basin_other_prod_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','AAPG_CODE_ERG',gas_prod_str]]
                                      .query("AAPG_CODE_ERG != '220' & AAPG_CODE_ERG != '395' & AAPG_CODE_ERG != '430'")
                                      .assign(proxy_data=lambda df: df[gas_prod_str])
                                      .drop(columns=[gas_prod_str, 'AAPG_CODE_ERG'])
                                      .rename(columns=lambda x: str(x).lower())
                                      .reset_index(drop=True)
                                      )
            basin_other_prod_df = pd.concat([basin_other_prod_df,basin_other_prod_imonth])

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

            # Well Completions
            # Conventional Gas Well Completions
            conv_well_comp_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','WELL_COUNT','HF','comp_year_month']]
                                    .query("HF != 'Y'")
                                    .drop(columns=["HF"])
                                    .rename(columns=lambda x: str(x).lower())
                                    .rename(columns={"well_count":"proxy_data"})
                                    .query(f"comp_year_month == {year_month_str}")
                                    .drop(columns=["comp_year_month"])
                                    .reset_index(drop=True)
                                    )
            conv_well_comp_df = pd.concat([conv_well_comp_df,conv_well_comp_imonth])
            
            # HF Gas Well Completions
            hf_well_comp_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','WELL_COUNT','HF','comp_year_month']]
                                  .query("HF == 'Y'")
                                  .drop(columns=["HF"])
                                  .rename(columns=lambda x: str(x).lower())
                                  .rename(columns={"well_count":"proxy_data"})
                                  .query(f"comp_year_month == '{year_month_str}'")
                                  .drop(columns=["comp_year_month"])
                                  .reset_index(drop=True)
                                  )
            hf_well_comp_df = pd.concat([hf_well_comp_df,hf_well_comp_imonth])

            # Drilled Gas Wells
            drilled_well_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','WELL_COUNT','HF','spud_year','first_prod_year']]
                                  .rename(columns=lambda x: str(x).lower())
                                  .rename(columns={"well_count":"proxy_data"})
                                  # wells with a spud date or first production date in the current year
                                  .query(f"spud_year == '{iyear}' | first_prod_year == '{iyear}'")
                                  # wells with a spud_year == iyear or if no spud date, first_prod_year == iyear
                                  .query(f"spud_year == '{iyear}' | spud_year == 'NaN'")
                                  .drop(columns=['hf', 'spud_year', 'first_prod_year'])
                                  .reset_index(drop=True)
                                  )
            drilled_well_df = pd.concat([drilled_well_df,drilled_well_imonth])
            
            # Offshore Well Counts and Production Volumes in State Waters in the Gulf of Mexico
            state_gom_offshore_states = ['AL','FL','LA','MS','TX']
            # Offshore State GOM Gas Well Counts
            state_gom_offshore_well_count_imonth = (ng_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE','WELL_COUNT']]
                                                   .rename(columns=lambda x: str(x).lower())
                                                   .rename(columns={"well_count":"proxy_data"})
                                                   .reset_index(drop=True)
                                                   )
            state_gom_offshore_well_count_df = pd.concat([state_gom_offshore_well_count_df,state_gom_offshore_well_count_imonth])
            # Offshore State GOM Gas Well Gas Production
            state_gom_offshore_well_prod_imonth = (ng_offshore_data_imonth_temp[['year','year_month','STATE_CODE','LATITUDE','LONGITUDE',gas_prod_str]]
                                                  .query("STATE_CODE.isin(@state_gom_offshore_states)")
                                                  .assign(proxy_data=lambda df: df[gas_prod_str])
                                                  .drop(columns=[gas_prod_str])
                                                  .rename(columns=lambda x: str(x).lower())
                                                  .reset_index(drop=True)
                                                  )
            state_gom_offshore_well_prod_df = pd.concat([state_gom_offshore_well_prod_df,state_gom_offshore_well_prod_imonth])

    # Calculate Relative Emissions
    def calc_enverus_rel_emi(df):
        df['rel_emi'] = df.groupby(["state_code", "year"])['proxy_data'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
        df = df.drop(columns='proxy_data')
        return df

    # Well Counts
    all_well_count_df = calc_enverus_rel_emi(all_well_count_df)
    conv_well_count_df = calc_enverus_rel_emi(conv_well_count_df)
    hf_well_count_df = calc_enverus_rel_emi(hf_well_count_df)
    # Well-Level Production Volumes
    all_well_prod_df = calc_enverus_rel_emi(all_well_prod_df)
    basin_220_prod_df = calc_enverus_rel_emi(basin_220_prod_df)
    basin_395_prod_df = calc_enverus_rel_emi(basin_395_prod_df)
    basin_430_prod_df = calc_enverus_rel_emi(basin_430_prod_df)
    basin_other_prod_df = calc_enverus_rel_emi(basin_other_prod_df)
    # Water Production Volumes
    water_prod_df = calc_enverus_rel_emi(water_prod_df)
    # Well Completions
    conv_well_comp_df = calc_enverus_rel_emi(conv_well_comp_df)
    hf_well_comp_df = calc_enverus_rel_emi(hf_well_comp_df)
    # Drilled Gas Wells
    drilled_well_df = calc_enverus_rel_emi(drilled_well_df)
    # Offshore Well Counts and Production Volumes in State Waters in the Gulf of Mexico
    state_gom_offshore_well_count_df = calc_enverus_rel_emi(state_gom_offshore_well_count_df)
    state_gom_offshore_well_prod_df = calc_enverus_rel_emi(state_gom_offshore_well_prod_df)

    # Format Proxy Data into Geodataframes
    def enverus_df_to_gdf(df):
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
            .loc[:, ["year", "year_month", "state_code", "rel_emi", "geometry"]]
        )
        return gdf

    # Well Counts
    all_well_count_gdf = enverus_df_to_gdf(all_well_count_df)
    conv_well_count_gdf = enverus_df_to_gdf(conv_well_count_df)
    hf_well_count_gdf = enverus_df_to_gdf(hf_well_count_df)
    # Well-Level Production Volumes
    all_well_prod_gdf = enverus_df_to_gdf(all_well_prod_df)
    basin_220_prod_gdf = enverus_df_to_gdf(basin_220_prod_df)
    basin_395_prod_gdf = enverus_df_to_gdf(basin_395_prod_df)
    basin_430_prod_gdf = enverus_df_to_gdf(basin_430_prod_df)
    basin_other_prod_gdf = enverus_df_to_gdf(basin_other_prod_df)
    # Water Production Volumes
    water_prod_gdf = enverus_df_to_gdf(water_prod_df)
    # Well Completions
    conv_well_comp_gdf = enverus_df_to_gdf(conv_well_comp_df)
    hf_well_comp_gdf = enverus_df_to_gdf(hf_well_comp_df)
    # Drilled Gas Wells
    drilled_well_gdf = enverus_df_to_gdf(drilled_well_df)
    # Offshore Well Counts and Production Volumes in State Waters in the Gulf of Mexico
    state_gom_offshore_well_count_gdf = enverus_df_to_gdf(state_gom_offshore_well_count_df)
    state_gom_offshore_well_prod_gdf = enverus_df_to_gdf(state_gom_offshore_well_prod_df)

    # STEP 2.4: Well and Production Data (from NEI)

    # NEI data is used for well counts, gas well completion counts, 
    # gas well drilled counts, and gas production volumes for IL and IN.

    # NEI data is used for water production volumes for IL, IN, KS, OK, and PA 
    # as well as WV for years less than 2016.

    # FIPS codes for relevant states (each code starts with 2 distinct characters):
    # IL: 17; IN: 18; KS: 20; OK: 40; PA: 42; WV: 54
    
    fips_codes_df = pd.DataFrame({'state_code': ['IL', 'IN', 'KS', 'OK', 'PA', 'WV'],
                                  'fips_code': ['17', '18', '20', '40', '42', '54']})

    # Function to get NEI textfile and shapefile data
    def get_NEI_data(ghgi_year, data_year, file_name):
        if data_year <= 2017:
            # NEI textfile data (data_year <= 2017) (2011, 2014, 2016, 2017)
            nei_textfile_name = f"CONUS_SA_FILES_{data_year}/{file_name}"
            nei_textfile_path = os.path.join(nei_path, nei_textfile_name)
            data_temp = pd.read_csv(nei_textfile_path, sep='\t', skiprows = 25)
            data_temp = data_temp.drop(["!"], axis=1)
            data_temp.columns = ['Code','FIPS','COL','ROW','Frac','Abs','FIPS_Total','FIPS_Running_Sum']
            data_temp = data_temp.astype({"FIPS": str})
            # if water production data (gas: 6832, oil: 6833)
            if file_name == 'USA_6832_NOFILL.txt' or file_name == 'USA_6833_NOFILL.txt':
                if data_year < 2016:
                    data_temp = (data_temp
                                # query states: IL, IN, KS, OK, PA, WV
                                .query("FIPS.str.startswith('17') | FIPS.str.startswith('18') | FIPS.str.startswith('20') | FIPS.str.startswith('40') | FIPS.str.startswith('42') | FIPS.str.startswith('54')")
                                .reset_index(drop=True)
                                )
                    colmax = data_temp['COL'].max()
                    colmin = data_temp['COL'].min()
                    rowmax = data_temp['ROW'].max()
                    rowmin = data_temp['ROW'].min()
                else:
                    data_temp = (data_temp
                                # query states: IL, IN, KS, OK, PA
                                .query("FIPS.str.startswith('17') | FIPS.str.startswith('18') | FIPS.str.startswith('20') | FIPS.str.startswith('40') | FIPS.str.startswith('42')")
                                .reset_index(drop=True)
                                )
                    colmax = data_temp['COL'].max()
                    colmin = data_temp['COL'].min()
                    rowmax = data_temp['ROW'].max()
                    rowmin = data_temp['ROW'].min()
            # non-water production proxies (IL, IN)
            else:
                data_temp = (data_temp
                            # query states: IL, IN
                            .query("FIPS.str.startswith('17') | FIPS.str.startswith('18')")
                            .reset_index(drop=True)
                            )
                colmax = data_temp['COL'].max()
                colmin = data_temp['COL'].min()
                rowmax = data_temp['ROW'].max()
                rowmin = data_temp['ROW'].min()
            # NEI reference grid shapefile with lat/lon locations
            nei_reference_grid_path = os.path.join(nei_path, "NEI_Reference_Grid_LCC_to_WGS84_latlon.shp")
            nei_reference_grid = (gpd.read_file(nei_reference_grid_path)
                                .to_crs(4326))
            nei_reference_grid = (nei_reference_grid
                                .assign(cellid_column = nei_reference_grid.cellid.astype(str).str[0:4].astype(int))
                                .assign(cellid_row = nei_reference_grid.cellid.astype(str).str[5:].astype(int))
                                .query(f"cellid_column <= {colmax} & cellid_column >= {colmin}")
                                .query(f"cellid_row <= {rowmax} & cellid_row >= {rowmin}")
                                .reset_index(drop=True)
                                )
            # Match lat/lon locations from reference grid to nei data
            for idx in np.arange(0,len(data_temp)):
                # Add in lat/lon
                icol = data_temp['COL'][idx]
                irow = data_temp['ROW'][idx]
                match = np.where((icol == nei_reference_grid.loc[:,'cellid_column']) & (irow == nei_reference_grid.loc[:,'cellid_row']))[0][0]
                match = int(match)
                # data_temp.loc[idx,'Lat'] = nei_reference_grid.loc[match, 'Latitude']
                # data_temp.loc[idx,'Lon'] = nei_reference_grid.loc[match, 'Longitude']
                data_temp.loc[idx,'geometry'] = nei_reference_grid.loc[match, 'geometry']
                # Add in state_code
                ifips = data_temp.loc[idx,'FIPS'][0:2]
                data_temp.loc[idx,'state_code'] = fips_codes_df.loc[np.where(ifips == fips_codes_df.loc[:, 'fips_code'])[0][0],'state_code']
            data_temp = data_temp[['state_code', 'Abs', 'geometry']]
            data_temp = data_temp.rename(columns={'Abs':'activity_data'})
        
        else:
            # NEI shapefile data (data_year > 2017) (2018, 2019, 2021, 2022)
            state_geometries = state_gdf[["state_code","geometry"]]
            nei_file_name = f"CONUS_SA_FILES_{data_year}"
            nei_file_path = os.path.join(nei_path, nei_file_name)
            data_temp = gpd.read_file(nei_file_path, layer=file_name)
            data_temp = data_temp.to_crs(4326)
            data_temp = gpd.tools.sjoin(data_temp, state_gdf, how="left")

            # water production data (IL, IN, KS, OK, PA)
            if file_name == 'PRODUCED_WATER_GAS' or file_name == '_6832' or file_name == 'ProducedWaterGasWells':
                states_to_query = ['IL', 'IN', 'KS', 'OK', 'PA']
            # non-water production proxies (IL, IN)
            else:
                states_to_query = ['IL', 'IN']
            
            # query relevant states
            data_temp = data_temp.query('state_code.isin(@states_to_query)')

            # grab activity data depending on column name (changes by year)
            if data_year == 2018 or data_year == 2019 or data_year == 2020:
                data_temp = data_temp[['state_code', 'ACTIVITY', 'geometry']]
                data_temp = data_temp.rename(columns={'ACTIVITY':'activity_data'})            
            if data_year == 2021:
                data_temp = data_temp[['state_code', 'GRID_AC', 'geometry']]
                data_temp = data_temp.rename(columns={'GRID_AC':'activity_data'})
            if data_year == 2022:
                data_temp = data_temp[['state_code', 'GRID_ACTIV', 'geometry']]
                data_temp = data_temp.rename(columns={'GRID_ACTIV':'activity_data'})
        
        # convert activity data to relative emissions (idata / sum(state data))
        data_temp['rel_emi'] = data_temp.groupby(["state_code"])['activity_data'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0)
        monthly_data_temp = data_temp.copy()
        monthly_data_temp['rel_emi'] = monthly_data_temp['rel_emi'] * 1/12
        monthly_data_temp = monthly_data_temp.drop(columns='activity_data')

        # convert proxy data to monthly (assume 1/12 of annual proxy is assigned to each month)
        nei_proxy_data = pd.DataFrame()
        for imonth in range(1,13):
            imonth_str = f"{imonth:02}"  # convert to 2-digit months
            data_temp_imonth = monthly_data_temp.copy()
            data_temp_imonth = data_temp_imonth.assign(year_month=str(ghgi_year)+'-'+imonth_str)
            nei_proxy_data = pd.concat([nei_proxy_data,data_temp_imonth])
        nei_proxy_data = nei_proxy_data.assign(year=ghgi_year)
        nei_proxy_data = (nei_proxy_data[['year', 'year_month', 'state_code', 'rel_emi', 'geometry']]
                          .reset_index(drop=True)
                          )
        return nei_proxy_data

    # NEI data year assignments
    # All years use the data affiliated with their year except the following exceptions:
        # 2012: use 2011 data
        # 2013: use 2014 data
        # 2015: use 2014 data
        # 2016: use 2017 data
    nei_data_years = pd.DataFrame({'year': [2012,
                                            2013,
                                            2014,
                                            2015,
                                            2016,
                                            2017,
                                            2018,
                                            2019,
                                            2020,
                                            2021,
                                            2022], 
                                   'nei_data': [2011,
                                                2014,
                                                2014,
                                                2014,
                                                2017,
                                                2017,
                                                2018,
                                                2019,
                                                2020,
                                                2021,
                                                2022]})

    # NEI Data Dataframes:
    # Well Counts
    nei_all_well_count_df = pd.DataFrame()  # Active gas well (conventional + HF) counts in a given month
    nei_conv_well_count_df = pd.DataFrame()  # Active conventional gas well counts in a given month
    nei_hf_well_count_df = pd.DataFrame()  # Active HF gas well counts in a given month
    # Well-Level Production Volumes
    nei_all_well_prod_df = pd.DataFrame()  # Active gas well (conventional + HF) gas production in a given month
    nei_basin_other_prod_df = pd.DataFrame()  # Gas well gas production in Other Basins in a given month
    # Water Production Volumes
    nei_water_prod_df = pd.DataFrame()
    # Well Completions
    nei_conv_well_comp_df = pd.DataFrame()  # Conventional gas well completions
    nei_hf_well_comp_df = pd.DataFrame()  # HF gas well completions
    # Drilled Gas Wells
    nei_drilled_well_df = pd.DataFrame()  # Gas wells drilled

    # NEI text file and shapefile names:
    # Well Counts
    well_count_file_names = pd.DataFrame({
        'data_year': [2011, 2014, 2017,
                      2018, 2019, 2020, 2021, 2022],
        'file_name': ['USA_698_NOFILL.txt', 'USA_698_NOFILL.txt', 'USA_698_NOFILL.txt',
                      'GAS_WELLS', 'GAS_WELLS', 'GAS_WELL', '_698', 'GasWells'],
        })
    # Well-Level Production Volumes
    gas_prod_file_names = pd.DataFrame({
        'data_year': [2011, 2014, 2017,
                      2018, 2019, 2020, 2021, 2022],
        'file_name': ['USA_696_NOFILL.txt', 'USA_696_NOFILL.txt', 'USA_696_NOFILL.txt',
                      'GAS_PRODUCTION', 'GAS_PRODUCTION', 'GAS_PRODUCTION', '_696', 'GasProduction'],
        })
    # Water Production Volumes
    water_prod_file_names = pd.DataFrame({
        'data_year': [2011, 2014, 2017,
                      2018, 2019, 2020, 2021, 2022],
        'file_name': ['USA_6832_NOFILL.txt', 'USA_6832_NOFILL.txt', 'USA_6832698_NOFILL.txt',
                      'PRODUCED_WATER_GAS', 'PRODUCED_WATER_GAS', 'PRODUCED_WATER_GAS', '_6832', 'ProducedWaterGasWells'],
        })
    # Well Completions
    comp_count_file_names = pd.DataFrame({
        'data_year': [2011, 2014, 2017,
                      2018, 2019, 2020, 2021, 2022],
        'file_name': ['USA_678_NOFILL.txt', 'USA_678_NOFILL.txt', 'USA_678_NOFILL.txt',
                      'COMPLETIONS_GAS', 'COMPLETIONS_GAS', 'COMPLETIONS_GAS', '_678', 'GasWellCompletions'],
        })
    # Drilled Gas Wells
    spud_count_file_names = pd.DataFrame({
        'data_year': [2011, 2014, 2017,
                      2018, 2019, 2020, 2021, 2022],
        'file_name': ['USA_671_NOFILL.txt', 'USA_671_NOFILL.txt', 'USA_671_NOFILL.txt',
                      'SPUD_GAS', 'SPUD_GAS', 'SPUD_GAS', '_671', 'SpudCountGasWells'],
        })
    
    
    def get_nei_file_name(nei_data_year, nei_file_names):
        nei_file_name = nei_file_names[nei_file_names['data_year'] == nei_data_year]['file_name'].values[0]
        return nei_file_name


    for iyear in years:
        nei_data_year = nei_data_years[nei_data_years['year'] == iyear]['nei_data'].values[0]
        # Well Count
        ifile_name = get_nei_file_name(nei_data_year, well_count_file_names)
        nei_all_well_count_iyear = get_NEI_data(iyear, nei_data_year, ifile_name)
        nei_all_well_count_df = pd.concat([nei_all_well_count_df, nei_all_well_count_iyear])
        # Gas Production
        ifile_name = get_nei_file_name(nei_data_year, gas_prod_file_names)
        nei_all_well_prod_iyear = get_NEI_data(iyear, nei_data_year, ifile_name)
        nei_all_well_prod_df = pd.concat([nei_all_well_prod_df, nei_all_well_prod_iyear])
        # Water Production
        ifile_name = get_nei_file_name(nei_data_year, water_prod_file_names)
        nei_water_prod_iyear = get_NEI_data(iyear, nei_data_year, ifile_name)
        nei_water_prod_df = pd.concat([nei_water_prod_df, nei_water_prod_iyear])
        # Completions Count
        ifile_name = get_nei_file_name(nei_data_year, comp_count_file_names)
        nei_conv_well_comp_iyear = get_NEI_data(iyear, nei_data_year, ifile_name)
        nei_conv_well_comp_df = pd.concat([nei_conv_well_comp_df, nei_conv_well_comp_iyear])
        # Spud Count
        ifile_name = get_nei_file_name(nei_data_year, spud_count_file_names)
        nei_drilled_well_iyear = get_NEI_data(iyear, nei_data_year, ifile_name)
        nei_drilled_well_df = pd.concat([nei_drilled_well_df, nei_drilled_well_iyear])
    
    # Copy Data to Other Dataframes
    nei_conv_well_count_df = nei_all_well_count_df.copy()
    nei_hf_well_count_df = nei_all_well_count_df.copy()
    nei_basin_other_prod_df = nei_all_well_prod_df.copy()
    nei_hf_well_comp_df = nei_conv_well_comp_df.copy()

    # Add NEI Data to Enverus Data
    # Well Counts
    all_well_count_gdf = pd.concat([all_well_count_gdf, nei_all_well_count_df]).reset_index(drop=True)
    conv_well_count_gdf = pd.concat([conv_well_count_gdf, nei_conv_well_count_df]).reset_index(drop=True)
    hf_well_count_gdf = pd.concat([hf_well_count_gdf, nei_hf_well_count_df]).reset_index(drop=True)
    # Well-Level Production Volumes
    all_well_prod_gdf = pd.concat([all_well_prod_gdf, nei_all_well_prod_df]).reset_index(drop=True)
    basin_220_prod_gdf = basin_220_prod_df.reset_index(drop=True)  # No IL/IN data to add
    basin_395_prod_gdf = basin_395_prod_df.reset_index(drop=True)  # No IL/IN data to add
    basin_430_prod_gdf = basin_430_prod_df.reset_index(drop=True)  # No IL/IN data to add
    basin_other_prod_gdf = pd.concat([basin_other_prod_gdf, nei_basin_other_prod_df]).reset_index(drop=True)
    # Water Production Volumes
    water_prod_gdf = pd.concat([water_prod_gdf, nei_water_prod_df]).reset_index(drop=True)
    # Well Completions
    conv_well_comp_gdf = pd.concat([conv_well_comp_gdf, nei_conv_well_comp_df]).reset_index(drop=True)
    hf_well_comp_gdf = pd.concat([hf_well_comp_gdf, nei_hf_well_comp_df]).reset_index(drop=True)
    # Drilled Gas Wells
    drilled_well_gdf = pd.concat([drilled_well_gdf, nei_drilled_well_df]).reset_index(drop=True)
    # Offshore Well Counts and Production Volumes in State Waters in the Gulf of Mexico
    state_gom_offshore_well_count_gdf = state_gom_offshore_well_count_df.reset_index(drop=True)  # No IL/IN data to add
    state_gom_offshore_well_prod_gdf = state_gom_offshore_well_prod_df.reset_index(drop=True)  # No IL/IN data to add

    # Output Proxy Parquet Files
    all_well_count_gdf.to_parquet(all_well_count_output_path)
    conv_well_count_gdf.to_parquet(conv_well_count_output_path)
    hf_well_count_gdf.to_parquet(hf_well_count_output_path)
    all_well_prod_gdf.to_parquet(all_well_prod_output_path)
    basin_220_prod_gdf.to_parquet(basin_220_prod_output_path)
    basin_395_prod_gdf.to_parquet(basin_395_prod_output_path)
    basin_430_prod_gdf.to_parquet(basin_430_prod_output_path)
    basin_other_prod_gdf.to_parquet(basin_other_prod_output_path)
    water_prod_gdf.to_parquet(water_prod_output_path)
    conv_well_comp_gdf.to_parquet(conv_well_comp_output_path)
    hf_well_comp_gdf.to_parquet(hf_well_comp_output_path)
    drilled_well_gdf.to_parquet(drilled_well_output_path)
    state_gom_offshore_well_count_gdf.to_parquet(state_gom_offshore_well_count_output_path)
    state_gom_offshore_well_prod_gdf.to_parquet(state_gom_offshore_well_prod_output_path)
    return None



    

