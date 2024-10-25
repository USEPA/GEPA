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
    output_path: Annotated[Path, Product] = proxy_data_dir_path / "gb_stations_proxy.parquet",
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
            .assign(WELL_COUNT=1)
            )

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
    # but the prsence of a well will only be included in maps in months where monthly gas prod > 
