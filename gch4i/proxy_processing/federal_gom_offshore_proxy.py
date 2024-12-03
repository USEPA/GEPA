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
    max_year,
    min_year,
)

from gch4i.utils import us_state_to_abbrev

# %%
@mark.persist
@task(id="federal_gom_offshore_proxy")
def task_get_federal_gom_offshore_proxy_data(
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    boem_data_directory_path: Path = sector_data_dir_path / "boem",
    ng_output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "federal_gom_offshore_proxy.parquet",
    oil_output_path: Annotated[Path, Product] = proxy_data_dir_path
    / "oil_gom_fed_proxy.parquet",
):
    """
    # TODO:
    """

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

    # get and format boem gom data for 2011, 2014, 2017, and 2021
    # NOTE: 2011 has tblPointER and tblPointEM but the rest of the years have one single table of data
    gom_df = pd.DataFrame()

    # 2011 GOADS Data

    # Read In and Format 2011 BEOM Data
    gom_file_name = f"2011_Gulfwide_Platform_Inventory.accdb"
    gom_file_path = os.path.join(boem_data_directory_path, gom_file_name)
    driver_str = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+gom_file_path+';'''
    conn = pyodbc.connect(driver_str)
    GOADS_locations = pd.read_sql("SELECT * FROM tblPointER", conn)
    GOADS_emissions = pd.read_sql("SELECT * FROM tblPointEM", conn)
    conn.close()

    # Format Location Data
    GOADS_locations = GOADS_locations[["strStateFacilityIdentifier","strEmissionReleasePointID","dblXCoordinate","dblYCoordinate"]]
    #Create platform-by-platform file
    GOADS_locations_Unique = pd.DataFrame({'strStateFacilityIdentifier':GOADS_locations['strStateFacilityIdentifier'].unique()})
    GOADS_locations_Unique['lon'] = 0.0
    GOADS_locations_Unique['lat'] = 0.0
    GOADS_locations_Unique['strEmissionReleasePointID'] = ''

    for iplatform in np.arange(len(GOADS_locations_Unique)):
        match_platform = np.where(GOADS_locations['strStateFacilityIdentifier'] == GOADS_locations_Unique['strStateFacilityIdentifier'][iplatform])[0][0]
        GOADS_locations_Unique.loc[iplatform,'lon',] = GOADS_locations['dblXCoordinate'][match_platform]
        GOADS_locations_Unique.loc[iplatform,'lat',] = GOADS_locations['dblYCoordinate'][match_platform]
        GOADS_locations_Unique.loc[iplatform,'strEmissionReleasePointID'] = GOADS_locations['strEmissionReleasePointID'][match_platform][:3]

    GOADS_locations_Unique.reset_index(inplace=True, drop=True)
    #display(GOADS_locations_Unique)

    #print(GOADS_emissions.columns)
    #Format Emissions Data (clean lease data string)
    GOADS_emissions = GOADS_emissions[["strStateFacilityIdentifier","strPollutantCode","dblEmissionNumericValue","BOEM-MONTH",
                                  "BOEM-LEASE_NUM","BOEM-COMPLEX_ID"]]
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('OCS','')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('-','')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace(' ','')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G1477','G01477')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G73','00073')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G605','00605')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G72','00072')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G599','00599')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G7155','G07155')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G2357','G02357')
    GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G4921','G04921')
    GOADS_emissions['Emis_tg'] = 0.0
    GOADS_emissions['Emis_tg'] = 9.0718474E-7 * GOADS_emissions['dblEmissionNumericValue'] #convert short tons to Tg
    GOADS_emissions = GOADS_emissions[GOADS_emissions['strPollutantCode'] == 'CH4']
    GOADS_emissions.reset_index(inplace=True, drop=True)

    #display(GOADS_emissions)

    # Use ERG Preprocessed data to determine if major or minor and oil or gas
    ERG_complex_crosswalk = pd.read_excel(ERG_GOADSEmissions_inputfile, sheet_name = "Complex Emissions by Source", usecols = "AJ:AM", nrows = 11143)
    #display(ERG_complex_crosswalk)

    # add data to map array, for the closest year to 2011
    year_diff = [abs(x - 2011) for x in year_range]
    iyear = year_diff.index(min(year_diff))

    #assign oil vs gas by lease/complex ID
    GOADS_emissions['LEASE_TYPE'] =''
    GOADS_emissions['MAJOR_STRUC'] =''
    for istruc in np.arange(0,len(GOADS_emissions)):
        imatch = np.where(np.logical_and(ERG_complex_crosswalk['BOEM COMPLEX ID.2']==int(GOADS_emissions['BOEM-COMPLEX_ID'][istruc]),\
                            ERG_complex_crosswalk['Year.2'] == 2011))
        if np.size(imatch) >0:
            imatch = imatch[0][0]
            GOADS_emissions.loc[istruc,'LEASE_TYPE'] = ERG_complex_crosswalk['Oil Gas Defn FINAL.1'][imatch]
            GOADS_emissions.loc[istruc,'MAJOR_STRUC'] = ERG_complex_crosswalk['Major / Minor.1'][imatch]
        else:
            print(istruc, GOADS_emissions['BOEM-COMPLEX_ID'][istruc])

        # for all gas platforms, match the platform to the emissions
        if GOADS_emissions['LEASE_TYPE'][istruc] =='Gas':
            match_platform = np.where(GOADS_locations_Unique.strStateFacilityIdentifier==GOADS_emissions['strStateFacilityIdentifier'][istruc])[0][0]
            ilat = int((GOADS_locations_Unique['lat'][match_platform] - Lat_low)/Res01)
            ilon = int((GOADS_locations_Unique['lon'][match_platform] - Lon_left)/Res01)
            imonth = GOADS_emissions['BOEM-MONTH'][istruc]-1 #dict is 1-12, not 0-11
            if GOADS_emissions['MAJOR_STRUC'][istruc] =='Major':
                Map_GOADSmajor_emissions[ilat,ilon,iyear,imonth] += GOADS_emissions['Emis_tg'][istruc]
            else:
                Map_GOADSminor_emissions[ilat,ilon,iyear,imonth] += GOADS_emissions['Emis_tg'][istruc]
            
            
    # sum complexes and emissions for diagnostic
    majcplx = GOADS_emissions[(GOADS_emissions['MAJOR_STRUC']=='Major')]
    majcplx = majcplx[majcplx['LEASE_TYPE'] =='Gas']
    num_majcplx = majcplx['BOEM-COMPLEX_ID'].unique()
    #print(np.shape(num_majcplx))
    mincplx = GOADS_emissions[GOADS_emissions['MAJOR_STRUC']=='Minor']
    mincplx = mincplx[mincplx['LEASE_TYPE'] =='Gas']
    num_mincplx = mincplx['BOEM-COMPLEX_ID'].unique()
    #print(np.size(num_mincplx))            
    del GOADS_emissions
    print('Number of Major Gas Complexes: ',(np.size(num_majcplx)))
    print('Emissions (Tg): ',np.sum(Map_GOADSmajor_emissions[:,:,iyear,:]))
    print('Number of Minor Gas Complexes: ',(np.size(num_mincplx)))
    print('Emissions (Tg): ',np.sum(Map_GOADSminor_emissions[:,:,iyear,:]))




    gom_data_years = ['2011', '2014', '2017', '2021']
    for idatayear in gom_data_years:
        gom_file_name = f"{idatayear}_Gulfwide_Platform_Inventory.accdb"
        gom_file_path = os.path.join(boem_data_directory_path, gom_file_name)
        driver_str = r'Driver={Microsoft Access Driver (*.mdb, *.accdb)};DBQ='+gom_file_path+';'''
        conn = pyodbc.connect(driver_str)
        GOADS_locations = pd.read_sql("SELECT * FROM tblPointER", conn)
        GOADS_emissions = pd.read_sql("SELECT * FROM tblPointEM", conn)
        conn.close()
                                    
        # Format Location Data
        GOADS_locations = GOADS_locations[["strStateFacilityIdentifier","strEmissionReleasePointID","dblXCoordinate","dblYCoordinate"]]
        #Create platform-by-platform file
        GOADS_locations_Unique = pd.DataFrame({'strStateFacilityIdentifier':GOADS_locations['strStateFacilityIdentifier'].unique()})
        GOADS_locations_Unique['lon'] = 0.0
        GOADS_locations_Unique['lat'] = 0.0
        GOADS_locations_Unique['strEmissionReleasePointID'] = ''

        for iplatform in np.arange(len(GOADS_locations_Unique)):
            match_platform = np.where(GOADS_locations['strStateFacilityIdentifier'] == GOADS_locations_Unique['strStateFacilityIdentifier'][iplatform])[0][0]
            GOADS_locations_Unique.loc[iplatform,'lon',] = GOADS_locations['dblXCoordinate'][match_platform]
            GOADS_locations_Unique.loc[iplatform,'lat',] = GOADS_locations['dblYCoordinate'][match_platform]
            GOADS_locations_Unique.loc[iplatform,'strEmissionReleasePointID'] = GOADS_locations['strEmissionReleasePointID'][match_platform][:3]

        GOADS_locations_Unique.reset_index(inplace=True, drop=True)
        #display(GOADS_locations_Unique)

        #print(GOADS_emissions.columns)
        #Format Emissions Data (clean lease data string)
        GOADS_emissions = GOADS_emissions[["strStateFacilityIdentifier","strPollutantCode","dblEmissionNumericValue","BOEM-MONTH",
                                    "BOEM-LEASE_NUM","BOEM-COMPLEX_ID"]]
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('OCS','')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('-','')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace(' ','')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G1477','G01477')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G73','00073')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G605','00605')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G72','00072')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G599','00599')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G7155','G07155')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G2357','G02357')
        GOADS_emissions['BOEM-LEASE_NUM'] = GOADS_emissions['BOEM-LEASE_NUM'].str.replace('G4921','G04921')
        GOADS_emissions['Emis_tg'] = 0.0
        GOADS_emissions['Emis_tg'] = 9.0718474E-7 * GOADS_emissions['dblEmissionNumericValue'] #convert short tons to Tg
        GOADS_emissions = GOADS_emissions[GOADS_emissions['strPollutantCode'] == 'CH4']
        GOADS_emissions.reset_index(inplace=True, drop=True)

        #display(GOADS_emissions)

        # Use ERG Preprocessed data to determine if major or minor and oil or gas
        ERG_complex_crosswalk = pd.read_excel(ERG_GOADSEmissions_inputfile, sheet_name = "Complex Emissions by Source", usecols = "AJ:AM", nrows = 11143)

        # add data to map array, for the closest year to 2011
        year_diff = [abs(x - 2011) for x in year_range]
        iyear = year_diff.index(min(year_diff))

        #assign oil vs gas by lease/complex ID
        GOADS_emissions['LEASE_TYPE'] =''
        GOADS_emissions['MAJOR_STRUC'] =''
        for istruc in np.arange(0,len(GOADS_emissions)):
            imatch = np.where(np.logical_and(ERG_complex_crosswalk['BOEM COMPLEX ID.2']==int(GOADS_emissions['BOEM-COMPLEX_ID'][istruc]),\
                                ERG_complex_crosswalk['Year.2'] == 2011))
            if np.size(imatch) >0:
                imatch = imatch[0][0]
                GOADS_emissions.loc[istruc,'LEASE_TYPE'] = ERG_complex_crosswalk['Oil Gas Defn FINAL.1'][imatch]
                GOADS_emissions.loc[istruc,'MAJOR_STRUC'] = ERG_complex_crosswalk['Major / Minor.1'][imatch]
            else:
                print(istruc, GOADS_emissions['BOEM-COMPLEX_ID'][istruc])

            # for all gas platforms, match the platform to the emissions
            if GOADS_emissions['LEASE_TYPE'][istruc] =='Gas':
                match_platform = np.where(GOADS_locations_Unique.strStateFacilityIdentifier==GOADS_emissions['strStateFacilityIdentifier'][istruc])[0][0]
                ilat = int((GOADS_locations_Unique['lat'][match_platform] - Lat_low)/Res01)
                ilon = int((GOADS_locations_Unique['lon'][match_platform] - Lon_left)/Res01)
                imonth = GOADS_emissions['BOEM-MONTH'][istruc]-1 #dict is 1-12, not 0-11
                if GOADS_emissions['MAJOR_STRUC'][istruc] =='Major':
                    Map_GOADSmajor_emissions[ilat,ilon,iyear,imonth] += GOADS_emissions['Emis_tg'][istruc]
                else:
                    Map_GOADSminor_emissions[ilat,ilon,iyear,imonth] += GOADS_emissions['Emis_tg'][istruc]
                
                
        # sum complexes and emissions for diagnostic
        majcplx = GOADS_emissions[(GOADS_emissions['MAJOR_STRUC']=='Major')]
        majcplx = majcplx[majcplx['LEASE_TYPE'] =='Gas']
        num_majcplx = majcplx['BOEM-COMPLEX_ID'].unique()
        #print(np.shape(num_majcplx))
        mincplx = GOADS_emissions[GOADS_emissions['MAJOR_STRUC']=='Minor']
        mincplx = mincplx[mincplx['LEASE_TYPE'] =='Gas']
        num_mincplx = mincplx['BOEM-COMPLEX_ID'].unique()
        #print(np.size(num_mincplx))            
        del GOADS_emissions
        print('Number of Major Gas Complexes: ',(np.size(num_majcplx)))
        print('Emissions (Tg): ',np.sum(Map_GOADSmajor_emissions[:,:,iyear,:]))
        print('Number of Minor Gas Complexes: ',(np.size(num_mincplx)))
        print('Emissions (Tg): ',np.sum(Map_GOADSminor_emissions[:,:,iyear,:]))

    
    # Create proxy gdf
    proxy_gdf = (
    gpd.GeoDataFrame(
        gb_stations_df,
        geometry=gpd.points_from_xy(
            gb_stations_df["lon"],
            gb_stations_df["lat"],
            crs=4326,
        ),
    )
    .drop(columns=["lat", "lon"])
    .loc[:, ["facility_name", "state_code", "geometry"]]
    )

    proxy_gdf.to_parquet(output_path)
    return None
