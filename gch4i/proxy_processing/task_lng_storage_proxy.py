# %%
from pathlib import Path
from typing import Annotated
import warnings

import geopandas as gpd
import numpy as np
import pandas as pd
from pytask import Product, mark, task

from gch4i.config import (
    global_data_dir_path,
    max_year,
    min_year,
    proxy_data_dir_path,
    sector_data_dir_path,
)
from gch4i.utils import us_state_to_abbrev_dict


@mark.persist
@task(id="lng_storage_proxy")
def get_lng_storage_proxy_data(
    # Inputs
    LNG_storage_input_folder: Path = sector_data_dir_path / 'lng_storage/annual-liquefied-natural-gas-2010-present/',  # individual files like annual_gas_distribution_2010.xlsx
    LNG_storage_Enverus_inputfile: Path = sector_data_dir_path / 'lng_storage/LNG_Terminals_AllUS_WGS84.xls',
    FracTracker_inputfile: Path = sector_data_dir_path / 'lng_storage/FracTracker_PeakShaving_WGS84.xls',
    
    # Outputs
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "lng_storage_proxy.parquet"
    )
):
    ########################################
    # load facility & location data
    ########################################

    # Load PHMSA LNG storage data. contains facility capacities by year. main proxy data source
    phmsa_dfs = []
    phmsa_files = list(LNG_storage_input_folder.glob("annual_liquefied_natural_gas_*.xlsx"))
    for file_path in phmsa_files:
        df = pd.read_excel(file_path, sheet_name='LNG AR Part B', skiprows=2)
        df = df[['REPORT_YEAR', 'FACILITY_NAME', 'PARTA2NAMEOFCOMP', 'TOTAL_CAPACITY_BBLS', 
                   'TOTAL_CAPACITY_MMCF', 'FACILITY_STATE', 'FACILITY_ZIP_CODE', 
                   'FACILITY_STATUS', 'TYPE_OF_FACILITY', 'FUNCTION_OF_FACILITY']]
        df = df[  #filter for storage facilities that are in service 
            df['FUNCTION_OF_FACILITY'].isin(['Storage w/ Liquefaction','Storage w/o Liquefaction','Storage w/ Both']) 
                &
           (df['FACILITY_STATUS'] == 'In Service')
        ]
        phmsa_dfs.append(df)
    phmsa_df = pd.concat(phmsa_dfs, ignore_index=True)
    phmsa_df['Lat'] = np.nan
    phmsa_df['Lon'] = np.nan
    phmsa_df = phmsa_df[phmsa_df['REPORT_YEAR'].isin(range(min_year, max_year + 1))] #filter by report year, and check that all years are present
    assert set(phmsa_df.REPORT_YEAR.unique()) == set(range(min_year, max_year + 1)), f"Years are not present in PHMSA data; {phmsa_df.REPORT_YEAR.unique().tolist()}"

    # Load Enverus LNG storage data. Contains latitude and longitude of facilities
    enverus_df = pd.read_excel(
        LNG_storage_Enverus_inputfile,
        usecols=['NAME', 'OPERATOR', 'TYPE', 'CAP_STO', 'STATE_NAME', 'CNTY_NAME', 'Longitude', 'Latitude']
    )
    enverus_df = enverus_df[~enverus_df['STATE_NAME'].isin(['Alaska', 'Hawaii'])].reset_index(drop=True)
    enverus_df['STATE_NAME'] = enverus_df['STATE_NAME'].map(us_state_to_abbrev_dict)

    # Load FracTracker LNG storage data. contains latitude and longitude of facilities
    fractracker_df = pd.read_excel(
        FracTracker_inputfile,
        usecols=['Company', 'City', 'State', 'Zip', 'Longitude', 'Latitude']
    )
    fractracker_df = fractracker_df[~fractracker_df['State'].isin(['Alaska', 'Hawaii'])].reset_index(drop=True)
    fractracker_df['Zip'] = fractracker_df['Zip'].astype(str).apply(lambda x: x.replace('.', '0').zfill(5)) #clean up zip codes
    fractracker_df['State'] = fractracker_df['State'].map(us_state_to_abbrev_dict)

    ########################################
    # Match facilities to locations
    ########################################

    # Step 1: Match PHMSA facilities with Enverus data
    def clean_facility_name(name):
        return str(name).lower().replace("lng", "").strip()
    
    unique_phmsa_facilities = phmsa_df[['FACILITY_NAME','PARTA2NAMEOFCOMP','FACILITY_STATE', 'FACILITY_ZIP_CODE']].drop_duplicates().reset_index(drop=True)
    for idx in unique_phmsa_facilities.index:
        facility = unique_phmsa_facilities.loc[idx]
        state = facility['FACILITY_STATE']
        
        # Skip if already matched through previous iteration
        mask = ((phmsa_df['FACILITY_NAME'] == facility['FACILITY_NAME']) & 
               (phmsa_df['FACILITY_STATE'] == facility['FACILITY_STATE']))
        if not phmsa_df.loc[mask, 'Lat'].isna().iloc[0]:
            continue
            
        # Skip AK/HI
        if state in ['AK', 'HI']:
            continue
            
        # Get matching state records from Enverus
        state_matches = enverus_df[enverus_df['STATE_NAME'] == state]
        
        if len(state_matches) == 0:
            continue
            
        # Create word sets for matching - include both company and facility name
        phmsa_words = set(clean_facility_name(facility['FACILITY_NAME']).split() + 
                         clean_facility_name(facility['PARTA2NAMEOFCOMP']).split())
        
        # Try to find matches with scoring
        matches = []
        match_scores = []
        for _, enverus_row in state_matches.iterrows():
            enverus_words = set(clean_facility_name(enverus_row['NAME']).split() + 
                              clean_facility_name(enverus_row['OPERATOR']).split() +
                              [enverus_row['CNTY_NAME'].lower()])
            
            # Special case for Chattanooga spelling
            if ('chatanooga' in enverus_words and 'chattanooga' in phmsa_words) or \
               ('chattanooga' in enverus_words and 'chatanooga' in phmsa_words):
                matches.append(enverus_row)
                match_scores.append(1)  # Give it a standard score
            # Word matching with score
            else:
                matching_words = phmsa_words & enverus_words
                if matching_words:
                    matches.append(enverus_row)
                    match_scores.append(len(matching_words))
        
        # Use best match if available
        if matches:
            best_match = matches[match_scores.index(max(match_scores))]
            phmsa_df.loc[mask, 'Lat'] = best_match['Latitude']
            phmsa_df.loc[mask, 'Lon'] = best_match['Longitude']

    # Step 2: Match remaining facilities with FracTracker data
    for idx in unique_phmsa_facilities.index:
        facility = unique_phmsa_facilities.loc[idx]
        
        # Skip if already matched
        mask = ((phmsa_df['FACILITY_NAME'] == facility['FACILITY_NAME']) & 
               (phmsa_df['FACILITY_STATE'] == facility['FACILITY_STATE']))
        if not phmsa_df.loc[mask, 'Lat'].isna().iloc[0]:
            continue
            
        # Try zip code match first
        zip_matches = fractracker_df[fractracker_df['Zip'] == facility['FACILITY_ZIP_CODE']]
        if len(zip_matches) == 1:
            phmsa_df.loc[mask, 'Lat'] = zip_matches.iloc[0]['Latitude']
            phmsa_df.loc[mask, 'Lon'] = zip_matches.iloc[0]['Longitude']
            continue
            
        # Try name matching if zip doesn't work
        if state not in ['AK', 'HI']:
            state_matches = fractracker_df[fractracker_df['State'] == state]
            
            # Create word sets for matching
            phmsa_words = set(clean_facility_name(facility['FACILITY_NAME']).split() + 
                            clean_facility_name(facility['PARTA2NAMEOFCOMP']).split())
            
            matches = []
            match_scores = []
            for _, frac_row in state_matches.iterrows():
                frac_words = set(clean_facility_name(frac_row['Company']).split() + 
                               clean_facility_name(frac_row['City']).split())
                
                matching_words = phmsa_words & frac_words
                if matching_words:
                    matches.append(frac_row)
                    match_scores.append(len(matching_words))
            
            # Use best match if available
            if matches:
                best_match = matches[match_scores.index(max(match_scores))]
                phmsa_df.loc[mask, 'Lat'] = best_match['Latitude']
                phmsa_df.loc[mask, 'Lon'] = best_match['Longitude']


    #Step 3 manual matching
    manual_matches = { # (lat, lon)
        # Direct matches
            ('CT', 'TOTAL PEAKING SERVICES LNG'): (41.230698, -73.064036),  # Direct match in FracTracker: "Total Peaking Services, LLC" in Milford, CT
            ('NY', 'ASTORIA'): (40.752447, -73.950751),  # Direct match in FracTracker: "Consolidated Edison LNG Plant" in Astoria, NY
            ('IN', 'KOKOMO LNG PLANT'): (40.486427, -86.133603),  # Direct match in FracTracker: "Kokomo Gas and Fuel Company" in Kokomo, IN
            ('GA', 'COLUMBUS LNG PLANT'): (32.440098, -84.966781),  # Direct match in FracTracker: "Atmos EnergyCorporation" in Columbus, GA
            ('WI', 'EAU CLAIRE'): (44.785183, -91.524632),  # Direct match in FracTracker: "Xcel Energy" in Eau Claire, WI
            ('IN', 'LAPORTE LNG PLANT'): (41.599937, -86.748083),  # Direct match in FracTracker: "Northern Indiana Public Service Company" in La Porte, IN
            ('NC', 'LNG PLANT'): (35.583923, -77.372864),  # Direct match in FracTracker: "Greenville Utilities Commission" in Greenville, NC
            ('MA', 'ACUSHNET LNG'): (42.228695, -71.522565),  # Direct match in FracTracker: "Hopkinton LNG Corp." in Hopkinton, MA
            ('NC', 'PSNC ENERGY LNG FACILITY'): (35.791324, -78.758735),  # Direct match in FracTracker: "Public Service Company of N.C., Inc." in Cary, NC
            ('MN', 'WESCOTT'): (44.779034, -93.037759),  # Direct match in FracTracker: "Xcel Energy" in Inver Grove Heights, MN
   
        # Indirect matches
            ('PA', 'TEMPLE LNG PLANT'): (40.421699, -75.927026),  # Close to FracTracker "UGI LNG, Inc." in Reading, PA (40.335648, -75.926875)
            ('WI', 'ELM RD LNG PLANT'): (42.873443, -87.865345),  # From Enverus: Wisconsin Electric Power Company in Oak Creek, WI - company name matches
            ('IN', 'LNG SOUTH'): (39.721988, -86.089985),  # From FracTracker: Citizens Gas & Coke Utility in Beech Grove, IN - approximate location based on zip code 46203
            ('IN', 'LNG NORTH'): (39.794121, -86.158235),  # From FracTracker: Citizens Gas & Coke Utility in Indianapolis, IN - approximate location based on zip code 46268
    
        # Facilities found on google maps
            ('PA', 'Bethlehem LNG Facility, UGI ENERGY SERVICES'): (40.608020, -75.305818),
            ('ND', 'Tioga Plant, NORTH DAKOTA LNG LLC'): (48.587406, -102.855774),
            ('UT', 'Magna LNG Plant, DOMINION ENERGY UTAH/WYOMING/IDAHO'): (40.189617, -111.631270),
            ('WA', 'Tacoma LNG, PUGET SOUND ENERGY'): (47.254585, -122.431859),
            ('WA', 'GIG HARBOR SATELLITE, PUGET SOUND ENERGY'): (47.221976, -122.478402),
            ('WA', 'LNG Mobile System, PUGET SOUND ENERGY'): (47.221976, -122.478402),
            ('PA', 'Steelton LNG, UGI ENERGY SERVICES'): (40.232148, -76.751703),

        # Facility NOT FOUND. Defaulting to approximate centroid of given zipcode
            ('ND', 'Fargo Mobile/Temporary, NORTHERN STATES POWER CO OF MINNESOTA'): (46.798616, -96.837554),
    
        # Facility NOT FOUND. Defaulting to industrial area on the coast of in Acusnet, MA
            ('MA', 'ACUSNET LNG, HOPKINTON LNG CO'): (41.667362, -70.914618)
    }
    for idx in unique_phmsa_facilities.index: #apply manual matches to dataframe
        facility = unique_phmsa_facilities.loc[idx]
        state = facility['FACILITY_STATE']
        facility_name = facility['FACILITY_NAME']
        company_name = facility['PARTA2NAMEOFCOMP']
        
        # Skip if already matched
        mask = ((phmsa_df['FACILITY_NAME'] == facility_name) & 
               (phmsa_df['FACILITY_STATE'] == state))
        if not phmsa_df.loc[mask, 'Lat'].isna().iloc[0]:
            continue
            
        # Try all possible matching formats
        match_found = False
        for key_format in [
            (state, facility_name),  # exact match
            (state, facility_name.upper()),  # uppercase match
            (state, f"{facility_name}, {company_name}"),  # with company name
            (state, f"{facility_name.upper()}, {company_name.upper()}")  # uppercase with company name
        ]:
            if key_format in manual_matches:
                lat, lon = manual_matches[key_format]
                phmsa_df.loc[mask, 'Lat'] = lat
                phmsa_df.loc[mask, 'Lon'] = lon
                match_found = True
                break

    # QA/QC Check. print total number of unique PHMSA facilities, and number of unique facilities latlon data.
    unique_phmsa_facilities = phmsa_df[['FACILITY_NAME','PARTA2NAMEOFCOMP','FACILITY_STATE', 'FACILITY_ZIP_CODE', 'Lat','Lon']
                                    ].sort_values(['FACILITY_STATE', 'FACILITY_STATE', 'Lat', 'Lon'], ascending=True
                                    ).drop_duplicates(['FACILITY_NAME','FACILITY_STATE'], keep='first').reset_index(drop=True)
    unmatched_facilities = unique_phmsa_facilities[unique_phmsa_facilities['Lat'].isna()]
    # print(f"Total number of unique PHMSA facilities: {len(unique_phmsa_facilities)}")
    # print(f"Number of unmatched unique facilities: {len(unmatched_facilities)}")
    if len(unmatched_facilities) > 0:
        warnings.warn(f"[task: lng_storage_proxy] There are {len(unmatched_facilities)} LNG storage facilities without identified locations in the data. These facilities will be dropped.")


    ########################################
    # Create proxy df
    ########################################

    # Create proxy gdf with columns [state_code, year, rel_emi, geometry, terminal_name]
    gdf_facilities = gpd.GeoDataFrame(
        phmsa_df,
        geometry=gpd.points_from_xy(phmsa_df.Lon, phmsa_df.Lat),
        crs="EPSG:4326"
    ).rename(columns={
        'FACILITY_STATE': 'state_code',
        'REPORT_YEAR': 'year',
        'FACILITY_NAME': 'facility_name',
        'PARTA2NAMEOFCOMP': 'company_name',
        'TOTAL_CAPACITY_BBLS': 'rel_emi'
    })

    # Aggregate volumes by terminal-year
    proxy_gdf = gpd.GeoDataFrame(gdf_facilities.dropna(subset=['geometry']).groupby(['facility_name', 'company_name', 'year', 'state_code', 'geometry'])['rel_emi'].sum().reset_index())

    # Normalize relative emissions to sum to 1 for each year and state
    proxy_gdf = proxy_gdf.groupby(['state_code', 'year']).filter(lambda x: x['rel_emi'].sum() > 0) #drop state-years with 0 total volume
    proxy_gdf['rel_emi'] = proxy_gdf.groupby(['year', 'state_code'])['rel_emi'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0) #normalize to sum to 1
    sums = proxy_gdf.groupby(["state_code", "year"])["rel_emi"].sum() #get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}" # assert that the sums are close to 1

    # Sort and select final columns
    proxy_gdf = proxy_gdf.sort_values(by=['year', 'state_code','company_name', 'facility_name']).reset_index(drop=True)[
        ['year', 'state_code', 'rel_emi', 'geometry', 'facility_name','company_name']
    ]

    # Save to parquet
    proxy_gdf.to_parquet(output_path)

    return #proxy_gdf, unique_phmsa_facilities

# df, unique_phmsa_facilities = get_lng_storage_proxy_data()
# df

