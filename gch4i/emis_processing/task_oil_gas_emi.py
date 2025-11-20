"""
Name:                  Oil & Gas Emissions
Date Last Modified:    2025-11-04
Authors Name:          Andrew Burnette(RTI International)
Purpose:               Clean and standardized carbides emissions data
Input Files:           - gch4i_data_guide_v4.xlsx
                       - {V4_DATA_PATH}/ghgi/oil_and_gas_v4_emissions_crosswalk.xlsx
Output Files:          - {emi_data_dir_path}/{oil_and_gas_emi_outputs}
Notes:                 - Three levels of GHGI data available:
                            - National, Basin, and State
"""

# %% Import Libraries
from pathlib import Path

import pandas as pd

from gch4i.config import (
    v3_emi_data_dir_path,
    v4_emi_data_dir_path,
    v4_ghgi_data_dir_path,
    max_year,
    min_year
)
from gch4i.utils import tg_to_kt

crosswalk_path = Path(v4_ghgi_data_dir_path) / "oil_and_gas_v4_emissions_crosswalk.xlsx"
year_list = [str(x) for x in list(range(min_year, max_year + 1))]



# %% Read in Crosswalk_df tabs

# Read the crosswalk data from Excel
crosswalk_df = pd.read_excel(crosswalk_path, sheet_name="crosswalk")
crosswalk_df = crosswalk_df.dropna(how='all')

# Read national, basin, and state level emissions data
# National
gas_national_ghgi = (
    pd.read_excel(crosswalk_path, sheet_name="gas_national_ghgi", skiprows=11)
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"segment/source": "segment"})
    .filter(items=['segment', 'emi_id'] + year_list, axis=1)
)
oil_national_ghgi = (
    pd.read_excel(crosswalk_path, sheet_name="oil_national_ghgi", skiprows=11)
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"segment/source": "segment"})
    .filter(items=['segment', 'emi_id'] + year_list, axis=1)
)
# Basin
gas_basin_ghgi = (
    pd.read_excel(crosswalk_path, sheet_name="gas_basin_ghgi", skiprows=1)
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"basin name": "basin_name", "basin id": "basin_id"})
    .filter(items=['segment', 'source', 'emi_id', 'basin_name', 'basin_id'] + year_list, axis=1)
)
oil_basin_ghgi = (
    pd.read_excel(crosswalk_path, sheet_name="oil_basin_ghgi", skiprows=1)
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"basin name": "basin_name", "basin id": "basin_id"})
    .filter(items=['segment', 'source', 'emi_id', 'basin_name', 'basin_id'] + year_list, axis=1)
)
# State
gas_state_ghgi = (
    pd.read_excel(crosswalk_path, sheet_name="gas_state_ghgi")
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"georef": "state_code"})
    .query('ghg == "CH4"')
    .filter(items=['state_code', 'subcategory1', 'emi_id'] + year_list, axis=1)
)
oil_state_ghgi = (
    pd.read_excel(crosswalk_path, sheet_name="oil_state_ghgi")
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"georef": "state_code", "ghg category": "ghg_category"})
    .query('ghg_category == "CH4"')
    .filter(items=['state_code', 'subcategory1', 'emi_id'] + year_list, axis=1)
)
# Surrogates
# Oil State Surrogate
oil_state_surrogate = (
    pd.read_excel(crosswalk_path, sheet_name="oil_state_surrogate")
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"state code": "state_code"})
    .filter(items=['surrogate', 'state_code'] + year_list, axis=1)
)
# Gas State Surrogate
gas_state_surrogate = (
    pd.read_excel(crosswalk_path, sheet_name="gas_state_surrogate")
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"state code": "state_code"})
    .filter(items=['surrogate', 'state_code'] + year_list, axis=1)
)
# Basin Surrogate
basin_surrogate = (
    pd.read_excel(crosswalk_path, sheet_name="basin_surrogate")
    .rename(columns=lambda x: str(x).lower())
    .rename(columns={"aapg_basin id": "aapg_basin_id", "state": "state_code"})
    .filter(items=['surrogate', 'aapg_basin_id', 'state_code'] + year_list, axis=1)
)



# %%
"""
Crosswalk Logic

If (ghgi_data_level == "national") AND (surrogate_tab == "blank"):
    emi_id: national --> national
elif (ghgi_data_level == "state"):
    emi_id: state --> state
elif (ghgi_data_level == "blank") AND (surrogate_tab == "blank"):
    pass
    emi_id: use v3_data
elif (ghgi_data_level == "national") AND (surrogate_tab == "gas/oil_state_surrogate"):
    emi_id: national_ghgi * state_prop
elif (ghgi_data_level == "basin") AND surrogate_tab == "basin_surrogate"):
    emi_id: state_ghgi * basin_prop
else:
    error

5 paths w/in function
    - Direct National Calculation
    - Direct State Calculation
    - Retrieve v3 Data Emi output
    - National to State by proportion Calculation
    - Basin to State by proportion Calculation
"""

def direct_national_emi(emi_id, ghgi_tab):
    """
    Direct National Emissions Calculation
    """
    df = ghgi_tab.copy()
    df = df.query('emi_id == @emi_id')
    df = df.drop(columns=['segment'])
    df = df.set_index("emi_id")
    # Replace NA values with 0
    df = df.replace(0, pd.NA)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    df = df.fillna(0)
    df = df.reset_index()
    # Sum emissions for each year
    year_emissions = df[year_list].sum().reset_index()
    year_emissions.columns = ['year', 'ghgi_ch4_kt']
    # Convert year to int
    year_emissions['year'] = year_emissions['year'].astype(int)
    return year_emissions[['year', 'ghgi_ch4_kt']]

#test = direct_national_emi('trans_offshore_emi', oil_national_ghgi)

def direct_state_emi(emi_id, ghgi_tab):
    """
    Direct State Emissions Calculation
    """
    df = ghgi_tab.copy()
    state_list = df["state_code"].unique().tolist()
    state_list = [
        state
        for state in state_list
        if state not in ["AS", "GU", "MP", "PR", "VI", "AK", "HI", "National"]
    ]

    df = df.query('emi_id == @emi_id')
    df = df.drop(columns=['subcategory1', 'emi_id'])
    df = df.query('state_code in @state_list')
    df = df.set_index("state_code")
    # Replace NA values with 0
    df = df.replace(0, pd.NA)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(how="all")
    df = df.fillna(0)
    df = df.reset_index()
    # Melt to long format
    df = df.melt(id_vars=['state_code'], var_name='year', value_name='ch4_tg')
    # Convert units to kt
    df = df.assign(ghgi_ch4_kt=lambda x: x['ch4_tg'] * tg_to_kt)
    df = df.drop(columns=['ch4_tg'])
    df = df.astype({"year": int, "ghgi_ch4_kt": float})
    df = df.fillna({"ghgi_ch4_kt": 0})
    # Convert year to int
    df['year'] = df['year'].astype(int)
    # Filter years
    df = df.query('year.between(@min_year, @max_year)')
    df = df.sort_values(by=['state_code', 'year']).reset_index(drop=True)
    # ADD if statement here for federal_gom_offshore_emi: don't report state_code
    if emi_id == 'federal_gom_offshore_emi':
        df = df[['year', 'ghgi_ch4_kt']]
    else:
        df = df[['state_code', 'year', 'ghgi_ch4_kt']]
    return df

#test = direct_state_emi('federal_gom_offshore_emi', gas_state_ghgi)

def retrieve_v3_emi(emi_id):
    """
    Retrieve v3 Emissions Data
    """
    df = pd.read_csv(
        Path(v3_emi_data_dir_path) / f"{emi_id}.csv",
        dtype={"state_code": str, "year": int, "ghgi_ch4_kt": float},
    )
    return df[['state_code', 'year', 'ghgi_ch4_kt']]

#test = retrieve_v3_emi('well_blowout_emi')

def national_to_state_emi(emi_id, ghgi_tab, surrogate_tab, surrogate_name):
    """
    National to State Emissions Calculation by Proportion
    """
    # Get National Emissions
    national_df = direct_national_emi(emi_id, ghgi_tab)

    # Get State Surrogate (already proportions)
    surrogate_df = surrogate_tab.copy()
    surrogate_df = surrogate_df.query('surrogate == @surrogate_name')
    surrogate_df = surrogate_df.drop(columns=['surrogate'])
    surrogate_df = surrogate_df.set_index('state_code')
    surrogate_df = surrogate_df.replace(0, pd.NA)
    surrogate_df = surrogate_df.apply(pd.to_numeric, errors="coerce")
    surrogate_df = surrogate_df.dropna(how="all")
    surrogate_df = surrogate_df.fillna(0)
    surrogate_df = surrogate_df.reset_index()
    surrogate_df = surrogate_df.melt(id_vars=['state_code'], var_name='year', value_name='proportion')
    surrogate_df['year'] = surrogate_df['year'].astype(int)

    # Merge with national emissions
    merged_df = pd.merge(surrogate_df, national_df, on='year', how='left')
    merged_df['ghgi_ch4_kt'] = merged_df['proportion'] * merged_df['ghgi_ch4_kt']

    final_df = merged_df[['state_code', 'year', 'ghgi_ch4_kt']]
    #final_df = final_df.fillna({'ghgi_ch4_kt': 0})
    final_df = final_df.sort_values(by=['state_code', 'year']).reset_index(drop=True)
    return final_df

#test = national_to_state_emi('trans_onshore_emi', oil_national_ghgi, oil_state_surrogate, 'Petroleum_Transport_Onshore')

# emi_id = 'allwell_emi'
# ghgi_tab = gas_basin_ghgi
# surrogate_tab = basin_surrogate
# surrogate_name = 'NG_Well_Counts'


def basin_to_state_emi(emi_id, ghgi_tab, surrogate_tab, surrogate_name):
    """
    Basin to State Emissions Calculation by Proportion
    """
    # Get Basin Emissions
    basin_df = ghgi_tab.copy()
    basin_df = basin_df.query('emi_id == @emi_id')
    basin_df = basin_df.drop(columns=['segment', 'source', 'emi_id'])
    basin_df = basin_df.melt(id_vars=['basin_name', 'basin_id'], var_name='year', value_name='ghgi_ch4_kt')
    basin_df['year'] = basin_df['year'].astype(int)


    # Get Basin Surrogate (already proportions)
    surrogate_df = surrogate_tab.copy()
    surrogate_df = surrogate_df.query('surrogate == @surrogate_name')
    surrogate_df = surrogate_df.drop(columns=['surrogate'])
    surrogate_df = surrogate_df.set_index(['aapg_basin_id', 'state_code'])
    surrogate_df = surrogate_df.replace(0, pd.NA)
    surrogate_df = surrogate_df.apply(pd.to_numeric, errors="coerce")
    surrogate_df = surrogate_df.dropna(how="all")
    surrogate_df = surrogate_df.fillna(0)
    surrogate_df = surrogate_df.reset_index()
    surrogate_df = surrogate_df.melt(id_vars=['aapg_basin_id', 'state_code'], var_name='year', value_name='proportion')
    surrogate_df['year'] = surrogate_df['year'].astype(int)

    # Merge with basin emissions
    merged_df = pd.merge(surrogate_df, basin_df, left_on=['aapg_basin_id', 'year'], right_on=['basin_id', 'year'], how='left')
    merged_df['ghgi_ch4_kt'] = merged_df['proportion'] * merged_df['ghgi_ch4_kt']

    final_df = merged_df[['state_code', 'year', 'ghgi_ch4_kt']]
    # Group by state_code and year, then sum ghgi_ch4_kt
    final_df = final_df.groupby(['state_code', 'year'])['ghgi_ch4_kt'].sum().reset_index()
    final_df = final_df.sort_values(by=['state_code', 'year']).reset_index(drop=True)
    return final_df

#test = basin_to_state_emi('allwell_emi', gas_basin_ghgi, basin_surrogate, 'NG_Well_Counts')

# %% Function call

def run_oil_gas_emissions_processing():
    """
    Main function to run all oil and gas emissions processing.
    This ensures all functions are available in the correct scope.
    """
    
    def process_crosswalk_emissions():
        """
        Process all emissions IDs from crosswalk_df using the specified logic paths.
        Creates a unique DataFrame for each emi_id based on the crosswalk logic.
        
        Returns:
            dict: Dictionary with emi_id as keys and processed emission DataFrames as values
        """
        results = {}
        
        # Get unique emi_ids from crosswalk_df
        unique_emi_ids = crosswalk_df['emi_id'].unique()
        
        for emi_id in unique_emi_ids:
            # Get the row for this emi_id
            row = crosswalk_df[crosswalk_df['emi_id'] == emi_id].iloc[0]
            
            # Get values and handle NaN/missing values properly
            ghgi_data_level = row.get('ghgi_data_level')
            surrogate_tab = row.get('surrogate_tab')
            surrogate_name = row.get('surrogate_name')
            
            # Convert to lowercase strings if not NaN, otherwise keep as NaN
            if pd.notna(ghgi_data_level):
                ghgi_data_level = str(ghgi_data_level).strip().lower()
            if pd.notna(surrogate_tab):
                surrogate_tab = str(surrogate_tab).strip().lower()
            if pd.notna(surrogate_name):
                surrogate_name = str(surrogate_name).strip()
            
            try:
                # Apply the crosswalk logic
                if ghgi_data_level == "national" and pd.isna(surrogate_tab):
                    # Direct National Calculation - check which DataFrame contains this emi_id
                    if emi_id in oil_national_ghgi['emi_id'].values:
                        df = direct_national_emi(emi_id, oil_national_ghgi)
                    elif emi_id in gas_national_ghgi['emi_id'].values:
                        df = direct_national_emi(emi_id, gas_national_ghgi)
                    else:
                        print(f"Warning: emi_id '{emi_id}' not found in national GHGI data")
                        continue
                        
                elif ghgi_data_level == "state":
                    # Direct State Calculation - check which DataFrame contains this emi_id
                    if emi_id in oil_state_ghgi['emi_id'].values:
                        df = direct_state_emi(emi_id, oil_state_ghgi)
                    elif emi_id in gas_state_ghgi['emi_id'].values:
                        df = direct_state_emi(emi_id, gas_state_ghgi)
                    else:
                        print(f"Warning: emi_id '{emi_id}' not found in state GHGI data")
                        continue
                        
                elif pd.isna(ghgi_data_level) and pd.isna(surrogate_tab):
                    # Retrieve v3 Data
                    df = retrieve_v3_emi(emi_id)
                    
                elif ghgi_data_level == "national" and pd.notna(surrogate_tab) and "state_surrogate" in surrogate_tab:
                    # National to State by proportion Calculation - check which DataFrames to use
                    if emi_id in oil_national_ghgi['emi_id'].values:
                        df = national_to_state_emi(emi_id, oil_national_ghgi, oil_state_surrogate, surrogate_name)
                    elif emi_id in gas_national_ghgi['emi_id'].values:
                        df = national_to_state_emi(emi_id, gas_national_ghgi, gas_state_surrogate, surrogate_name)
                    else:
                        print(f"Warning: emi_id '{emi_id}' not found in national GHGI data for state proportion calculation")
                        continue
                        
                elif ghgi_data_level == "basin" and surrogate_tab == "basin_surrogate":
                    # Basin to State by proportion Calculation - check which DataFrame contains this emi_id
                    if emi_id in oil_basin_ghgi['emi_id'].values:
                        df = basin_to_state_emi(emi_id, oil_basin_ghgi, basin_surrogate, surrogate_name)
                    elif emi_id in gas_basin_ghgi['emi_id'].values:
                        df = basin_to_state_emi(emi_id, gas_basin_ghgi, basin_surrogate, surrogate_name)
                    else:
                        print(f"Warning: emi_id '{emi_id}' not found in basin GHGI data")
                        continue
                        
                else:
                    print(f"Error: No matching logic path for emi_id '{emi_id}' with ghgi_data_level='{ghgi_data_level}' and surrogate_tab='{surrogate_tab}'")
                    continue
                    
                # Store the result
                results[emi_id] = df
                print(f"Successfully processed emi_id: {emi_id}")
                
            except Exception as e:
                print(f"Error processing emi_id '{emi_id}': {str(e)}")
                continue
        
        return results
    
    # Run the processing
    return process_crosswalk_emissions()

# Run the emissions processing
emission_results = run_oil_gas_emissions_processing()

# %% Write results to CSV files
print(f"\nWriting {len(emission_results)} emission files to {v4_emi_data_dir_path}...")



# Write each emission result to a CSV file
for emi_id, df in emission_results.items():
    output_path = v4_emi_data_dir_path / f"{emi_id}.csv"
    df.to_csv(output_path, index=False)
    print(f"Wrote {emi_id}.csv with {len(df)} rows")

print(f"\nSuccessfully wrote all {len(emission_results)} emission files to CSV!")
