"""
Name:                  task_download_breweries.py
Date Last Modified:    2025-09-25
Authors Name:          Andrew Burnette (RTI International)
Purpose:               This script is used to retrieve a dataset of breweries in the
                            United States from a public URL and save it to a specified
                            directory.
Input Files:           - url: ("https://www.openbrewerydb.org/breweries")
Output Files:          - {V4_DATA}/open_data/industrial_wastewater/
                            openbrewerydb_geolocated.csv
Notes:                 - API documentation: https://www.openbrewerydb.org/documentation
"""
# %% Import Libraries & Setup Paths

import requests
import geopandas as gpd
import pandas as pd

from gch4i.config import (
    v4_global_data_dir_path,
    v4_open_data_dir_path,
    min_year
)

state_path = v4_global_data_dir_path / "tl_2020_us_state.zip"
output_path = v4_open_data_dir_path / "industrial_wastewater" / "openbrewerydb_geolocated.csv"


# %% Retrieve Breweries Data

def get_all_breweries_by_type(brewery_type, per_page=200):
    url = "https://api.openbrewerydb.org/v1/breweries"
    page = 1
    all_results = []

    while True:
        params = {"by_type": brewery_type, "per_page": per_page, "page": page}
        resp = requests.get(url, params=params)
        resp.raise_for_status()
        data = resp.json()

        if not data:  # no more results
            break

        all_results.extend(data)
        page += 1

    return all_results

# Collect all three types
combined = []
for t in ["micro", "regional", "large"]:
    breweries = get_all_breweries_by_type(t)
    print(f"Fetched {len(breweries)} {t} breweries")
    combined.extend(breweries)

print(f"\nTotal combined breweries: {len(combined)}")

# Example: print first 5
for b in combined[:5]:
    print(b["name"], "-", b["brewery_type"], "-", b["city"], b["state"])


# %% Grab State data

state_df = (
    gpd.read_file(state_path)
    .loc[:, ["NAME", "STUSPS", "STATEFP"]]
    .rename(columns=str.lower)
    .rename(columns={"stusps": "state_code", "name": "state_name"})
    .astype({"statefp": int})
    # get only lower 48 + DC
    .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
    .drop(columns=["statefp"])
    .reset_index(drop=True)
    )

# %% CLEAN

# Convert to DataFrame
brewery_df = pd.DataFrame(combined)

# Missouri is misspelled as MIssouri in some entries
# Recommend titlecase and convert to abbreviation

brewery_df = (
    brewery_df[['state', 'country', 'latitude', 'longitude']]
    .query("country == 'United States'")
    .dropna(subset=['latitude', 'longitude'])
    .assign(state=lambda x: x['state'].str.title())
    .assign(data_source="brewerydb")
    .assign(year=min_year)
    .query('state != "Hawaii" and state != "Alaska"')
    .merge(state_df, left_on='state', right_on='state_name', how='left')
)

output_df = brewery_df[[
    'year',
    'state_code',
    'latitude',
    'longitude',
    'data_source'
    ]].sort_values(by='state_code', ascending=True).reset_index(drop=True)

# %% Save Output
output_df.to_csv(output_path, index=False)



# %%
