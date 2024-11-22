from pathlib import Path
from typing import Annotated

import geopandas as gpd
import pandas as pd
import numpy as np
from pytask import Product, mark, task

from gch4i.config import (
    max_year,
    min_year,
    proxy_data_dir_path,
    V3_DATA_PATH
)

@mark.persist
@task(id="generators_proxy")
def get_generators_proxy_data(
    #Inputs
    # N/A. No proxy data available. Assuming uniform distribution, as in v2 proxy.

    #Outputs
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "generators_proxy.parquet"
    ),
):
    # load state geometries for continental US
    state_shapes = gpd.read_file(V3_DATA_PATH / 'geospatial/cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
    state_shapes = state_shapes.rename(columns={'STUSPS': 'state_code'})[['state_code', 'geometry']]
    lower_48 = ['AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 
                'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT',
                'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA',
                'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY']
    state_shapes = state_shapes[state_shapes['state_code'].isin(lower_48)]

    # create uniform proxy table
    df = []
    for year in range(min_year, max_year + 1):
        for state in lower_48:
            df.append([year, state, 1])
    df = pd.DataFrame(df, columns=['year', 'state_code', 'rel_emi'])

    # merge on geometries and ensure proxy_df is a GeoDataFrame
    proxy_gdf = gpd.GeoDataFrame(state_shapes.merge(df, on='state_code', how='left'))
    proxy_gdf[['year','state_code','rel_emi','geometry']]

    ###############################################################
    # Normalize and save
    ###############################################################

    # Normalize relative emissions to sum to 1 for each year and state
    proxy_gdf = proxy_gdf.groupby(['state_code', 'year']).filter(lambda x: x['rel_emi'].sum() > 0) #drop state-years with 0 total volume
    proxy_gdf['rel_emi'] = proxy_gdf.groupby(['year', 'state_code'])['rel_emi'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0) #normalize to sum to 1
    sums = proxy_gdf.groupby(["state_code", "year"])["rel_emi"].sum() #get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}" # assert that the sums are close to 1

    # Save to parquet
    proxy_gdf.to_parquet(output_path)

    return
