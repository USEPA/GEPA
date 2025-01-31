#%%
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
    sector_data_dir_path,
    V3_DATA_PATH
)
from gch4i.utils import (
    us_state_to_abbrev_dict
)

lower_48_and_DC = ('AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 
            'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT',
            'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA',
            'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY', 'DC')

@mark.persist
@task(id="trans_pipeline_proxy")
def get_trans_pipeline_proxy_data(
    #Inputs
    enverus_midstream_pipelines_path: Path = sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb",

    #Outputs
    output_path: Annotated[Path, Product] = (
        proxy_data_dir_path / "trans_pipeline_proxy.parquet"
    ),
):
    ###############################################################
    # Load data
    ###############################################################

    # load state geometries for continental US
    state_shapes = gpd.read_file(V3_DATA_PATH / 'geospatial/cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
    state_shapes = state_shapes.rename(columns={'STUSPS': 'state_code'})[['state_code', 'geometry']]
    state_shapes = state_shapes[state_shapes['state_code'].isin(lower_48_and_DC)]
    state_shapes = state_shapes.to_crs(epsg=4326)  # Ensure state_shapes is in EPSG:4326

    # load enverus pipelines geometries data
    enverus_pipelines_gdf = gpd.read_file(enverus_midstream_pipelines_path, layer='NaturalGasPipelines')
    enverus_pipelines_gdf = enverus_pipelines_gdf.to_crs(epsg=4326)  # Ensure enverus_pipelines_gdf is in EPSG:4326

    #drop unneeded pipelines and columns
    # enverus_pipelines_gdf = enverus_pipelines_gdf[enverus_pipelines_gdf['CNTRY_NAME']=='United States'] #dropped, because not used in v2 and state intersection will filter this.
    enverus_pipelines_gdf = enverus_pipelines_gdf[enverus_pipelines_gdf['STATUS']=='Operational']
    enverus_pipelines_gdf = enverus_pipelines_gdf[enverus_pipelines_gdf['TYPE'] =='Transmission']
    enverus_pipelines_gdf = enverus_pipelines_gdf[['LOC_ID','DIAMETER','geometry', #'INSTALL_YR',
                                                   ]].reset_index(drop=True)
    
    # Split pipelines at state boundaries and keep only the segments that fall within states
    enverus_pipelines_gdf_split_by_state = gpd.overlay(enverus_pipelines_gdf, state_shapes, how='intersection', keep_geom_type=True)

    #create each year's proxy, and combine to make the full proxy table
    # NOTE: this assumes the geometries do not change year to year, since we dont have good data for when each pipeline began operation. 
    year_gdfs = []
    for year in range(min_year, max_year+1):
        year_gdf = enverus_pipelines_gdf_split_by_state.copy()
        # year_gdf = year_gdf[year_gdf['INSTALL_YR'] > year] # not applicable since most INSTALL_YR are NaN
        year_gdf['year'] = year
        year_gdfs.append(year_gdf)
    proxy_gdf = pd.concat(year_gdfs, ignore_index=True)

    #compute the length of each pipeline within each state
    proxy_gdf = proxy_gdf.to_crs("ESRI:102003")  # Convert to ESRI:102003
    proxy_gdf['pipeline_length_within_state'] = proxy_gdf.geometry.length  
    proxy_gdf = proxy_gdf.to_crs(epsg=4326)  # Convert back to EPSG:4326

    #get rel_emi
    proxy_gdf = proxy_gdf.rename(columns = {'pipeline_length_within_state':'rel_emi'}) #simply assume rel_emi is proportional to length.

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

    return proxy_gdf

# proxy_gdf = get_trans_pipeline_proxy_data()

# #%%
# proxy_gdf.info()
# proxy_gdf
