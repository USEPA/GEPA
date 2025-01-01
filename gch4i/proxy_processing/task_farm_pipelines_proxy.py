#%%
from pathlib import Path
from typing import Annotated

import geopandas as gpd
import pandas as pd
import numpy as np
from pytask import Product, mark, task
import rasterio
from rasterio.features import rasterize, shapes
from shapely.geometry import shape

from gch4i.config import (
    max_year,
    min_year,
    proxy_data_dir_path,
    sector_data_dir_path,
    V3_DATA_PATH
)

lower_48_and_DC = ('AL', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DE', 'FL', 'GA', 'IA', 'ID', 'IL', 
            'IN', 'KS', 'KY', 'LA', 'MA', 'MD', 'ME', 'MI', 'MN', 'MO', 'MS', 'MT',
            'NC', 'ND', 'NE', 'NH', 'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA',
            'RI', 'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', 'WY', 'DC')

verbose = False

@mark.persist
@task(id="farm_pipeline_proxy")
def get_farm_pipeline_proxy_data(
    #Inputs
    enverus_midstream_pipelines_path: Path = sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb",
    croplands_path_template: Path = sector_data_dir_path / "nass_cdl/{year}_30m_cdls_all_crop_binary.tif",

    #Outputs
    output_path: Annotated[Path, Product] = proxy_data_dir_path / "farms_pipeline_proxy.parquet"
):
    
    ###############################################################
    # Load and prepare pipeline data
    ###############################################################

    if verbose: print('loading state and pipeline data')
    
    # Load state geometries and pipeline data (keeping existing code)
    state_shapes = gpd.read_file(V3_DATA_PATH / 'geospatial/cb_2018_us_state_500k/cb_2018_us_state_500k.shp')
    state_shapes = state_shapes.rename(columns={'STUSPS': 'state_code'})[['state_code', 'geometry']]
    state_shapes = state_shapes[state_shapes['state_code'].isin(lower_48_and_DC)]
    state_shapes = state_shapes.to_crs(epsg=4326)

    # Load and filter pipeline data
    enverus_pipelines_gdf = gpd.read_file(enverus_midstream_pipelines_path, layer='NaturalGasPipelines')
    enverus_pipelines_gdf = enverus_pipelines_gdf.to_crs(epsg=4326)
    enverus_pipelines_gdf = enverus_pipelines_gdf[
        (enverus_pipelines_gdf['STATUS'] == 'Operational') &
        (enverus_pipelines_gdf['TYPE'] == 'Transmission')
    ][['LOC_ID', 'DIAMETER', 'geometry']].reset_index(drop=True)

    ###############################################################
    # Make one year's proxy gdf
    ###############################################################

    def process_year(year, enverus_pipelines_gdf, croplands_path, state_shapes):

        year_pipelines_gdf = enverus_pipelines_gdf.copy()

        ###############################################################
        # Load cropland data and find raster intersection
        ###############################################################

        if verbose: print('  loading cropland raster data')    
        with rasterio.open(croplands_path) as src:
            # Read cropland data
            cropland_mask = src.read(1) > 0
            transform = src.transform
            raster_crs = src.crs

            #reproject pipelines to raster's CRS for raster operations
            year_pipelines_gdf_raster_crs = year_pipelines_gdf.to_crs(raster_crs).copy()
                
            # Rasterize pipelines to same grid as cropland
            if verbose: print('  rasterizing pipelines')
            pipeline_shapes = ((geom, 1) for geom in year_pipelines_gdf_raster_crs.geometry)
            pipeline_raster = rasterize(
                pipeline_shapes, 
                out_shape=cropland_mask.shape,
                transform=transform,
                dtype=np.uint8
            )

            # Find intersection areas
            if verbose: print('  finding cropland-pipelines intersection')
            intersection_mask = (cropland_mask & (pipeline_raster > 0))
            
        ###############################################################
        # Find vector intersection using raster intersection
        ###############################################################

            # Vectorize only the intersection areas
            if verbose: print('  vectorizing cropland raster bins which intersect pipeline bins')
            intersection_shapes = shapes(
                intersection_mask.astype(np.uint8), 
                transform=transform,
                connectivity=4,
                mask=intersection_mask
            )
            intersection_geometries = [
                {"geometry": shape(geom), "properties": {"value": val}}
                for geom, val in intersection_shapes if val == 1
            ]
        
            # Convert intersection shapes to GeoDataFrame
            if verbose: print('  put intersecting cropland geometries into gdf')
            if intersection_geometries:
                intersection_gdf = gpd.GeoDataFrame.from_features(
                    intersection_geometries,
                    crs=raster_crs
                ).to_crs(epsg=4326)
                
                # Intersect with original pipelines to maintain pipeline attributes
                if verbose: print('  computing vector overlay between cropland and pipelines')
                year_pipelines_gdf = gpd.overlay(
                    year_pipelines_gdf,
                    intersection_gdf,
                    how='intersection',
                    keep_geom_type=True
                )

                # Recombine each pipeline into single geometry
                if verbose: print('  recombining split pipeline geometries')
                year_pipelines_gdf = year_pipelines_gdf.dissolve(
                    by=['LOC_ID', 'DIAMETER'], #if rows match on all these columns, then combine their geometries into a single geometry in one row
                    aggfunc='first' #for other columns besides geometry and the matched columns, use the first row's values. there are no other columns, though.
                ).reset_index()

            else:
                # Create empty GeoDataFrame with same columns as enverus_pipelines_gdf
                year_pipelines_gdf = gpd.GeoDataFrame(
                    columns=year_pipelines_gdf.columns,
                    geometry=[],
                    crs=year_pipelines_gdf.crs
                )

        ###############################################################
        # Split by state and create vector proxy gdf
        ###############################################################

        if verbose: print('  splitting by state')
        
        # Split pipelines at state boundaries
        enverus_pipelines_gdf_split_by_state = gpd.overlay(
            year_pipelines_gdf, 
            state_shapes,
            how='intersection', 
            keep_geom_type=True
        )

        proxy_gdf = enverus_pipelines_gdf_split_by_state.copy()
        proxy_gdf['year'] = year

        ###############################################################
        # Calculate pipeline lengths and normalize rel_emi
        ###############################################################

        if verbose: print('  computing pipeline lengths and final proxy gdf')
        
        # Calculate pipeline lengths
        proxy_gdf = proxy_gdf.to_crs("ESRI:102003")
        proxy_gdf['rel_emi'] = proxy_gdf.geometry.length
        proxy_gdf = proxy_gdf.to_crs(epsg=4326)

        # Normalize relative emissions to sum to 1 for each year and state
        proxy_gdf = proxy_gdf.groupby(['state_code', 'year']).filter(lambda x: x['rel_emi'].sum() > 0) #drop state-years with 0 total volume
        proxy_gdf['rel_emi'] = proxy_gdf.groupby(['year', 'state_code'])['rel_emi'].transform(lambda x: x / x.sum() if x.sum() > 0 else 0) #normalize to sum to 1
        sums = proxy_gdf.groupby(["state_code", "year"])["rel_emi"].sum() #get sums to check normalization
        assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}" # assert that the sums are close to 1

        return proxy_gdf
    

    ###############################################################
    # Process each year, aggregate, and save
    ###############################################################

    year_proxy_gdfs = []
    for year in range(min_year, max_year+1):
        if verbose: print(f'Processing year {year}')
        croplands_path = croplands_path_template.parent / croplands_path_template.name.format(year=year)
        year_proxy_gdf = process_year(year, enverus_pipelines_gdf, croplands_path, state_shapes)
        year_proxy_gdfs.append(year_proxy_gdf)
    
    # Double check normalization
    sums = proxy_gdf.groupby(["state_code", "year"])["rel_emi"].sum() #get sums to check normalization
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}" # assert that the sums are close to 1

    # Combine all years into single dataframe
    proxy_gdf = pd.concat(year_proxy_gdfs, ignore_index=True)
    
    # Save to parquet
    proxy_gdf.to_parquet(output_path)

#     return proxy_gdf
# df = get_farm_pipeline_proxy_data()
# df.info()
# df


# %%
