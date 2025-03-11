"""
Name:                   task_farm_pipelines_proxy.py
Date Last Modified:     2025-02-07
Authors Name:           John Bollenbacher (RTI International)
Purpose:                This file builds spatial proxies for natural gas pipelines on
                        farmlands.
                        Reads pipeline geometries, Finds spatial intersection of
                        pipelines and cropland for each year, and saves out a table of
                        the resulting farm pipeline geometries for each year and state.
Input Files:            - {sector_data_dir_path}/enverus/midstream/
                            Rextag_Natural_Gas.gdb
                        - {sector_data_dir_path}/nass_cdl/
                            {year}_30m_cdls_all_crop_binary.tif
                        - {global_data_dir_path}/tl_2020_us_state.zip
Output Files:           - farm_pipelines_proxy.parquet
"""
# %% Import Libraries
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
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path
)

verbose = False


# %% Pytask Function
@mark.persist
@task(id="farm_pipeline_proxy")
def get_farm_pipeline_proxy_data(
    # Inputs
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    # Outputs
    output_path: Annotated[Path, Product] = proxy_data_dir_path /
        "farm_pipelines_proxy.parquet"
):

    ###############################################################
    # Load and prepare pipeline data
    ###############################################################

    # Input path cannnot be in function because it is a directory, not a file
    croplands_path_template: Path = sector_data_dir_path / "nass_cdl/{year}_30m_cdls_all_crop_binary.tif"
    enverus_midstream_pipelines_path: Path = sector_data_dir_path / "enverus/midstream/Rextag_Natural_Gas.gdb"

    if verbose:
        print('loading state and pipeline data')

    # Load state shapes
    state_shapes = (
        gpd.read_file(state_path)
        .rename(columns={"STUSPS": "state_code"})
        .astype({"STATEFP": int})
        # get only lower 48 + DC
        .query("(STATEFP < 60) & (STATEFP != 2) & (STATEFP != 15)")
        .loc[:, ["state_code", "geometry"]]
        .to_crs(epsg=4326)
    )

    # Load and filter pipeline data
    enverus_pipelines_gdf = gpd.read_file(
        enverus_midstream_pipelines_path,
        layer='NaturalGasPipelines')
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

        if verbose:
            print('  loading cropland raster data')
        with rasterio.open(croplands_path) as src:
            # Read cropland data
            cropland_mask = src.read(1) > 0
            transform = src.transform
            raster_crs = src.crs

            # Reproject pipelines to raster's CRS for raster operations
            year_pipelines_gdf_raster_crs = year_pipelines_gdf.to_crs(raster_crs).copy()

            # Rasterize pipelines to same grid as cropland
            if verbose:
                print('  rasterizing pipelines')
            pipeline_shapes = (
                (geom, 1) for geom in year_pipelines_gdf_raster_crs.geometry)
            pipeline_raster = rasterize(
                pipeline_shapes,
                out_shape=cropland_mask.shape,
                transform=transform,
                dtype=np.uint8
            )

            # Find intersection areas
            if verbose:
                print('  finding cropland-pipelines intersection')
            intersection_mask = (cropland_mask & (pipeline_raster > 0))

        ###############################################################
        # Find vector intersection using raster intersection
        ###############################################################

            # Vectorize only the intersection areas
            if verbose:
                print(' vectorizing cropland raster bins which intersect pipeline bins')
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
            if verbose:
                print('  put intersecting cropland geometries into gdf')
            if intersection_geometries:
                intersection_gdf = gpd.GeoDataFrame.from_features(
                    intersection_geometries,
                    crs=raster_crs
                ).to_crs(epsg=4326)

                # Intersect with original pipelines to maintain pipeline attributes
                if verbose:
                    print('  computing vector overlay between cropland and pipelines')
                year_pipelines_gdf = gpd.overlay(
                    year_pipelines_gdf,
                    intersection_gdf,
                    how='intersection',
                    keep_geom_type=True
                )

                # Recombine each pipeline into single geometry
                if verbose:
                    print('  recombining split pipeline geometries')
                year_pipelines_gdf = year_pipelines_gdf.dissolve(
                    # if rows match on all these columns, then combine their geometries
                    # into a single geometry in one row
                    by=['LOC_ID', 'DIAMETER'],
                    # for other columns besides geometry and the matched columns, use
                    # the first row's values. there are no other columns, though.
                    aggfunc='first'
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

        if verbose:
            print('  splitting by state')

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

        if verbose:
            print('  computing pipeline lengths and final proxy gdf')

        # Calculate pipeline lengths
        proxy_gdf = proxy_gdf.to_crs("ESRI:102003")
        proxy_gdf['rel_emi'] = proxy_gdf.geometry.length
        proxy_gdf = proxy_gdf.to_crs(epsg=4326)

        # Normalize relative emissions to sum to 1 for each year and state
        # Drop years with zero volume
        proxy_gdf = proxy_gdf.groupby(['year']).filter(lambda x: x['rel_emi'].sum() > 0)
        # normalize to sum to 1
        proxy_gdf['rel_emi'] = (
            proxy_gdf.groupby(['year'])['rel_emi']
            .transform(lambda x: x / x.sum() if x.sum() > 0 else 0))
        # get sums to check normalization
        sums = proxy_gdf.groupby(["year"])["rel_emi"].sum()
        # assert that the sums are close to 1
        assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"

        return proxy_gdf

    ###############################################################
    # Process each year, aggregate, and save
    ###############################################################

    year_proxy_gdfs = []
    for year in range(min_year, max_year+1):
        if verbose:
            print(f'Processing year {year}')
        croplands_path = croplands_path_template.parent / croplands_path_template.name.format(year=year)
        year_proxy_gdf = process_year(year, enverus_pipelines_gdf, croplands_path, state_shapes)
        year_proxy_gdfs.append(year_proxy_gdf)

    # Combine all years into single dataframe
    proxy_gdf = pd.concat(year_proxy_gdfs, ignore_index=True)

    # Double check normalization
    # get sums to check normalization
    sums = proxy_gdf.groupby(["year"])["rel_emi"].sum()
    # assert that the sums are close to 1
    assert np.isclose(sums, 1.0, atol=1e-8).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"

    # Drop state_code column
    # proxy_gdf = proxy_gdf.drop(columns=['state_code'])

    # Save to parquet
    proxy_gdf.to_parquet(output_path)

#     return proxy_gdf
# df = get_farm_pipeline_proxy_data()
# df.info()
# df

# %%
