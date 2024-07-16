# %%
# download the census geometry files we need for gridded methane v3.
from pathlib import Path
from typing import Annotated
from bs4 import BeautifulSoup
from pyarrow import parquet   # noqa: F401
import geopandas as gpd
import pandas as pd
import osgeo   # noqa: F401
import requests
from pytask import Product, mark, task

from gch4i.config import (census_geometry_list, grid_geometry_list, split_geometry_dict,
                          global_data_dir_path, years)
from gch4i.gridding import vector_to_gepa_grid


def create_kwargs(geometry_list, grid=False, split=False):
    """
    Creates a dictionary of pytask arguments representing tasks for each year and
    each of the listed TIGERLINE geometry types.
    """
    id_to_kwargs = {}
    # iterate through years and geometry types to create tasks
    for year in years:
        for geometry_type in geometry_list:
            input_path = (global_data_dir_path /
                          f"raw/tl_{year}_us_{geometry_type.lower()}.parquet")
            if grid:  # creates task for gridding a specific geometry type
                label = f"{str(year)}_{geometry_type}"
                grid_path = (global_data_dir_path /
                             f"gridded_{year}_{geometry_type.lower()}.parquet")
                id_to_kwargs[label] = {"vector_data": input_path,
                                       "grid_path": grid_path}
            elif split:  # creates task for splitting geometry types by a specific field
                field = list(split_geometry_dict[geometry_type].keys())[0]
                for k, v in split_geometry_dict[geometry_type][field].items():
                    label = f"{str(year)}_{k}"
                    # the output file uses the TIGER file naming convention
                    output_path = (global_data_dir_path /
                                   f"raw/tl_{year}_us_{k.lower()}.parquet")
                    id_to_kwargs[label] = {"vector_data": input_path,
                                           "field": field,
                                           "value": v,
                                           "output_path": output_path}
            else:  # creates task for downloading and combining for a geometry type
                label = f"{str(year)}_{geometry_type}"
                # create the URL
                url = (
                    f"https://www2.census.gov/geo/tiger/"
                    f"TIGER{year}/{geometry_type.upper()}/"
                )
                # the output file uses the TIGER file naming convention
                output_path = (global_data_dir_path /
                               f"raw/tl_{year}_us_{geometry_type.lower()}.parquet")
                id_to_kwargs[label] = {"url": url, "output_path": output_path}
    return id_to_kwargs


_ID_TO_DL_KWARGS = create_kwargs(census_geometry_list)
for _id, kwargs in _ID_TO_DL_KWARGS.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_download_census_geo(url: str, output_path: Annotated[Path, Product]):
        with requests.get(url) as r:
            soup = BeautifulSoup(r.content, features="html.parser")
        # get all the zip files on the page
        zip_urls = []
        for x in soup.find_all("a"):
            if x.text.endswith(".zip"):
                zip_url = url + x.get("href")
                zip_urls.append(zip_url)
        # if there is a US file, use that one.
        us_zip_file = [x for x in zip_urls if "_us_" in x]
        if us_zip_file:
            zip_urls = us_zip_file
            out_df = gpd.read_file(us_zip_file[0])
            out_df.to_parquet(output_path)
        # otherwise, read the files and concat together.
        else:
            out_gdf = pd.concat([gpd.read_file(x) for x in zip_urls])
            out_gdf.to_parquet(output_path)


_ID_TO_SPLIT_KWARGS = create_kwargs(split_geometry_dict.keys(), split=True)
for _id, kwargs in _ID_TO_SPLIT_KWARGS.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_split_global_data(vector_data: str, field: str, value: str,
                               output_path: Annotated[Path, Product]):
        vector_gdf = gpd.read_parquet(vector_data)
        vector_gdf = vector_gdf[vector_gdf[field] == value]
        vector_gdf.to_parquet(output_path)


_ID_TO_GRID_KWARGS = create_kwargs(grid_geometry_list, grid=True)
for _id, kwargs in _ID_TO_GRID_KWARGS.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_grid_global_data(vector_data: str, grid_path: Annotated[Path, Product]):
        vector_to_gepa_grid(vector_data, grid_path)
