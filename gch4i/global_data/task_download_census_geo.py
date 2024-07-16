# %%
# download the census geometry files we need for gridded methane v3.
from pathlib import Path
from typing import Annotated
from bs4 import BeautifulSoup
import geopandas as gpd
import pandas as pd
import requests
from pytask import Product, mark, task

from gch4i.config import census_geometry_list, global_data_dir_path, years


def create_kwargs(geometry_list):
    id_to_kwargs = {}
    for year in years:
        # for each of the geometry types listed
        for geometry_type in geometry_list:
            # create the URL
            url = (
                f"https://www2.census.gov/geo/tiger/"
                f"TIGER{year}/{geometry_type.upper()}/"
            )
            # the output file uses the TIGER file naming convention
            output_path = (global_data_dir_path / 
                           f"tl_{year}_us_{geometry_type.lower()}.zip")
            id_to_kwargs[geometry_type] = {"url": url, "output_path": output_path}
    return id_to_kwargs


_ID_TO_KWARGS = create_kwargs(census_geometry_list)

for _id, kwargs in _ID_TO_KWARGS.items():

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

        # read the files and concat together.
        out_gdf = pd.concat([gpd.read_file(x) for x in zip_urls])
        # write output
        out_gdf.to_file(output_path, driver="ESRI Shapefile")
