"""
Name:                   task_dask_import_allroads.py
Date Last Modified:     2024-10-14
Authors Name:           A. Burnette (RTI International)
Purpose:                Import all roads data from the US Census Bureau
Input Files:            -
Output Files:           - tl_{year}_us_allroads.parquet
Notes:                  - Script uses dask to mitigate computational load
                        - Writes out parquet files for each year to raw_roads directory
"""


from pathlib import Path
from typing import Annotated
from bs4 import BeautifulSoup
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry
import dask.dataframe as dd
from dask import delayed
import geopandas as gpd
from zipfile import ZipFile
import io
import tempfile
import os

from gch4i.config import (global_data_dir_path, years)

from pytask import Product, mark, task

raw_roads_path = Path(global_data_dir_path) / "raw_roads"
census_geometry_list = ["roads"]


def create_kwargs(geometry_list):
    id_to_kwargs = {}
    for year in years:
        for geometry_type in geometry_list:
            label = f"{str(year)}_{geometry_type}"
            url = f"https://www2.census.gov/geo/tiger/TIGER{year}/{geometry_type.upper()}/"
            output_path = raw_roads_path / f"tl_{year}_us_all{geometry_type.lower()}.parquet"
            id_to_kwargs[label] = {"url": url, "output_path": output_path}
    return id_to_kwargs


# Set up retry strategy
retry_strategy = Retry(
    total=10,
    backoff_factor=3,
    status_forcelist=[429, 500, 502, 503, 504]
)
adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("https://", adapter)
session.mount("http://", adapter)


def download_and_process_file(zip_url, valid_mtfcc, state_fips_code):
    try:
        with session.get(zip_url, stream=True) as response:
            response.raise_for_status()
            with ZipFile(io.BytesIO(response.content)) as zip_file:
                with tempfile.TemporaryDirectory() as temp_dir:
                    zip_file.extractall(path=temp_dir)
                    shp_file = next((os.path.join(temp_dir, f) for f in os.listdir(temp_dir) if f.endswith(".shp")), None)
                    if shp_file:
                        gdf = gpd.read_file(shp_file)
                        gdf_filtered = gdf[gdf['MTFCC'].isin(valid_mtfcc)]
                        gdf_filtered['state_fips'] = state_fips_code
                        return gdf_filtered
    except Exception as e:
        print(f"Error processing {zip_url}: {str(e)}")
    return None


_ID_TO_DL_KWARGS = create_kwargs(census_geometry_list)
for _id, kwargs in _ID_TO_DL_KWARGS.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_download_census_geo(url: str, output_path: Annotated[Path, Product]):
        with session.get(url) as r:
            soup = BeautifulSoup(r.content, features="html.parser")

        valid_fips = list(range(1, 2)) + list(range(4, 7)) + list(range(8, 14)) + list(range(16, 43)) + list(range(44, 52)) + list(range(53, 57))
        valid_mtfcc = ['S1100', 'S1200', 'S1400', 'S1630', 'S1640']

        zip_urls = [url + x.get("href") for x in soup.find_all("a") if x.text.endswith(".zip") and int(x.text.split("_")[2][:2]) in valid_fips]

        print(f"Total number of files to process: {len(zip_urls)}")

        # Use Dask to process files in parallel
        dask_gdf = dd.from_delayed([
            delayed(download_and_process_file)(zip_url, valid_mtfcc, int(zip_url.split("_")[2][:2]))
            for zip_url in zip_urls
        ])

        # Compute and write to parquet
        result = dask_gdf.compute()
        if not result.empty:
            result.to_parquet(output_path)
            print(f"All files combined and written to {output_path}")
        else:
            print("No data was processed. The result is empty.")

# Main execution
if __name__ == "__main__":
    # Ensure the output directory exists
    raw_roads_path.mkdir(parents=True, exist_ok=True)

    # Run the task for each set of arguments
    for kwargs in _ID_TO_DL_KWARGS.values():
        task_download_census_geo(**kwargs)
