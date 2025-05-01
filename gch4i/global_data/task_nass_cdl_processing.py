"""
Name:                  task_nass_cdl_processing.py
Date Last Modified:    2025-01-21
Authors Name:          Nick Kruskamp (RTI International)
Purpose:               This script is used to process the nass Cropland Data Layer data
                        into the intermediate data that serve several proxies (all crop,
                        farm pipelines, rice area). The script downloads the raw data,
                        unzips it, and then creates the binary and percentage layers
                        for each crop class value sets.
Input Files:           - nass_cdl_path: {sector_data_dir_path}/nass_cdl
                       - url: ("https://www.nass.usda.gov/Research_and_Science/Cropland/
                            Release/datasets/" f"{zip_file_name}")
Output Files:          - {proxy_data_dir_path}/abd_coal_proxy.parquet
"""

# %% Import Libraries & Setup Paths
# %load_ext autoreload
# %autoreload 2


import multiprocessing
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

import numpy as np
from pytask import Product, mark, task

from gch4i.config import sector_data_dir_path, years
from gch4i.utils import download_url, make_raster_binary, warp_to_gepa_grid

NUM_WORKERS = multiprocessing.cpu_count()

nass_cdl_path = sector_data_dir_path / "nass_cdl"

# https://www.nass.usda.gov/Research_and_Science/Cropland/metadata/metadata_nc23.htm
# the list of all "crop" classification values in CDL. Any value in this list will be
# marked as 1 in the binary layer, and the percentage layer will be the percentage of
# 1 cells in the percentage layer during warping.
cdl_crop_vals = np.concat([np.arange(1, 60), np.arange(66, 77), np.arange(204, 254)]) #EEM Question - what happens here if the code is looking for crop id #7, but there is no category code #7 in the dataset?
rice_crop_vals = np.array([3])
# cdl_other_vals = np.concatenate([np.arange(61, 66), np.arange(81, 195)])

crop_val_dict = {
    "all_crop": cdl_crop_vals,
    "rice": rice_crop_vals,
}

# %% Unzip CDL

def unzip_cdl(zip_path, output_path):
    """unzip an input path zip_path to the output path"""
    with ZipFile(zip_path, "r") as z:
        z.extract(output_path.name, output_path.parent)


# %% Download all the years of NASS CDL data
cdl_download_dict = {}
for year in years:

    zip_file_name = f"{year}_30m_cdls.zip"
    tif_file_name = f"{year}_30m_cdls.tif"

    url = (
        "https://www.nass.usda.gov/Research_and_Science/Cropland/Release/datasets/"
        f"{zip_file_name}"
    )
    zip_path = nass_cdl_path / zip_file_name
    cdl_input_path = nass_cdl_path / tif_file_name

    cdl_download_dict[f"cdl_dl_{year}"] = dict(
        url=url,
        zip_path=zip_path,
        cdl_path=cdl_input_path,
    )


# %% Pytask Function
for _id, kwargs in cdl_download_dict.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_download_cdl(
        url: str,
        zip_path: Annotated[Path, Product],
    ):
        download_url(url, zip_path)


# %% Unzip all the CDL data we need
cdl_unzip_dict = {}
for year in years:

    zip_file_name = f"{year}_30m_cdls.zip"
    tif_file_name = f"{year}_30m_cdls.tif"

    zip_path = nass_cdl_path / zip_file_name
    cdl_input_path = nass_cdl_path / tif_file_name

    cdl_unzip_dict[f"cdl_unzip_{year}"] = dict(
        zip_path=zip_path,
        cdl_path=cdl_input_path,
    )

# %% Pytask Function
for _id, kwargs in cdl_unzip_dict.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_unzip_cdl(
        zip_path: Annotated[Path, Product],
        cdl_path: Annotated[Path, Product],
    ):
        unzip_cdl(zip_path, cdl_path)


# %% Calculate the binary and percentage crop layers

# create the binary and percentage crop layers for each crop type
# the will take the raw 30 meter data, create a binary layer for the list of crop values
# in CDL, and then warp that binary layer to the GEPA grid
# the byproducts of these are then used in different proxies.
calc_crop_perc = {}
for crop_name, crop_vals in crop_val_dict.items():
    for year in years:
        tif_file_name = f"{year}_30m_cdls.tif"
        cdl_input_path = nass_cdl_path / tif_file_name
        output_path_binary = nass_cdl_path / (
            cdl_input_path.stem + f"_{crop_name}_binary.tif"
        )
        output_path_perc = nass_cdl_path / (
            cdl_input_path.stem + f"_{crop_name}_perc.tif"
        )
        calc_crop_perc[f"cdl_{year}_{crop_name}"] = dict(
            cdl_input_path=cdl_input_path,
            output_path_binary=output_path_binary,
            output_path_perc=output_path_perc,
            crop_vals=crop_vals,
        )
calc_crop_perc

# %% Pytask Function
for _id, kwargs in calc_crop_perc.items():

    @mark.persist
    @task(id=_id, kwargs=kwargs)
    def task_calc_cdl_perc(
        cdl_input_path: Path,
        output_path_binary: Annotated[Path, Product],
        output_path_perc: Annotated[Path, Product],
        crop_vals: np.array,
    ):

        make_raster_binary(
            input_path=cdl_input_path,
            output_path=output_path_binary,
            true_vals=crop_vals,
            num_workers=NUM_WORKERS,
        )
        warp_to_gepa_grid(
            input_path=output_path_binary,
            output_path=output_path_perc,
            num_threads=NUM_WORKERS,
        )
