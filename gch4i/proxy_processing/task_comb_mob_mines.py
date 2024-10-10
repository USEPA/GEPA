# %%

# import concurrent
import multiprocessing

# import threading
from pathlib import Path
from typing import Annotated
from zipfile import ZipFile

# from pyarrow import parquet  # noqa
import osgeo  # noqa
import geopandas as gpd
import matplotlib.pyplot as plt

# import numpy as np
import pandas as pd

# import rasterio
import requests

# import rioxarray

# import xarray as xr
# import pytask
from pytask import Product, mark, task

# from rasterio import shutil as rio_shutil
# from rasterio.enums import Resampling

from gch4i.config import (
    V3_DATA_PATH,
    global_data_dir_path,
    proxy_data_dir_path,
    sector_data_dir_path,
    years,
)
# from gch4i.gridding import GEPA_spatial_profile

num_workers = multiprocessing.cpu_count()



# %%


@mark.persist
@task(id="mines_proxy")
def task_mob_comb_mines(
    msha_path: Path = V3_DATA_PATH / "sector/abandoned_mines/Mines.zip",
    state_path: Path = global_data_dir_path / "tl_2020_us_state.zip",
    output_path: Annotated[Path, Product] = proxy_data_dir_path / "mines_proxy.parquet",
):

    with ZipFile(msha_path) as z:
        with z.open("Mines.txt") as f:
            msha_df = (
                pd.read_table(
                    f,
                    sep="|",
                    encoding="ISO-8859-1",
                    usecols=["MINE_ID", "LATITUDE", "LONGITUDE", "CURRENT_MINE_STATUS"],
                )
                .query("CURRENT_MINE_STATUS == 'Active'")
                .dropna(subset=["LATITUDE", "LONGITUDE"])
                .set_index("MINE_ID")
            )

    state_gdf = (
        gpd.read_file(state_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
    )

    state_union = state_gdf.union_all()

    msha_gdf = gpd.GeoDataFrame(
        msha_df.drop(columns=["LATITUDE", "LONGITUDE"]),
        geometry=gpd.points_from_xy(msha_df["LONGITUDE"], msha_df["LATITUDE"]),
        crs=4326,
    )
    msha_gdf = msha_gdf[msha_gdf.intersects(state_union)]
    print("Active mines with location: ", len(msha_gdf))
    msha_gdf.to_parquet(output_path)

    _, ax = plt.subplots(figsize=(20, 20), dpi=150)
    state_gdf.boundary.plot(color="xkcd:slate", ax=ax)
    msha_gdf.plot(ax=ax, color="xkcd:goldenrod", markersize=1)

    # %%


# if ReCalc_Crop ==1:
#     map_crop = np.zeros([len(Lat_01),len(Lon_01),num_years])
#     map_crop_nongrid = np.zeros(num_years)

#     for iyear in np.arange(0,num_years):
#         crop_loc = pd.read_csv(Crop_file+year_range_str[iyear]+'_001x001.csv', sep=',')
#         for idx in np.arange(0,len(crop_loc)):
#             if crop_loc['FIRST_Longitude'][idx] > Lon_left and crop_loc['FIRST_Longitude'][idx] < Lon_right and \
#                 crop_loc['FIRST_Latitude'][idx] > Lat_low and crop_loc['FIRST_Latitude'][idx] < Lat_up:
#                 ilat = int((crop_loc['FIRST_Latitude'][idx]  - Lat_low)/Res01)
#                 ilon = int((crop_loc['FIRST_Longitude'][idx] - Lon_left)/Res01)
#                 map_crop[ilat,ilon,iyear] += crop_loc['SUM_Area_AllCrops'][idx]
#             else:
#                 map_crop_nongrid[iyear] += crop_loc['SUM_Area_AllCrops'][idx]
#         print('Year:', year_range_str[iyear])
#         print ('Database crop area: ', np.sum(crop_loc['SUM_Area_AllCrops']))
#         print ('Gridded  crop area: ', np.sum(map_crop[:,:,iyear]))
#         ct = datetime.datetime.now()
#         print("current time:", ct)
#     np.save('./IntermediateOutputs/Crops_tempoutput', map_crop)
#     np.save('./IntermediateOutputs/Crops_nongrid_tempoutput', map_crop_nongrid)

# else:
#     map_crop = np.load('./IntermediateOutputs/Crops_tempoutput.npy')
#     map_crop_nongrid = np.load('./IntermediateOutputs/Crops_nongrid_tempoutput.npy')
#     for iyear in np.arange(0,num_years):
#         print('Year:', year_range_str[iyear])
#         #print ('Database crop area: ', np.sum(crop_loc['SUM_Area_AllCrops']))
#         print ('Gridded  crop area: ', np.sum(map_crop[:,:,iyear]))
#         ct = datetime.datetime.now()
#         print("current time:", ct)
# %%




# session = pytask.build(
#     tasks=[task_download_cdl(**kwargs) for _, kwargs in _ID_TO_KWARGS_DL_CDL.items()]
# )


# %%


# zip_path: Path = nass_cdl_path / "2012_30m_cdls.zip"
# input_path = nass_cdl_path / "2012_30m_cdls.tif"
# output_path_crop_binary = nass_cdl_path / (input_path.stem + "_binary.tif")
# output_path_crop_count = nass_cdl_path / (input_path.stem + "_crop_count.tif")
# # %%


# @task
# def task_reclass_cdl(
#     input_path: Path, output_path: Annotated[Path, Product], num_workers: int = 4
# ):

#     with rasterio.open(input_path) as src:

#         # Create a destination dataset based on source params. The
#         # destination will be tiled, and we'll process the tiles
#         # concurrently.
#         profile = src.profile
#         profile.update(blockxsize=128, blockysize=128, tiled=True)

#         with rasterio.open(output_path, "w", **src.profile) as dst:
#             windows = [window for ij, window in dst.block_windows()]

#             # We cannot write to the same file from multiple threads
#             # without causing race conditions. To safely read/write
#             # from multiple threads, we use a lock to protect the
#             # DatasetReader/Writer
#             read_lock = threading.Lock()
#             write_lock = threading.Lock()

#             def process(window):
#                 with read_lock:
#                     src_array = src.read(window=window)

#                 # NOTE: from the CDL metadata, valid CROP values are 1-60
#                 # The computation can be performed concurrently
#                 result = np.where((src_array >= 1) & (src_array <= 60), 1, 0)

#                 with write_lock:
#                     dst.write(result, window=window)

#             # We map the process() function over the list of
#             # windows.
#             with concurrent.futures.ThreadPoolExecutor(
#                 max_workers=num_workers
#             ) as executor:
#                 executor.map(process, windows)


# # %%
# def task_warp_cdl(input_path: Path, output_path: Path, vrt_options: dict):
#     # NOTE: mode == majority from the previous GIS methodology
#     profile = GEPA_spatial_profile(0.1)
#     vrt_options = {
#         "resampling": Resampling.sum,
#         # "resampling": Resampling.mode,
#         "crs": profile.profile["crs"],
#         "transform": profile.profile["transform"],
#         "height": profile.height,
#         "width": profile.width,
#     }

#     env = rasterio.Env(
#         GDAL_DISABLE_READDIR_ON_OPEN="EMPTY_DIR",
#         CPL_VSIL_CURL_USE_HEAD=False,
#         CPL_VSIL_CURL_ALLOWED_EXTENSIONS="TIF",
#     )
#     # with env:
#     #     with rasterio.open(output_path) as src:
#     #         with WarpedVRT(src, **vrt_options) as vrt:
#     #             # At this point 'vrt' is a full dataset with dimensions,
#     #             # CRS, and spatial extent matching 'vrt_options'.

#     #             # Read all data into memory.
#     #             # data = vrt.read()
#     #             # rio_shutil.copy(vrt, output_path_crop_count, driver='GTiff')

#     # with env:
#     #     with rasterio.open(input_path) as src:
#     #         with rasterio.vrt.WarpedVRT(src, **vrt_options) as vrt:
#     #             rds = rioxarray.open_rasterio(vrt)
#     #             rds.sel(band=1).plot.imshow()

#     # https://corteva.github.io/rioxarray/html/examples/reproject.html
#     # - current recommendation for large warping. Still very memory intensive.

#     # https://github.com/rasterio/rasterio/issues/1990
#     # - there are updates coming in rasterio but it will be a while.

#     # https://gdal.org/en/latest/programs/gdalwarp.html
#     # - we could use gdalwarp directly but it is not as easy to use as rasterio.
#     with env:
#         with rasterio.open(input_path) as src:
#             with rasterio.vrt.WarpedVRT(src, **vrt_options) as vrt:
#                 rds = rioxarray.open_rasterio(vrt)
#                 rds.sel(band=1).plot.imshow()
#                 rds.rio.write_crs(vrt_options["crs"]).to_netcdf(
#                     output_path.with_suffix(".nc")
#                 )

#     # area_path = Path(
#     #     "C:/Users/nkruskamp/Environmental Protection Agency (EPA)/"
#     #     "Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data/global/"
#     #     "gridded_area_01_cm2.tif"
#     # )
#     # area_ds = rioxarray.open_rasterio(area_path)

#     # crop_ds = rioxarray.open_rasterio(output_path).rio.reproject_match(
#     #     area_ds, resampling=Resampling.sum
#     # )
#     # crop_ds

#     # https://docs.xarray.dev/en/stable/generated/xarray.ones_like.html
#     # if we needed to count the number of total pixels in a grid cell to calculate
#     # the percentage of crop pixels, we could create an all 1 array, sum it up, the
#     # divide the crop array by the total pixel count array.


# def task_stack_cdl(input_paths: Path, output_path: Path):
#     pass


# # %%
