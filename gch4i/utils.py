import calendar
import concurrent
import logging
import threading
import time
import warnings
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import numpy as np
import osgeo  # noqa f401
import pandas as pd
import rasterio
import requests
import rioxarray  # noqa f401
import seaborn as sns
import xarray as xr
from geocube.api.core import make_geocube
from geopy.exc import GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import Nominatim
from joblib import Parallel, delayed
from rasterio.enums import Resampling
from rasterio.features import rasterize, shapes
from rasterio.plot import show
from rasterio.profiles import default_gtiff_profile
from rasterio.warp import reproject
from tqdm.auto import tqdm

from gch4i.config import (
    V3_DATA_PATH,
    figures_data_dir_path,
    global_data_dir_path,
    years,
)

# import warnings
Avogadro = 6.02214129 * 10 ** (23)  # molecules/mol
Molarch4 = 16.04  # CH4 molecular weight (g/mol)
tg_to_kt = 1000  # conversion factor, teragrams to kilotonnes
# tg_scale = (
#    0.001  # Tg conversion factor
# )

# global warming potential of CH4 relative to CO2
# (used to convert mass to CO2e units, from IPPC AR4)
GWP_CH4 = 25
# EEM: add constants (note, we should try to do conversions using variable names, so
#      that we don't have constants hard coded into the scripts)


# define a function to normalize the population data by state and year
def normalize(x):
    return x / x.sum() if x.sum() > 0 else 0


# NOTE: xarray does not like the if else of the default normalize method, so I have
# defined a new one here to appease xarray.
def normalize_xr(x):
    return x / x.sum()


def get_cell_gdf() -> gpd.GeoDataFrame:
    """create a geodata frame of cell polygons matching our output grid"""
    profile = GEPA_spatial_profile()
    # get the number of cells
    ncells = np.multiply(*profile.arr_shape)
    # create an empty array of the right shape, assign each cell a unique value
    tmp_arr = np.arange(ncells, dtype=np.int32).reshape(profile.arr_shape)

    # get the cells as individual items in a dictionary holding their id and geom
    results = [
        {"properties": {"raster_val": v}, "geometry": s}
        for i, (s, v) in enumerate(
            shapes(tmp_arr, transform=profile.profile["transform"])
        )
    ]

    # turn geom dictionary into a geodataframe
    cell_gdf = gpd.GeoDataFrame.from_features(results, crs=4326).drop(
        columns="raster_val"
    )
    return cell_gdf


def load_area_matrix(resolution=0.1) -> np.array:
    """load the raster array of grid cell area in square meters"""
    res_text = str(resolution).replace(".", "")
    input_path = global_data_dir_path / f"gridded_area_{res_text}_cm2.tif"
    with rasterio.open(input_path) as src:
        arr = src.read(1)
    return arr


def write_ncdf_output(
    in_da: xr.DataArray,
    dst_path: Path,
    description: str,
    title: str,
    units: str = "moleccm-2s-1",
    resolution: float = 0.1,
    # month_flag: bool = False,
) -> None:
    """take dict of year:array pairs and write to dst_path with attrs"""
    # year_list = list(raster_dict.keys())
    # array_stack = np.stack(list(raster_dict.values()))

    # TODO: update function for this
    # if month_flag:
    #     pass

    # TODO: accept different resolutions for this function
    # if resolution:
    #     pass

    gepa_profile = GEPA_spatial_profile(resolution)
    min_year = np.min(in_da.time.values)
    max_year = np.max(in_da.time.values)

    data_xr = (
        in_da.rename({"y": "lat", "x": "lon"})
        .assign_coords({"lat": gepa_profile.y, "lon": gepa_profile.x})
        .rio.set_attrs(
            {
                "title": title,
                "description": description,
                "year": f"{min_year}-{max_year}",
                "units": units,
            }
        )
        .rio.write_crs(gepa_profile.profile["crs"])
        .rio.write_transform(gepa_profile.profile["transform"])
        .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        .rio.write_nodata(0.0, encoded=True)
    )
    data_xr.to_netcdf(dst_path.with_suffix(".nc"))
    return None


def name_formatter(col: pd.Series) -> pd.Series:
    """standard name formatted to allow for matching between datasets

    casefold
    replace any repeated spaces with just one
    remove any non-alphanumeric chars

    input:
        col = pandas Series
    returns = pandas series
    """
    return (
        col.str.strip()
        .str.casefold()
        .str.replace("\s+", " ", regex=True)  # noqa w605
        .replace("[^a-zA-Z0-9 -]", "", regex=True)
    )


us_state_to_abbrev_dict = {
    "Alabama": "AL",
    "Alaska": "AK",
    "Arizona": "AZ",
    "Arkansas": "AR",
    "California": "CA",
    "Colorado": "CO",
    "Connecticut": "CT",
    "Delaware": "DE",
    "Florida": "FL",
    "Georgia": "GA",
    "Hawaii": "HI",
    "Idaho": "ID",
    "Illinois": "IL",
    "Indiana": "IN",
    "Iowa": "IA",
    "Kansas": "KS",
    "Kentucky": "KY",
    "Louisiana": "LA",
    "Maine": "ME",
    "Maryland": "MD",
    "Massachusetts": "MA",
    "Michigan": "MI",
    "Minnesota": "MN",
    "Mississippi": "MS",
    "Missouri": "MO",
    "Montana": "MT",
    "Nebraska": "NE",
    "Nevada": "NV",
    "New Hampshire": "NH",
    "New Jersey": "NJ",
    "New Mexico": "NM",
    "New York": "NY",
    "North Carolina": "NC",
    "North Dakota": "ND",
    "Ohio": "OH",
    "Oklahoma": "OK",
    "Oregon": "OR",
    "Pennsylvania": "PA",
    "Rhode Island": "RI",
    "South Carolina": "SC",
    "South Dakota": "SD",
    "Tennessee": "TN",
    "Texas": "TX",
    "Utah": "UT",
    "Vermont": "VT",
    "Virginia": "VA",
    "Washington": "WA",
    "West Virginia": "WV",
    "Wisconsin": "WI",
    "Wyoming": "WY",
    "District of Columbia": "DC",
    "American Samoa": "AS",
    "Guam": "GU",
    "Northern Mariana Islands": "MP",
    "Puerto Rico": "PR",
    "United States Minor Outlying Islands": "UM",
    "U.S. Virgin Islands": "VI",
}


def us_state_to_abbrev(state_name: str) -> str:
    """converts a full US state name to the two-letter abbreviation"""
    return (
        us_state_to_abbrev_dict[state_name]
        if state_name in us_state_to_abbrev_dict
        else state_name
    )


def download_url(url, output_path):
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
        # print("File downloaded successfully!")
    except requests.exceptions.RequestException as e:
        # print("Error downloading the file:", e)
        raise e


class GEPA_spatial_profile:
    lon_left = -130  # deg
    lon_right = -60  # deg
    lat_top = 55  # deg
    lat_bottom = 20  # deg
    # lon_left = -129.95  # deg
    # lon_right = -59.95  # deg
    # lat_top = 54.95  # deg
    # lat_bottom = 20.05  # deg
    valid_resolutions = [0.1, 0.01]

    def __init__(self, resolution: float = 0.1):
        self.resolution = resolution
        self.check_resolution(self.resolution)
        self.x = np.arange(self.lon_left, self.lon_right, self.resolution)
        self.y = np.arange(self.lat_top, self.lat_bottom, -self.resolution)
        self.height, self.width = self.arr_shape = (len(self.y), len(self.x))
        self.profile = self.get_profile(self.resolution)

    def check_resolution(self, resolution):
        if resolution not in self.valid_resolutions:
            raise ValueError(
                f"resolution must be one of {', '.join(self.valid_resolutions)}"
            )

    def get_profile(self, resolution):
        base_profile = default_gtiff_profile.copy()
        base_profile.update(
            transform=rasterio.Affine(
                resolution, 0.0, self.lon_left, 0.0, -resolution, self.lat_top
            ),
            height=self.height,
            width=self.width,
            crs=4326,
            dtype="float32",
        )
        return base_profile


def make_raster_binary(
    input_path: Path, output_path: Path, true_vals: np.array, num_workers: int = 1
):
    """take a raster file and convert it to a binary raster file based on the true_vals
    array"""
    with rasterio.open(input_path) as src:

        # Create a destination dataset based on source params. The
        # destination will be tiled, and we'll process the tiles
        # concurrently.
        profile = src.profile
        profile.update(nodata=-99, dtype="int8")

        with rasterio.open(output_path, "w", **profile) as dst:
            windows = [window for ij, window in dst.block_windows()]

            # We cannot write to the same file from multiple threads
            # without causing race conditions. To safely read/write
            # from multiple threads, we use a lock to protect the
            # DatasetReader/Writer
            read_lock = threading.Lock()
            write_lock = threading.Lock()

            def process(window):
                with read_lock:
                    src_array = src.read(window=window)

                # result = np.where((src_array >= 1) & (src_array <= 60), 1, 0)
                result = np.isin(src_array, true_vals)

                with write_lock:
                    dst.write(result, window=window)

            # We map the process() function over the list of
            # windows.
            with concurrent.futures.ThreadPoolExecutor(
                max_workers=num_workers
            ) as executor:
                executor.map(process, windows)


def mask_raster_parallel(
    input_path: Path, output_path: Path, mask_path: Path, num_workers: int = 1
):
    """take a raster file and convert it to a binary raster file based on the true_vals
    array

    alpha version of the function. Not fully tested yet. Use with caution.
    This does not check if the mask raster is the same size as the input raster. It
    assumes that the mask raster is the same size as the input raster. If not, you will
    need to run the warp function with the input_path as the target_path to get the mask
    aligned correctly. Otherwise this will throw and error.
    """
    with rasterio.open(input_path) as src:
        with rasterio.open(mask_path) as msk:

            # Create a destination dataset based on source params. The
            # destination will be tiled, and we'll process the tiles
            # concurrently.
            # profile = src.profile
            # profile.update(
            #     blockxsize=128, blockysize=128, tiled=True, nodata=0, dtype="float32"
            # )

            with rasterio.open(output_path, "w", **src.profile) as dst:
                windows = [window for ij, window in dst.block_windows()]

                # We cannot write to the same file from multiple threads
                # without causing race conditions. To safely read/write
                # from multiple threads, we use a lock to protect the
                # DatasetReader/Writer
                read_lock = threading.Lock()
                write_lock = threading.Lock()

                def process(window):
                    with read_lock:
                        src_array = src.read(window=window)
                        msk_array = msk.read(window=window)

                    # result = np.where((src_array >= 1) & (src_array <= 60), 1, 0)
                    result = np.where(msk_array == 1, src_array, 0)

                    with write_lock:
                        dst.write(result, window=window)

                # We map the process() function over the list of
                # windows.
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=num_workers
                ) as executor:
                    executor.map(process, windows)


# take any input raster file and warp it to match the GEPA_PROFILE
def warp_to_gepa_grid(
    input_path: Path,
    output_path: Path,
    target_path: Path = None,
    resampling: str = "average",
    num_threads: int = 1,
    resolution: float = 0.1,
):

    if target_path is None:
        # print("warping to GEPA grid")
        profile = GEPA_spatial_profile().profile
    else:
        # print("warping to other raster")
        with rasterio.open(target_path) as src:
            profile = src.profile

    try:
        resamp_method = getattr(Resampling, resampling)
    except AttributeError as ex:
        raise ValueError(
            f"resampling method {resampling} not found in rasterio.Resampling"
        ) from ex

    with rasterio.open(input_path) as src:
        profile.update(count=src.count, dtype="float32")
        with rasterio.open(output_path, "w", **profile) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=profile["transform"],
                    dst_crs=profile["crs"],
                    dst_nodata=0.0,
                    resampling=resamp_method,
                    num_threads=num_threads,
                )


# take any vector data input and grid it to the standard GEPA grid
def vector_to_gepa_grid():
    pass


def stack_rasters(input_paths: list[Path], output_path: Path):
    profile = GEPA_spatial_profile().profile
    raster_list = []
    years = [int(x.name.split("_")[2]) for x in input_paths]
    for in_file in input_paths:
        if not in_file.exists():
            continue
        with rasterio.open(in_file) as src:
            raster_list.append(src.read(1))

    output_data = np.stack(raster_list, axis=0)

    profile.update(count=len(raster_list))

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(output_data)
        dst.descriptions = tuple([str(x) for x in years])


def proxy_from_stack(
    input_path: Path,
    state_geo_path: Path,
    output_path: Path,
):

    # read in the state file and filter to lower 48 + DC
    state_gdf = (
        gpd.read_file(state_geo_path)
        .loc[:, ["NAME", "STATEFP", "STUSPS", "geometry"]]
        .rename(columns=str.lower)
        .rename(columns={"stusps": "state_code", "name": "state_name"})
        .astype({"statefp": int})
        # get only lower 48 + DC
        .query("(statefp < 60) & (statefp != 2) & (statefp != 15)")
        .to_crs(4326)
    )

    # read in the raw population data raster stack as a xarray dataset
    # masked will read the nodata value and set it to NA
    # xr_ds =xr.open_dataarray(input_path)
    xr_ds = rioxarray.open_rasterio(input_path, masked=True).rename({"band": "year"})

    with rasterio.open(input_path) as src:
        ras_crs = src.crs

    # assign the band as our years so the output raster data has year band names
    xr_ds["year"] = years
    # remove NA values
    # pop_ds = pop_ds.where(pop_ds != -99999)

    # create a state grid to match the input population array
    # we use fill here to fill in the nodata values with 99 so that when we do the
    # groupby, the nodata area is not collapsed, and the resulting dimensions align
    # with the v3 gridded data.
    state_grid = make_geocube(
        vector_data=state_gdf, measurements=["statefp"], like=xr_ds, fill=99
    )
    state_grid
    # assign the state grid as a new variable in the population dataset
    xr_ds["statefp"] = state_grid["statefp"]
    xr_ds

    # plot the data to check
    xr_ds["statefp"].plot()

    # apply the normalization function to the population data
    out_ds = (
        xr_ds.groupby(["year", "statefp"])
        .apply(normalize_xr)
        .sortby(["year", "y", "x"])
        .to_dataset(name="rel_emi")
    )
    out_ds["rel_emi"].shape

    # check that the normalization worked
    all_eq_df = (
        out_ds["rel_emi"]
        .groupby(["statefp", "year"])
        .sum()
        .rename("sum_check")
        .to_dataframe()
        .drop(columns="spatial_ref")
        .assign(
            # NOTE: Due to floating point rouding, we need to check if the sum is
            # close to 1, not exactly 1.
            is_close=lambda df: (np.isclose(df["sum_check"], 1))
            | (np.isclose(df["sum_check"], 0))
        )
    )

    vals_are_one = all_eq_df["is_close"].all()
    print(f"are all state/year norm sums equal to 1? {vals_are_one}")
    if not vals_are_one:
        raise ValueError("not all values are normed correctly!")

    # plot. Not hugely informative, but shows the data is there.
    out_ds["rel_emi"].sel(year=2020).plot.imshow()

    out_ds["rel_emi"].transpose("year", "y", "x").round(10).rio.write_crs(
        ras_crs
    ).to_netcdf(output_path)


def geocode_address(df, address_column):
    """
    Geocode addresses using Nominatim, handling missing values

    Parameters:
    df: DataFrame containing addresses
    address_column: Name of column containing addresses

    Returns:
    DataFrame with added latitude and longitude columns
    """
    # Check if address column exists
    if address_column not in df.columns:
        raise ValueError(f"Column {address_column} not found in DataFrame")

    # Initialize geocoder
    geolocator = Nominatim(user_agent="my_app")

    # Create cache dictionary
    geocode_cache = {}

    def get_lat_long(address):
        # Handle missing values
        if pd.isna(address) or str(address).strip() == "":
            return (None, None)

        # Check cache first
        if address in geocode_cache:
            return geocode_cache[address]

        try:
            # Add delay to respect Nominatim's usage policy
            time.sleep(1)
            location = geolocator.geocode(str(address))
            if location:
                result = (location.latitude, location.longitude)
                geocode_cache[address] = result
                return result
            return (None, None)

        except (GeocoderTimedOut, GeocoderServiceError):
            return (None, None)

    # Create lat_long column
    df["lat_long"] = None

    # Check if longitude column exists
    if "longitude" not in df.columns:
        df["longitude"] = None

    # Only geocode rows where both latitude and longitude are missing
    mask = (
        (df["longitude"].isna() | (df["longitude"] == ""))
        & (df["latitude"].isna() | (df["latitude"] == ""))
        & df[address_column].notna()
    )

    # Apply geocoding only to rows that need it
    df.loc[mask, "lat_long"] = df.loc[mask, address_column].apply(get_lat_long)

    # Update latitude/longitude only for rows that were geocoded
    df.loc[mask, "latitude"] = df.loc[mask, "lat_long"].apply(
        lambda x: x[0] if x else None
    )
    df.loc[mask, "longitude"] = df.loc[mask, "lat_long"].apply(
        lambda x: x[1] if x else None
    )

    # Drop temporary column
    df = df.drop("lat_long", axis=1)

    return df


def create_final_proxy_df(proxy_df):
    """
    Function to create the final proxy df that is ready for gridding

    Parameters:
    - proxy_df: DataFrame containing proxy data.

    Returns:
    - final_proxy_df: DataFrame containing the processed emissions data for each state.
    """

    # Create a GeoDataFrame and generate geometry from longitude and latitude
    proxy_gdf = gpd.GeoDataFrame(
        proxy_df,
        geometry=gpd.points_from_xy(
            proxy_df["longitude"], proxy_df["latitude"], crs="EPSG:4326"
        ),
    )

    # subset to only include the columns we want to keep
    proxy_gdf = proxy_gdf[["state_code", "year", "emis_kt", "geometry"]]

    # Normalize relative emissions to sum to 1 for each year and state
    proxy_gdf = proxy_gdf.groupby(["state_code", "year"]).filter(
        lambda x: x["emis_kt"].sum() > 0
    )  # drop state-years with 0 total volume
    proxy_gdf["rel_emi"] = proxy_gdf.groupby(["year", "state_code"])[
        "emis_kt"
    ].transform(
        lambda x: x / x.sum() if x.sum() > 0 else 0
    )  # normalize to sum to 1
    sums = proxy_gdf.groupby(["state_code", "year"])[
        "rel_emi"
    ].sum()  # get sums to check normalization
    # assert that the sums are close to 1
    assert np.isclose(
        sums, 1.0, atol=1e-8
    ).all(), f"Relative emissions do not sum to 1 for each year and state; {sums}"

    # Drop the original emissions column
    proxy_gdf = proxy_gdf.drop(columns=["emis_kt"])

    # Rearrange columns
    proxy_gdf = proxy_gdf[["state_code", "year", "rel_emi", "geometry"]]

    return proxy_gdf
