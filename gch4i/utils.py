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

from gch4i.config import global_data_dir_path, figures_data_dir_path, V3_DATA_PATH, max_year, min_year, load_state_ansi, load_road_globals
from gch4i.gridding import GEPA_spatial_profile
from gch4i.config import RoadProxyGlobals

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
        # NOTE: the x and y are in the order of what xarray expects and we add 0.05 to
        # align these with how xarray lists coordinates as cell centers. This is
        # slightly different than GDAL, which will list top left corner coords.
        self.x = np.arange(self.lon_left, self.lon_right, self.resolution) + 0.05
        self.y = np.arange(self.lat_bottom, self.lat_top, self.resolution) + 0.05
        self.height, self.width = self.arr_shape = (len(self.y), len(self.x))
        self.get_profile()

    def check_resolution(self, resolution):
        if resolution not in self.valid_resolutions:
            raise ValueError(
                f"resolution must be one of {', '.join(self.valid_resolutions)}"
            )

    def get_profile(self):
        base_profile = default_gtiff_profile.copy()
        base_profile.update(
            transform=rasterio.Affine(
                self.resolution, 0.0, self.lon_left, 0.0, -self.resolution, self.lat_top
            ),
            height=self.height,
            width=self.width,
            crs=4326,
            dtype="float32",
        )
        self.profile = base_profile


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

def convert_FIPS_to_two_letter_code(df, state_column):
        """
        Convert numeric FIPS state values in a DataFrame column to two-letter state codes.

        Parameters:
        df (pd.DataFrame): The DataFrame containing the state names.
        state_column (str): The name of the column containing the FIPS numeric names.

        Returns:
        pd.DataFrame: DataFrame with the state column changed to two-letter state codes.
        """
        
        # Dictionary mapping full state names to their two-letter codes
        fips_state_abbr = {
        "1": "AL", "2": "AK", "4": "AZ", "5": "AR", "6": "CA",
        "8": "CO", "9": "CT", "10": "DE", "11": "DC", "12": "FL", 
        "13": "GA", "15": "HI", "16": "ID", "17": "IL", "18": "IN",
        "19": "IA", "20": "KS", "21": "KY", "22": "LA", "23": "ME", 
        "24": "MD", "25": "MA", "26": "MI", "27": "MN", "28": "MS",
        "29": "MO", "30": "MT", "31": "NE", "32": "NV", "33": "NH",
        "34": "NJ", "35": "NM", "36": "NY", "37": "NC", "38": "ND",
        "39": "OH", "40": "OK", "41": "OR", "42": "PA", "44": "RI", 
        "45": "SC", "46": "SD", "47": "TN", "48": "TX", "49": "UT",
        "50": "VT", "51": "VA", "53": "WA", "54": "WV", "55": "WI",
        "56": "WY"
        }

        # Map the full state names to their two-letter codes using the dictionary
        df['state_code'] = df[state_column].astype(int).astype(str).map(fips_state_abbr)
        
        return df


# ROAD PROXY FUNCTIONS
########################################################################################

# %% STEP 0.2 Load Path Files
raw_path, raw_roads_path, road_file, raw_road_file, task_outputs_path, global_path, gdf_state_files, global_input_path, state_ansi_path, GEPA_Comb_Mob_path, State_vmt_file,  State_vdf_file, State_ANSI, name_dict, state_mapping, start_year, end_year, year_range, year_range_str, num_years = load_road_globals()

def get_overlay_dir(year, 
                    out_dir: Path=task_outputs_path / 'overlay_cell_state_region'):
    return out_dir / f'cell_state_region_{year}.parquet'

def get_overlay_gdf(year, crs="EPSG:4326"):
    crs_obj = CRS(crs)

    gdf = gpd.read_parquet(get_overlay_dir(year))

    if gdf.crs != crs_obj:
        print(f"Converting overlay to crs {crs_obj.to_epsg()}")
        gdf.to_crs(crs, inplace=True)

    return gdf

# Read in State Spatial Data
def get_states_gdf(crs="EPSG:4326"):
    """
    Read in State spatial data
    """
    crs_obj = CRS(crs)

    gdf_states = gpd.read_file(gdf_state_files)


    gdf_states = gdf_states[~gdf_states['STUSPS'].isin(
        ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI']
        )]
    gdf_states = gdf_states[['STUSPS', 'NAME', 'geometry']]

    if gdf_states.crs != crs_obj:
        print(f"Converting states to crs {crs_obj.to_epsg()}")
        gdf_states.to_crs(crs, inplace=True)

    return gdf_states

# Read in Region Spatial Data
def get_region_gdf(year, crs=4326):
    """
    Read in region spatial data
    """
    crs = CRS(crs)
    road_loc = (
        gpd.read_parquet(f"{road_file}{year}_us_uac.parquet", columns=['geometry'])
        .assign(year=year)
        .to_crs(crs)
        .assign(urban=1)
    )
    return road_loc

def benchmark_load(func):
    def wrapper(*args, **kwargs):
        start = datetime.now()
        result = func(*args, **kwargs)
        print(f"{func.__name__} took {datetime.now() - start}")
        return result
    return wrapper

def read_reduce_data(year):
    """
    Read in All Roads data. Deduplicate and reduce data early.
    """
    # check if file already exists
    out_path = task_outputs_path / f"reduced_roads_{year}.parquet"
    if out_path.exists():
        print(f'Found reduced roads file at path {out_path}.')
        print()
        return out_path
    # Order road types
    road_type_order = ['Primary', 'Secondary', 'Other']

    df = (gpd.read_parquet(f"{raw_road_file}{year}_us_allroads.parquet",
                           columns=['MTFCC', 'geometry'])
          .assign(year=year))

    road_data = (
        df.to_crs("ESRI:102003")
        .assign(
            geometry=lambda df: df.normalize(),
            road_type=lambda df: pd.Categorical(
                np.select(
                    [
                        df['MTFCC'] == 'S1100',
                        df['MTFCC'] == 'S1200',
                        df['MTFCC'].isin(['S1400', 'S1630', 'S1640'])
                    ],
                    [
                        'Primary',
                        'Secondary',
                        'Other'
                    ],
                    default=None
                ),
                categories=road_type_order,  # Define the categories
                ordered=True  # Ensure the categories are ordered
            )
        )
    )
    # Sort
    road_data = road_data.sort_values('road_type').reset_index(drop=True)
    # Explode to make LineStrings
    road_data = road_data.explode(index_parts=True).reset_index(drop=True)
    # Remove duplicates of geometries
    road_data = road_data.drop_duplicates(subset='geometry', keep='first')

    # Separate out Road Types
    prim_year = road_data[road_data['road_type'] == 'Primary']
    sec_year = road_data[road_data['road_type'] == 'Secondary']
    oth_year = road_data[road_data['road_type'] == 'Other']

    buffer_distance = 3  # meters

    # Set buffers
    prim_buffer = prim_year.buffer(buffer_distance)
    prim_buffer = gpd.GeoDataFrame(geometry=prim_buffer, crs=road_data.crs)

    prisec_buffer = pd.concat([prim_year, sec_year], ignore_index=True)
    prisec_buffer = prisec_buffer.buffer(buffer_distance)
    prisec_buffer = gpd.GeoDataFrame(geometry=prisec_buffer, crs=road_data.crs)

    # Overlay
    sec_red = gpd.overlay(sec_year, prim_buffer, how='difference')
    other_red = gpd.overlay(oth_year, prisec_buffer, how='difference')

    # Combine
    road_data = pd.concat([prim_year, sec_red, other_red], ignore_index=True)

    road_data = road_data[['year', 'road_type', 'geometry']]

    # Write to parquet
    road_data.to_parquet()

    del road_data, prim_year, sec_year, oth_year, prim_buffer, prisec_buffer, sec_red, other_red

    gc.collect()

    return out_path


def get_roads_path(year, raw_roads_path: Path=V3_DATA_PATH / "global/raw_roads", raw=True):
    '''
    If raw is True, it returns the parquet files of the original roads data at at path:
        C:/Users/<USER>/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - RTI 2024 Task Order/Task 2/ghgi_v3_working/v3_data/global/raw_roads/tl_{year}_us_allroads.parquet
    If raw is False, it returns the parquet files of the reduced roads data (as generated in Andrew's original task_roads_proxy.py script `reduce_roads()`): 
        C:/Users/<USER>/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - RTI 2024 Task Order/Task 2/ghgi_v3_working/v3_data/global/raw_roads/task_outputs/reduced_roads_{year}.parquet
    
    '''
    if raw:
        return Path(raw_roads_path) / f"tl_{year}_us_allroads.parquet"
    else:
        return Path(raw_roads_path) / f"task_outputs/reduced_roads_{year}.parquet"


@benchmark_load
def read_roads(year, raw=True, crs='EPSG:4326'):
    crs_obj = CRS(crs)
    gdf = gpd.read_parquet(get_roads_path(year, raw=raw))
    if gdf.crs != crs_obj:
        print(f"Converting {year} roads to crs {crs_obj.to_epsg()}")
        return gdf.to_crs(crs)
    else:
        return gdf
    
def intersect_sindex(cell, roads):
    '''
    Based of fthis geoff boeing blog post:
    https://geoffboeing.com/2016/10/r-tree-spatial-index-python/
    '''
    # first, add rtree spatial index to roads

    
    spatial_index = roads.sindex
    possible_matches_index = list(spatial_index.intersection(cell.bounds))
    possible_matches = roads.iloc[possible_matches_index]
    precise_matches = possible_matches[possible_matches.intersects(cell)]
    return precise_matches

# @benchmark_load
def intersect_and_clip(roads, state_geom, state_abbrev=None, simplify=None, dask=False):
    '''
    The simplify parameer is used to simplify the geometry of the roads before clipping to the state boundary
    '''
    state_box = box(*state_geom.bounds)
    # Get the roads that intersect with the state bounding box
    state_roads = intersect_sindex(state_box, roads)
    # print(f"Found {len(state_roads)} roads in {state_abbrev}")
    if not state_roads.empty:
        if simplify:
            state_geom = state_geom.simplify(simplify)
        

        # clip roads to state boundary
        result = state_roads.clip(state_geom)
        return result
    return state_roads


# Read in VM2 Data
def get_vm2_arrays(num_years):
    # Initialize arrays
    Miles_road_primary = np.zeros([2, len(State_ANSI), num_years])
    Miles_road_secondary = np.zeros([2, len(State_ANSI), num_years])
    Miles_road_other = np.zeros([2, len(State_ANSI), num_years])
    total = np.zeros(num_years)
    total2 = np.zeros(num_years)

    headers = ['STATE', 'RURAL - INTERSTATE', 'RURAL - FREEWAYS', 'RURAL - PRINCIPAL',
               'RURAL - MINOR', 'RURAL - MAJOR COLLECTOR', 'RURAL - MINOR COLLECTOR',
               'RURAL - LOCAL', 'RURAL - TOTAL', 'URBAN - INTERSTATE',
               'URBAN - FREEWAYS', 'URBAN - PRINCIPAL', 'URBAN - MINOR',
               'URBAN - MAJOR COLLECTOR', 'URBAN - MINOR COLLECTOR', 'URBAN - LOCAL',
               'URBAN - TOTAL', 'TOTAL']

    for iyear in np.arange(num_years):
        VMT_road = pd.read_excel(State_vmt_file + year_range_str[iyear] + '.xls',
                                 sheet_name='A',
                                 skiprows=13,
                                 nrows=51)
        VMT_road.columns = headers

        VMT_road = (VMT_road
                    .assign(STATE=lambda x: x['STATE'].str.replace("(2)", ""))
                    .assign(STATE=lambda x: x['STATE'].str.replace("Dist. of Columbia",
                                                                   "District of Columbia" ""))
                    )

        for idx in np.arange(len(VMT_road)):
            VMT_road.loc[idx, 'ANSI'] = name_dict[VMT_road.loc[idx, 'STATE'].strip()]
            istate = np.where(VMT_road.loc[idx, 'ANSI'] == State_ANSI['ansi'])
            Miles_road_primary[0, istate, iyear] = VMT_road.loc[idx, 'URBAN - INTERSTATE']
            Miles_road_primary[1, istate, iyear] = VMT_road.loc[idx, 'RURAL - INTERSTATE']
            Miles_road_secondary[0, istate, iyear] = VMT_road.loc[idx, 'URBAN - FREEWAYS'] + \
                VMT_road.loc[idx, 'URBAN - PRINCIPAL'] + \
                VMT_road.loc[idx, 'URBAN - MINOR']
            Miles_road_secondary[1, istate, iyear] = VMT_road.loc[idx, 'RURAL - FREEWAYS'] + \
                VMT_road.loc[idx, 'RURAL - PRINCIPAL'] + \
                VMT_road.loc[idx, 'RURAL - MINOR']
            Miles_road_other[0, istate, iyear] = VMT_road.loc[idx, 'URBAN - MAJOR COLLECTOR'] + \
                VMT_road.loc[idx, 'URBAN - MINOR COLLECTOR'] + \
                VMT_road.loc[idx, 'URBAN - LOCAL']
            Miles_road_other[1, istate, iyear] = VMT_road.loc[idx, 'RURAL - MAJOR COLLECTOR'] + \
                VMT_road.loc[idx, 'RURAL - MINOR COLLECTOR'] + \
                VMT_road.loc[idx, 'RURAL - LOCAL']
            total[iyear] += np.sum(Miles_road_primary[:, istate, iyear]) + \
                np.sum(Miles_road_secondary[:, istate, iyear]) + \
                np.sum(Miles_road_other[:, istate, iyear])
            total2[iyear] += VMT_road.loc[idx, 'TOTAL']

        abs_diff = abs(total[iyear] - total2[iyear])/((total[iyear]+total2[iyear])/2)

        if abs(abs_diff) < 0.0001:
            print('Year ' + year_range_str[iyear] + ': Difference < 0.01%: PASS')
            print(total[iyear])
            print(total2[iyear])
        else:
            print('Year ' + year_range_str[iyear] + ': Difference > 0.01%: FAIL, diff: ' + str(abs_diff))
            print(total[iyear])
            print(total2[iyear])

    return Miles_road_primary, Miles_road_secondary, Miles_road_other, total, total2


# Read in VM4 Data
def get_vm4_arrays(num_years):
    # Initialize arrays
    Per_vmt_mot = np.zeros([2, 3, len(State_ANSI), num_years])
    Per_vmt_pas = np.zeros([2, 3, len(State_ANSI), num_years])
    Per_vmt_lig = np.zeros([2, 3, len(State_ANSI), num_years])
    Per_vmt_hea = np.zeros([2, 3, len(State_ANSI), num_years])
    total_R = np.zeros(num_years)
    total_U = np.zeros(num_years)
    total = np.zeros(num_years)
    total2_U = np.zeros(num_years)
    total2 = np.zeros(num_years)
    total2_R = np.zeros(num_years)

    for iyear in np.arange(0, num_years):
        if year_range[iyear] == 2012 or year_range[iyear] == 2016:
            continue  # deal with missing data at the end
        else:
            # Read in Rural Sheet
            names = pd.read_excel(State_vdf_file + year_range_str[iyear]+'.xls',
                                  sheet_name='A', skiprows=12, header=0, nrows=1)
            colnames = names.columns.values
            VMT_type_R = pd.read_excel(State_vdf_file + year_range_str[iyear]+'.xls',
                                       na_values=['-'], sheet_name='A', names=colnames,
                                       skiprows=13, nrows=51)

            VMT_type_R.rename(columns={'MOTOR-': 'INTERSTATE - MOTORCYCLES',
                                       'PASSENGER': 'INTERSTATE - PASSENGER CARS',
                                       'LIGHT': 'INTERSTATE - LIGHT TRUCKS',
                                       'Unnamed: 4': 'INTERSTATE - BUSES',
                                       'SINGLE-UNIT': 'INTERSTATE - SINGLE-UNIT TRUCKS',
                                       'COMBINATION': 'INTERSTATE - COMBINATION TRUCKS',
                                       'Unnamed: 7': 'INTERSTATE - TOTAL',
                                       'MOTOR-.1': 'ARTERIALS - MOTORCYCLES',
                                       'PASSENGER.1': 'ARTERIALS - PASSENGER CARS',
                                       'LIGHT.1': 'ARTERIALS - LIGHT TRUCKS',
                                       'Unnamed: 11': 'ARTERIALS - BUSES',
                                       'SINGLE-UNIT.1': 'ARTERIALS - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.1': 'ARTERIALS - COMBINATION TRUCKS',
                                       'Unnamed: 14': 'ARTERIALS - TOTAL',
                                       'MOTOR-.2': 'OTHER - MOTORCYCLES',
                                       'PASSENGER.2': 'OTHER - PASSENGER CARS',
                                       'LIGHT.2': 'OTHER - LIGHT TRUCKS',
                                       'Unnamed: 18': 'OTHER - BUSES',
                                       'SINGLE-UNIT.2': 'OTHER - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.2': 'OTHER - COMBINATION TRUCKS',
                                       'Unnamed: 21': 'OTHER - TOTAL'}, inplace=True)

            VMT_type_R = (
                VMT_type_R
                .assign(STATE=lambda x: x['STATE'].str.replace("(2)", ""))
                .assign(STATE=lambda x: x['STATE'].str.replace("Dist. of Columbia",
                                                               "District of Columbia"))
                .assign(ANSI=0)
                .fillna(0)
                )

            # Read in Urban Sheet
            names = pd.read_excel(State_vdf_file + year_range_str[iyear] + '.xls',
                                  sheet_name='B', skiprows=12, header=0, nrows=1)
            colnames = names.columns.values
            VMT_type_U = pd.read_excel(State_vdf_file + year_range_str[iyear] + '.xls',
                                       na_values=['-'], sheet_name='B', names=colnames,
                                       skiprows=13, nrows=51)

            VMT_type_U.rename(columns={'MOTOR-': 'INTERSTATE - MOTORCYCLES',
                                       'PASSENGER': 'INTERSTATE - PASSENGER CARS',
                                       'LIGHT': 'INTERSTATE - LIGHT TRUCKS',
                                       'Unnamed: 4': 'INTERSTATE - BUSES',
                                       'SINGLE-UNIT': 'INTERSTATE - SINGLE-UNIT TRUCKS',
                                       'COMBINATION': 'INTERSTATE - COMBINATION TRUCKS',
                                       'Unnamed: 7': 'INTERSTATE - TOTAL',
                                       'MOTOR-.1': 'ARTERIALS - MOTORCYCLES',
                                       'PASSENGER.1': 'ARTERIALS - PASSENGER CARS',
                                       'LIGHT.1': 'ARTERIALS - LIGHT TRUCKS',
                                       'Unnamed: 11': 'ARTERIALS - BUSES',
                                       'SINGLE-UNIT.1': 'ARTERIALS - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.1': 'ARTERIALS - COMBINATION TRUCKS',
                                       'Unnamed: 14': 'ARTERIALS - TOTAL',
                                       'MOTOR-.2': 'OTHER - MOTORCYCLES',
                                       'PASSENGER.2': 'OTHER - PASSENGER CARS',
                                       'LIGHT.2': 'OTHER - LIGHT TRUCKS',
                                       'Unnamed: 18': 'OTHER - BUSES',
                                       'SINGLE-UNIT.2': 'OTHER - SINGLE-UNIT TRUCKS',
                                       'COMBINATION.2': 'OTHER - COMBINATION TRUCKS',
                                       'Unnamed: 21': 'OTHER - TOTAL'}, inplace=True)

            VMT_type_U = (
                VMT_type_U
                .assign(STATE=lambda x: x['STATE'].str.replace("(2)", ""))
                .assign(STATE=lambda x: x['STATE'].str.replace("Dist. of Columbia",
                                                               "District of Columbia"))
                .assign(ANSI=0)
                .fillna(0)
                )

            # Distribute to 4 output types: passenger, light, heavy, motorcycle
            for idx in np.arange(len(VMT_type_R)):
                VMT_type_R.loc[idx, 'ANSI'] = name_dict[VMT_type_R.loc[idx, 'STATE']
                                                        .strip()]
                istate_R = np.where(VMT_type_R.loc[idx, 'ANSI'] == State_ANSI['ansi'])
                Per_vmt_mot[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - MOTORCYCLES']
                Per_vmt_mot[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - MOTORCYCLES']
                Per_vmt_mot[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - MOTORCYCLES']
                Per_vmt_pas[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - PASSENGER CARS']
                Per_vmt_pas[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - PASSENGER CARS']
                Per_vmt_pas[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - PASSENGER CARS']
                Per_vmt_lig[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - LIGHT TRUCKS']
                Per_vmt_lig[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - LIGHT TRUCKS']
                Per_vmt_lig[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - LIGHT TRUCKS']
                Per_vmt_hea[1, 0, istate_R, iyear] = VMT_type_R.loc[idx, 'INTERSTATE - BUSES'] + \
                    VMT_type_R.loc[idx, 'INTERSTATE - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_R.loc[idx, 'INTERSTATE - COMBINATION TRUCKS']
                Per_vmt_hea[1, 1, istate_R, iyear] = VMT_type_R.loc[idx, 'ARTERIALS - BUSES'] + \
                    VMT_type_R.loc[idx, 'ARTERIALS - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_R.loc[idx, 'ARTERIALS - COMBINATION TRUCKS']
                Per_vmt_hea[1, 2, istate_R, iyear] = VMT_type_R.loc[idx, 'OTHER - BUSES'] + \
                    VMT_type_R.loc[idx, 'OTHER - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_R.loc[idx, 'OTHER - COMBINATION TRUCKS']
                total_R[iyear] += np.sum(Per_vmt_mot[1, :, istate_R, iyear]) + \
                    np.sum(Per_vmt_pas[1, :, istate_R, iyear]) + \
                    np.sum(Per_vmt_lig[1, :, istate_R, iyear]) + \
                    np.sum(Per_vmt_hea[1, :, istate_R, iyear])
                total2_R[iyear] += VMT_type_R.loc[idx, 'INTERSTATE - TOTAL'] + \
                    VMT_type_R.loc[idx, 'ARTERIALS - TOTAL'] + \
                    VMT_type_R.loc[idx, 'OTHER - TOTAL']

            for idx in np.arange(len(VMT_type_U)):
                VMT_type_U.loc[idx, 'ANSI'] = name_dict[VMT_type_U.loc[idx, 'STATE']
                                                        .strip()]
                istate_U = np.where(VMT_type_U.loc[idx, 'ANSI'] == State_ANSI['ansi'])
                Per_vmt_mot[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - MOTORCYCLES']
                Per_vmt_mot[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - MOTORCYCLES']
                Per_vmt_mot[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - MOTORCYCLES']
                Per_vmt_pas[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - PASSENGER CARS']
                Per_vmt_pas[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - PASSENGER CARS']
                Per_vmt_pas[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - PASSENGER CARS']
                Per_vmt_lig[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - LIGHT TRUCKS']
                Per_vmt_lig[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - LIGHT TRUCKS']
                Per_vmt_lig[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - LIGHT TRUCKS']
                Per_vmt_hea[0, 0, istate_U, iyear] = VMT_type_U.loc[idx, 'INTERSTATE - BUSES'] + \
                    VMT_type_U.loc[idx, 'INTERSTATE - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_U.loc[idx, 'INTERSTATE - COMBINATION TRUCKS']
                Per_vmt_hea[0, 1, istate_U, iyear] = VMT_type_U.loc[idx, 'ARTERIALS - BUSES'] + \
                    VMT_type_U.loc[idx, 'ARTERIALS - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_U.loc[idx, 'ARTERIALS - COMBINATION TRUCKS']
                Per_vmt_hea[0, 2, istate_U, iyear] = VMT_type_U.loc[idx, 'OTHER - BUSES'] + \
                    VMT_type_U.loc[idx, 'OTHER - SINGLE-UNIT TRUCKS'] + \
                    VMT_type_U.loc[idx, 'OTHER - COMBINATION TRUCKS']
                total_U[iyear] += np.sum(Per_vmt_mot[0, :, istate_U, iyear]) + \
                    np.sum(Per_vmt_pas[0, :, istate_U, iyear]) + \
                    np.sum(Per_vmt_lig[0, :, istate_U, iyear]) + \
                    np.sum(Per_vmt_hea[0, :, istate_U, iyear])
                total2_U[iyear] += VMT_type_U.loc[idx, 'INTERSTATE - TOTAL'] + \
                    VMT_type_U.loc[idx, 'ARTERIALS - TOTAL'] + \
                    VMT_type_U.loc[idx, 'OTHER - TOTAL']

            # Check for differences
            total[iyear] = total_U[iyear] + total_R[iyear]
            total2[iyear] = total2_R[iyear] + total2_U[iyear]
            abs_diff1 = abs(total[iyear] - total2[iyear]) / ((total[iyear] + total2[iyear]) / 2)

            if abs(abs_diff1) < 0.0001:
                print('Year ' + year_range_str[iyear] + ': Urban Difference < 0.01%: PASS')
            else:
                print('Year ' + year_range_str[iyear] + ': Urban Difference > 0.01%: FAIL, diff: ' + str(abs_diff1))
                print(total[iyear])
                print(total2[iyear])

    # Correct Years (assign 2012 to 2013), assign 2016 as average of 2015 and 2017
    idx_2012 = (2012-start_year)
    idx_2016 = (2016-start_year)
    Per_vmt_mot[:, :, :, idx_2012] = Per_vmt_mot[:, :, :, idx_2012 + 1]
    Per_vmt_pas[:, :, :, idx_2012] = Per_vmt_pas[:, :, :, idx_2012 + 1]
    Per_vmt_lig[:, :, :, idx_2012] = Per_vmt_lig[:, :, :, idx_2012 + 1]
    Per_vmt_hea[:, :, :, idx_2012] = Per_vmt_hea[:, :, :, idx_2012 + 1]

    Per_vmt_mot[:, :, :, idx_2016] = 0.5 * (Per_vmt_mot[:, :, :, idx_2016 - 1] +
                                            Per_vmt_mot[:, :, :, idx_2016 + 1])
    Per_vmt_pas[:, :, :, idx_2016] = 0.5 * (Per_vmt_pas[:, :, :, idx_2016 - 1] +
                                            Per_vmt_pas[:, :, :, idx_2016 + 1])
    Per_vmt_lig[:, :, :, idx_2016] = 0.5 * (Per_vmt_lig[:, :, :, idx_2016 - 1] +
                                            Per_vmt_lig[:, :, :, idx_2016 + 1])
    Per_vmt_hea[:, :, :, idx_2016] = 0.5 * (Per_vmt_hea[:, :, :, idx_2016 - 1] +
                                            Per_vmt_hea[:, :, :, idx_2016 + 1])

    # Optional: Combine Per_vmt_mot and Per_vmt_pas as Per_vmt_pas
    # Consult with EPA to determine whether to keep change
    Per_vmt_pas = Per_vmt_pas + Per_vmt_mot

    # Multiply by 0.01 to convert to percentage
    Per_vmt_mot = Per_vmt_mot * 0.01
    Per_vmt_pas = Per_vmt_pas * 0.01
    Per_vmt_lig = Per_vmt_lig * 0.01
    Per_vmt_hea = Per_vmt_hea * 0.01

    # Keep Per_vmt_mot reported for now, but it is now accounted for in Per_vmt_pas
    return Per_vmt_mot, Per_vmt_pas, Per_vmt_lig, Per_vmt_hea


# Calculate State Level Proxies
def calculate_state_proxies(num_years,
                            Miles_road_primary,
                            Miles_road_secondary,
                            Miles_road_other,
                            Per_vmt_pas,
                            Per_vmt_lig,
                            Per_vmt_hea):
    """
    array dimensions:
    region(urban/rural), road type(primary, secondary, other), state, year

    Example: vmt_pas : Miles_road_(primary, secondary, other) * Per_vmt_pas
    """

    # Initialize vmt_arrays
    vmt_pas = np.zeros([2, 3, len(State_ANSI), num_years])
    vmt_lig = np.zeros([2, 3, len(State_ANSI), num_years])
    vmt_hea = np.zeros([2, 3, len(State_ANSI), num_years])
    vmt_tot = np.zeros([2, len(State_ANSI), num_years])

    # Caclulate absolute number of VMT by region, road type, vehicle type, state, year
    # e.g. vmt_pas = VMT for passenger vehicles with dimensions = region (urban/rural),
    # road type (primary, secondary, other), state, and year

    # vmt_tot = region x state, year
    # road mile variable dimensions (urban/rural, state, year)

    for iyear in np.arange(0, num_years):
        vmt_pas[:, 0, :, iyear] = Miles_road_primary[:, :, iyear] * \
            Per_vmt_pas[:, 0, :, iyear]
        vmt_pas[:, 1, :, iyear] = Miles_road_secondary[:, :, iyear] * \
            Per_vmt_pas[:, 1, :, iyear]
        vmt_pas[:, 2, :, iyear] = Miles_road_other[:, :, iyear] * \
            Per_vmt_pas[:, 2, :, iyear]

        vmt_lig[:, 0, :, iyear] = Miles_road_primary[:, :, iyear] * \
            Per_vmt_lig[:, 0, :, iyear]
        vmt_lig[:, 1, :, iyear] = Miles_road_secondary[:, :, iyear] * \
            Per_vmt_lig[:, 1, :, iyear]
        vmt_lig[:, 2, :, iyear] = Miles_road_other[:, :, iyear] * \
            Per_vmt_lig[:, 2, :, iyear]

        vmt_hea[:, 0, :, iyear] = Miles_road_primary[:, :, iyear] * \
            Per_vmt_hea[:, 0, :, iyear]
        vmt_hea[:, 1, :, iyear] = Miles_road_secondary[:, :, iyear] * \
            Per_vmt_hea[:, 1, :, iyear]
        vmt_hea[:, 2, :, iyear] = Miles_road_other[:, :, iyear] * \
            Per_vmt_hea[:, 2, :, iyear]

        vmt_tot[:, :, iyear] += Miles_road_primary[:, :, iyear] + \
            Miles_road_secondary[:, :, iyear] + \
            Miles_road_other[:, :, iyear]

    # Initialize denominators
    tot_pas = np.zeros([len(State_ANSI), num_years])
    tot_lig = np.zeros([len(State_ANSI), num_years])
    tot_hea = np.zeros([len(State_ANSI), num_years])

    # Calculate total VMT for state/year by vehicle
    for istate in np.arange(0, len(name_dict)):
        for iyear in np.arange(0, num_years):
            tot_pas[istate, iyear] = np.sum(vmt_pas[:, :, istate, iyear])
            tot_lig[istate, iyear] = np.sum(vmt_lig[:, :, istate, iyear])
            tot_hea[istate, iyear] = np.sum(vmt_hea[:, :, istate, iyear])

    # Calculate proxy values
    print(f"CALCULATING")
    pas_proxy = vmt_pas / tot_pas
    lig_proxy = vmt_lig / tot_lig
    hea_proxy = vmt_hea / tot_hea

    return pas_proxy, lig_proxy, hea_proxy, vmt_tot


# Unpack State Proxy Arrays
def unpack_state_proxy(state_proxy_array, proxy_name='Emission Allocation'):
    reshaped_state_proxy = state_proxy_array.reshape(-1)

    row_index = np.repeat(['urban', 'rural'], 3 * 57 * num_years)
    col1_index = np.tile(np.repeat(['Primary', 'Secondary', 'Other'], 57 * num_years), 2)
    col2_index = np.tile(np.repeat(np.arange(1, 58), num_years), 2 * 3)
    col3_index = np.tile(np.arange(min_year, max_year + 1), 2 * 3 * 57)

    df = pd.DataFrame({
        'Region': row_index,
        'Road Type': col1_index,
        'State': col2_index,
        'Year': col3_index,
        proxy_name: reshaped_state_proxy
    })

    df['State_abbr'] = df['State'].map(state_mapping)

    cols = df.columns.tolist()
    state_index = cols.index('State')
    cols.insert(state_index + 1, cols.pop(cols.index('State_abbr')))
    df = df[cols]

    return df


# Unpack State Total Proxy Arrays
def unpack_state_allroads_proxy(vmt_tot):
    reshaped_state_proxy = vmt_tot.reshape(-1)

    row_index = np.repeat(['urban', 'rural'], 57 * num_years)
    col1_index = np.tile(np.repeat(np.arange(1, 58), num_years), 2)
    col2_index = np.tile(np.arange(min_year, max_year + 1), 2 * 57)

    df = pd.DataFrame({
        'Region': row_index,
        'State': col1_index,
        'Year': col2_index,
        'Proxy': reshaped_state_proxy
    })

    df['State_abbr'] = df['State'].map(state_mapping)

    cols = df.columns.tolist()
    state_index = cols.index('State')
    cols.insert(state_index + 1, cols.pop(cols.index('State_abbr')))
    df = df[cols]

    return df


# Generate Roads Proportions Data
def get_roads_proportion_data(pas_proxy, lig_proxy, hea_proxy, out_path, proxy_name='Emission Allocation'):
    """
    Formats data for roads proxy emissions
    """
    proxy_name = proxy_name.lower()
    # Add Vehicle Type column
    pas_proxy['Vehicle'] = 'Passenger'
    lig_proxy['Vehicle'] = 'Light'
    hea_proxy['Vehicle'] = 'Heavy'

    # Combine DataFrames
    vmt_roads_proxy = pd.concat([pas_proxy,
                                 lig_proxy,
                                 hea_proxy], axis=0).reset_index(drop=True)
    vmt_roads_proxy = (
        vmt_roads_proxy.rename(columns={'State_abbr': 'state_code',
                                        'Road Type': 'road_type'})
                       .rename(columns=lambda x: str(x).lower())
                       .query("state_code not in ['VI', 'MP', 'GU', 'AS', 'PR', 'AK', 'HI', 'UM']")
    )
    vmt_roads_proxy = vmt_roads_proxy[['state_code', 'year', 'vehicle', 'region',
                                       'road_type', proxy_name]]

    vmt_roads_proxy.to_csv(out_path, index=False)

    del pas_proxy, lig_proxy, hea_proxy

    gc.collect()

    return None


def get_road_proxy_data(road_proxy_out_path: Path=task_outputs_path / "roads_proportion_data.csv"):
    if not road_proxy_out_path.exists():
        # Proportional Allocation of Roads Emissions
        # The "Proxy" column in the output data represents the proportion of VMT by vehicle type for each unique combination of urban/rural and road type
        # The following code on the output df results in all 1's:
        # road_proxy_df.groupby(['vehicle', 'state_code', 'year']).Proxy.sum()

        # What this means is that each vehicle type's emissions would need to be allocated at the state/year level
        # What I don't understand is why not use miles travelled if that's the case?
        # The road types from this  output are: Primary, Secondary, and Other
        # These align with TIGER roads classifications

        # Ultimately, I join on two datasets on road type, along with state, year, and region. But this isn't usesful for the proxy value because it's a proportion of VMT across region/road type 

        #################
        # VM2 Outputs
        Miles_road_primary, Miles_road_secondary, Miles_road_other, total, total2 = get_vm2_arrays(num_years)

        # VM4 Outputs
        Per_vmt_mot, Per_vmt_pas, Per_vmt_lig, Per_vmt_hea = get_vm4_arrays(num_years)

        # State Proxy Outputs
        pas_proxy, lig_proxy, hea_proxy, vmt_tot = calculate_state_proxies(num_years,
                                                                        Miles_road_primary,
                                                                        Miles_road_secondary,
                                                                        Miles_road_other,
                                                                        Per_vmt_pas,
                                                                        Per_vmt_lig,
                                                                        Per_vmt_hea)

        # Unpack State Proxy Outputs
        pas_proxy = unpack_state_proxy(pas_proxy, num_years)
        lig_proxy = unpack_state_proxy(lig_proxy, num_years)
        hea_proxy = unpack_state_proxy(hea_proxy, num_years)
        # tot_proxy = unpack_state_allroads_proxy(vmt_tot)

        # Generate Roads Proportions Data
        get_roads_proportion_data(pas_proxy, lig_proxy, hea_proxy, out_path=road_proxy_out_path)

        return pas_proxy, lig_proxy, hea_proxy
    
    return pd.read_csv(road_proxy_out_path)


@benchmark_load
def overlay_cell_state_region(cell_gdf, region_gdf, state_gdf):
    '''
    This function overlays the cell grid with the state and region boundaries for each year of region data
    '''    
    # Overlay the cell grid with the state and region boundaries
    print(f"Overlaying cell grid with state and region boundaries: {datetime.now()}")
    cell_state_region_gdf = gpd.overlay(cell_gdf, state_gdf, how='union')
    print(f"Overlayed cell grid with state boundaries: {datetime.now()}")
    cell_state_region_gdf = gpd.overlay(cell_state_region_gdf, region_gdf, how='union')
    
    # where urban is NaN, set to 0 (since the "region" dataset is urban geometries) and drop variable for year (since it is redundant and in the file name)
    cell_state_region_gdf['urban'] = cell_state_region_gdf['urban'].fillna(0).astype(int)
    cell_state_region_gdf.drop(columns=['year'], inplace=True)

    # drop rows with no cell_id, as this indicates that the geometry falls outside of the US and so doesn't need to be processed
    cell_state_region_gdf.dropna(subset=['cell_id'], inplace=True)

    return cell_state_region_gdf

@benchmark_load
def run_overlay_for_year(year, out_dir):
    # Save the overlaid geoparquet file
    out_path = get_overlay_dir(year, out_dir)
    if out_path.exists():
        print(f"File already exists for {year}: {out_path}")
        return None
    
    # Load the region, cell and state data
    print(f"Reading datasets for {year}: {datetime.now()}")
    cell_gdf = get_cell_gdf().to_crs(4326).reset_index().rename(columns={'index': 'cell_id'})
    region_gdf = get_region_gdf(year)
    state_gdf = get_states_gdf()

    # Overlay the cell grid with the state and region boundaries
    cell_state_region_gdf = overlay_cell_state_region(cell_gdf, region_gdf, state_gdf)

    cell_state_region_gdf.to_parquet(out_path)

    print(f"Saved overlaid geoparquet file for {year} to {out_dir / f'cell_state_region_{year}.parquet'}")