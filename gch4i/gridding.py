# a collection of functions that do standard gridding
from pyarrow import parquet  # noqa: F401
import osgeo  # noqa: F401
import dask_geopandas as dgpd
import geopandas as gpd
import pyogrio  # noqa: F401
import rasterio
from rasterio.profiles import default_gtiff_profile
from rasterio.features import shapes
import numpy as np

# specs for raster outputs. These are the default specs for which all input and output
# raster files should match.
res01 = 0.1  # deg
lon_left = -130  # deg
lon_right = -60  # deg
lat_up = 55  # deg
lat_low = 20  # deg
x = np.arange(lon_left, lon_right, res01)
y = np.arange(lat_low, lat_up, res01)
HEIGHT, WIDTH = ARR_SHAPE = (len(y), len(x))

GEPA_PROFILE = default_gtiff_profile.copy()
GEPA_PROFILE.update(
    transform=rasterio.Affine(res01, 0.0, lon_left, 0.0, -res01, lat_up),
    height=HEIGHT,
    width=WIDTH,
    crs=4326,
    dtype=np.float64,
)


# adapted from https://github.com/geopandas/dask-geopandas/issues/236
def overlay(left: dgpd.GeoDataFrame, right: dgpd.GeoDataFrame):
    left["left_geom"] = left.geometry
    right["right_geom"] = right.geometry
    overlap = left.sjoin(right)
    overlap.geometry = overlap.left_geom.intersection(overlap.right_geom)
    overlap = overlap.drop(columns=["left_geom", "right_geom"])
    return overlap


# take any input raster file and warp it to match the GEPA_PROFILE
def warp_to_gepa_grid():
    pass


# function to create an empty x/array of the standard GEPA grid, or fill it if desired
def create_empty_grid_gdf():
    # get the number of cells
    ncells = np.multiply(*ARR_SHAPE)
    # create an empty array of the right shape, assign each cell a unique value
    tmp_arr = np.arange(ncells, dtype=np.int32).reshape(ARR_SHAPE)
    # get the cells as individual items in a dictionary holding their id and geom
    results = [
        {"properties": {"raster_val": v}, "geometry": s}
        for i, (s, v) in enumerate(shapes(tmp_arr, transform=GEPA_PROFILE["transform"]))
    ]

    # turn geom dictionary into a geodataframe, reproject to equal area, calculate the
    # cell area in square meters
    empty_gdf = (
        gpd.GeoDataFrame.from_features(results, crs=4326)
        .to_crs("ESRI:102003")
    )

    return empty_gdf


# function to create an empty x/array of the standard GEPA grid, or fill it if desired
def create_empty_grid(output_path):
    # We have to resort the dataframe on the id value to get it in the right order for
    # turning into a matrix
    area_matrix = create_empty_grid_gdf().sort_values("raster_val", ascending=False)[
        "cell_area_sq_m"
    ].values.reshape(ARR_SHAPE)

    # get the GEPA profile, make the count 1
    dst_profile = GEPA_PROFILE.copy()
    dst_profile.update(count=1)

    # save the file for all other tasks to use
    with rasterio.open(output_path, "w", **dst_profile) as dst:
        dst.write(area_matrix, 1)


# function to take gridded summary of vector data in grid cells (length, area, or count)
def vector_to_gepa_grid(vector_data, grid_path):
    vector_gdf = (dgpd.read_parquet(vector_data, npartitions=30)
                  .to_crs("ESRI:102003")
                  .spatial_shuffle(shuffle="tasks"))
    grid_gdf = (dgpd.from_geopandas(create_empty_grid_gdf(), npartitions=30)
                .spatial_shuffle(shuffle="tasks"))
    overlap_gdf = overlay(grid_gdf, vector_gdf).compute()
    if vector_gdf.geom_type.str.contains("LineString").all().compute():
        overlap_col = "length_m"
        overlap_gdf[overlap_col] = overlap_gdf.geometry.length
    elif vector_gdf.geom_type.str.contains("Polygon").all().compute():
        overlap_col = "area_m2"
        overlap_gdf[overlap_col] = overlap_gdf.geometry.area
    elif vector_gdf.geom_type.str.contains("Point").all().compute():
        overlap_col = "count"
        overlap_gdf[overlap_col] = 1
    by_cell = overlap_gdf.dissolve(by="raster_val", aggfunc="sum")
    out_grid = (grid_gdf.merge(by_cell[[overlap_col]], on="raster_val", how="outer")
                .fillna(0)
                .set_geometry("geometry")
                .compute())
    out_grid.to_parquet(grid_path)
