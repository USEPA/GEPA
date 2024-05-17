from pathlib import Path

import osgeo  # noqa f401
import numpy as np
import rasterio
import xarray as xr
import rioxarray  # noqd f401

from gch4i.config import global_data_dir_path
from gch4i.gridding import GEPA_PROFILE, x, y

Avogadro = 6.02214129 * 10 ** (23)  # molecules/mol
Molarch4 = 16.04  # CH4 molecular weight (g/mol)
Res01 = 0.1  # output resolution (degrees)
tg_to_kt = 1000  # conversion factor, teragrams to kilotonnes
# tg_scale = (
#    0.001  # Tg conversion factor
# )
GWP_CH4 = 25  # global warming potential of CH4 relative to CO2 (used to convert mass to CO2e units, from IPPC AR4)
tg_to_kt = 1000  # conversion factor, teragrams to kilotonnes


def calc_conversion_factor(days_in_year: int, cell_area_matrix: np.array) -> float:
    return (
        10**9
        * Avogadro
        / float(Molarch4 * days_in_year * 24 * 60 * 60)
        / cell_area_matrix
    )


def write_tif_output(in_dict: dict, dst_path: Path) -> None:
    """take an input dictionary with year/array items, write raster to dst_path"""
    out_array = np.stack(list(in_dict.values()))

    dst_profile = GEPA_PROFILE.copy()

    dst_profile.update(count=out_array.shape[0])
    with rasterio.open(dst_path, "w", **dst_profile) as dst:
        dst.write(out_array)
        dst.descriptions = [str(x) for x in in_dict.keys()]
    return None


def load_area_matrix() -> np.array:
    input_path = global_data_dir_path / "gridded_area_m2.tif"
    with rasterio.open(input_path) as src:
        arr = src.read(1)
    return arr


# %%
def write_ncdf_output(
    raster_dict: dict,
    dst_path: Path,
    title: str,
    description: str,
    # resolution: float,
    units: str = "moleccm-2s-1",
    # month_flag: bool = False,
) -> None:
    """take dict of year:array pairs and write to dst_path with attrs"""
    year_list = list(raster_dict.keys())
    array_stack = np.stack(list(raster_dict.values()))

    # TODO: update function for this
    # if month_flag:
    #     pass

    # TODO: accept different resolutions for this function
    # if resolution:
    #     pass

    data_xr = (
        xr.DataArray(
            array_stack,
            coords={
                "time": year_list,
                "lat": y,
                "lon": x,
            },
            dims=[
                "time",
                "lat",
                "lon",
            ],
        )
        .rio.set_attrs(
            {
                "title": title,
                "description": description,
                "year": f"{min(year_list)}-{max(year_list)}",
                "units": units,
            }
        )
        .rio.write_crs(GEPA_PROFILE["crs"])
        .rio.write_transform(GEPA_PROFILE["transform"])
        .rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        # I think this only write out the year names in a separate table. Doesn't
        # seem useful.
        # .rio.write_coordinate_system()
        .rio.write_nodata(0.0, encoded=True)
    )
    data_xr.to_netcdf(dst_path)
    return None


# %%
