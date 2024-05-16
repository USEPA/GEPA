import numpy as np
from gch4i.gridding import GEPA_PROFILE
from gch4i.config import global_data_dir_path
import osgeo
import rasterio
from pathlib import Path


Avogadro = 6.02214129 * 10 ** (23)  # molecules/mol
Molarch4 = 16.04  # g/mol
Res01 = 0.1  # degrees
tg_scale = (
    0.001  # Tg scale number [New file allows for the exclusion of the territories]
)
GWP_CH4 = 25  # global warming potential of CH4 relative to CO2 (used to convert mass to CO2e units)

# TODO finish this and move to utils
def calc_conversion_factor(days_in_year: int, cell_area_matrix: np.array):
     return (
            10**9 * Avogadro / float(Molarch4 * days_in_year * 24 * 60 * 60) / cell_area_matrix
        )


def write_outputs(in_dict: dict, dst_path: Path) -> None:
    """take an input dictionary with year/array items, write raster to dst_path"""
    out_array = np.stack(list(in_dict.values()))

    dst_profile = GEPA_PROFILE.copy()

    dst_profile.update(count=out_array.shape[0])
    with rasterio.open(dst_path, "w", **dst_profile) as dst:
        dst.write(out_array)
        dst.descriptions = [str(x) for x in in_dict.keys()]


def load_area_matrix():
    input_path = global_data_dir_path / "gridded_area_m2.tif"
    with rasterio.open(input_path) as src:
        arr = src.read(1)
    return arr