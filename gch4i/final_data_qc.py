import pandas as pd
import xarray as xr
from pathlib import Path
import calendar
import rioxarray
import numpy as np
import warnings


from gch4i.config import (
    V3_DATA_PATH,
    final_gridded_dir,
    years,
)

from gch4i.utils import (
    Avogadro,
    GEPA_spatial_profile,
    Molarch4,
    get_cell_gdf,
    load_area_matrix,
    normalize,
)


# Days in year
def get_days_in_year(year):
    year = int(year)
    return 366 if calendar.isleap(year) else 365


# Mass to Flux Conversion Factor
def calc_conversion_factor(year_days: int, area_matrix):
    """calculate emissions in kt to flux (in units of molec. cm-2 s-1) """
    return (
        10**9 * Avogadro / float(Molarch4 * year_days * 24 * 60 * 60) / area_matrix
    )


# # Calculate the total national emissions by source and year
# for iyear in years:
#     iyear_days = get_days_in_year(iyear)  # number of days in the year
#     final_data_path: Path = final_gridded_dir / f"Gridded_GHGI_Methane_v3_{iyear}_AugTest.nc"  # path to final gridded data
#     ds = xr.open_dataset(final_data_path)  # final gridded netcdf file
#     area_matrix = ds['grid_cell_area']  # final area matrix within netcdf file
#     gridding_group_list = [item for item in list(xr.open_dataset(final_data_path).variables.keys()) if item.startswith('emi')]  # list of emi variables in netcdf file
#     group_flux_data = ds[gridding_group_list]  # flux data for all gridding groups
#     group_mass_sums = (group_flux_data / calc_conversion_factor(iyear_days, area_matrix)).sum()


class QCFinalNetCDFs:
    def __init__(self, group_name, final_gridded_dir):
        # the path to the directory where the final gridded data is saved
        self.group_name = group_name
        self.final_gridded_dir = final_gridded_dir
        self.flux_data_files = list(self.final_gridded_dir.glob("*AugTest.nc"))
        self.gepa_profile = GEPA_spatial_profile()


    def qc_national_mass_sums(self):
        v3_data_dict = {}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for out_path in self.flux_data_files:
                v3_year = int(out_path.stem.split("_")[-2])
                # v3_year = int(out_path.stem.split("_")[-1])
                final_group_name = 'emi_ch4_'+self.group_name
                v3_data = rioxarray.open_rasterio(out_path, variable=final_group_name)[
                    final_group_name
                ].values.squeeze(axis=0)
                self.v3_area_matrix = rioxarray.open_rasterio(out_path, variable='grid_cell_area')[
                    'grid_cell_area'
                ].values.squeeze(axis=0)
                v3_data_dict[v3_year] = v3_data

        v3_arr = np.array(list(v3_data_dict.values()))

        self.v3_flux_da = xr.DataArray(
            v3_arr,
            dims=["time", "y", "x"],
            coords=[
                list(v3_data_dict.keys()),
                self.gepa_profile.y,
                self.gepa_profile.x,
            ],
            name=final_group_name,
        )

        def calculate_mass(self, in_ds):
            """calculates mass for dictionary of total flux year/array pairs"""
            times = in_ds.time.values

            def get_days_in_year(year):
                year = int(year)
                return 366 if calendar.isleap(year) else 365
            
            def calc_conversion_factor(self, year_days: int, area_matrix: np.array) -> np.array:
                """calculate emissions in kt to flux (in units of molec. cm-2 s-1) """
                return (
                    10**9 * Avogadro / float(Molarch4 * year_days * 24 * 60 * 60) / area_matrix
                )

            days_in_year = [get_days_in_year(x) for x in times]
            conv_factors = [
                calc_conversion_factor(x, self.area_matrix) for x in days_in_year
            ]

            conv_ds = xr.DataArray(
                conv_factors,
                # np.flip(conv_factors, 1),
                dims=["time", "y", "x"],
                coords=[times, self.gepa_profile.y, self.gepa_profile.x],
                name="conversion_factor",
            )
            flux_out_da = in_ds / conv_ds
            return flux_out_da

        self.v3_mass_da = calculate_mass(self.v3_flux_da)
        v3_mass_yearly_sums = np.nansum(self.v3_mass_da.values, axis=(1, 2))
        
        mass_sums_df = pd.DataFrame(
                {
                    "year": list(v3_data_dict.keys()),
                    "v3_sum": v3_mass_yearly_sums,
                }
            ).assign(metric="mass")
        return mass_sums_df


# %%
file_checker = QCFinalNetCDFs()
file_checker.create_qc_outputs()
file_writer.plot_data()
