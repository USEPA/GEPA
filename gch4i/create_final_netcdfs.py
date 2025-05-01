# %%
# %load_ext autoreload
# %autoreload 2
# %%
import re

import pandas as pd
import xarray as xr

from gch4i.config import final_gridded_dir, prelim_gridded_dir
from gch4i.config import years as YEARS


class CreateFinalNetCDFs:
    def __init__(self):
        # the path to the directory where the group gridded data is stored
        self.input_dir = prelim_gridded_dir
        # the path to the directory where the final gridded data will be saved
        self.output_dir = final_gridded_dir
        # the path to the directory where the monthly scaling factors are stored
        self.monthly_scaling_dir = prelim_gridded_dir / "monthly_scaling"
        # flux units
        self.units = "moleccm-2s-1"
        # attributes written to the final files
        self.attrs = {
            "title": "Gridded U.S. Greenhouse Gas Inventory (Version 3): Annual methane emissions",
            "publication": "A gridded inventory of annual 2012-2022 U.S. anthropogenic methane emissions",
            "authors": "Joannes D. Maasakkers, Erin E. McDuffie, Julie Powers, Shane Cofffield, Yasmine Farhat, Nicholas Kruskamp, Hannah Lohman, Melissa P. Sulprizio, Candice Chen, Maggie Schultz, Lily Brunelle, Ryan Thrush, John Steller, Christopher Sherry, Daniel J. Jacob, Seongeun Jeong, Bill Irving, and Melissa Weitz",
            "history": "May 1, 2025",
            "conventions": "COARDS",
            "version": "1.0 - Publication version (Data equals the preprint version)",
            "contact": "Powers.Julie@epa.gov and McDuffie.Erin.E@epa.gov",
            # NOTE: this is a placeholder for the year of the data
            # the expectation is that year will be replaced with the year of the data
            # during processing
            "year": "xxxx",
        }
        self._get_month_scale_attrs()
        self.flux_data_files = list(self.input_dir.glob("*ch4_emi_flux.tif"))
        self.monthly_scale_files = list(
            self.monthly_scaling_dir.glob("*_monthly_scaling.tif")
        )

    def _get_month_scale_attrs(self):
        # make the monthly scaling factors attributes
        # this updates the title and adds a "how to" use section
        self.scale_attrs = self.attrs.copy()
        self.scale_attrs["title"] = (
            "Gridded U.S. Greenhouse Gas Inventory (Version 3): Monthly scaling "
            "factors for methane emissions."
        )
        self.scale_attrs["how_to_use"] = (
            "Sector-specific factors in this file can be multiplied by the annual "
            "methane emission flux data to estimate monthly emission fluxes for source "
            "sectors with strong monthly variability."
        )

    def create_final_netcdfs(self):
        # gch4i_flux_data_dict = {}
        for i, year in enumerate(YEARS):
            year_data_dict = {}
            out_path = final_gridded_dir / f"Gridded_GHGI_Methane_v3_{year}_draft.nc"
            for in_path in self.flux_data_files:
                # Get the file name and extract the source category and long name
                # in_path = v3_flux_data_files[i]
                source_cat = in_path.stem.split("_")[0]
                name_parts = in_path.stem.split("_")[:-3]
                long_name = f"""
                    {year} Methane emissions from IPCC source category 
                    {' '.join(name_parts)}"""
                var_name = f"emi_ch4_{'_'.join(name_parts)}"

                var_attrs = {
                    "source": source_cat,
                    "standard_name": "annual emissions",
                    "long_name": long_name,
                    "units": self.units,
                }

                group_ds = (
                    xr.open_dataset(in_path)
                    .sel(band=i + 1)
                    .rename(
                        {"band_data": var_name, "band": "time", "x": "lon", "y": "lat"}
                    )
                    # .drop_vars(["spatial_ref"])
                    .assign_attrs(var_attrs)
                    .expand_dims({"time": 1})
                    .assign_coords({"time": pd.DatetimeIndex([f"{year}-01-01"])})
                    .set_coords(["time", "lon", "lat"])
                )
                group_ds

                year_data_dict[var_name] = group_ds
            year_ds = xr.merge(year_data_dict.values())
            year_ds.attrs = self.attrs.copy().update(year=year)
            year_ds.to_netcdf(out_path, mode="w", format="NETCDF4")
            print(f"Saved {out_path.name}")
            # gch4i_flux_data_dict[year] = year_flux_ds

    def create_montly_scaling_files(self):
        gch4i_flux_data_dict = {}
        for i, year in enumerate(YEARS):
            year_data_dict = {}
            out_path = (
                final_gridded_dir
                / f"Gridded_GHGI_Methane_v3_Monthly_Scale_Factors_{year}_draft.nc"
            )
            for in_path in self.monthly_scale_files:
                # Get the file name and extract the source category and long name
                # in_path = v3_flux_data_files[i]
                source_cat = in_path.stem.split("_")[0]
                name_parts = in_path.stem.split("_")[:-3]
                long_name = f"{year} Monthly scale factors for IPCC source category {' '.join(name_parts)}"
                var_name = f"monthly_scale_factor_{'_'.join(name_parts)}"

                var_attrs = {
                    "source": source_cat,
                    "standard_name": "monthly_scaling",
                    "long_name": long_name,
                    "units": self.units,
                }

                group_ds = (
                    xr.open_dataset(in_path)
                    .sel(band=i + 1)
                    .rename(
                        {"band_data": var_name, "band": "time", "x": "lon", "y": "lat"}
                    )
                    # .drop_vars(["spatial_ref"])
                    .assign_attrs(var_attrs)
                    .expand_dims({"time": 1})
                    .assign_coords({"time": pd.DatetimeIndex([f"{year}-01-01"])})
                    .set_coords(["time", "lon", "lat"])
                )

                year_data_dict[var_name] = group_ds
            year_ds = xr.merge(year_data_dict.values())
            year_ds.attrs = self.scale_attrs.copy().update(year=year)
            year_ds.to_netcdf(out_path, mode="w", format="NETCDF4")
            print(f"Saved {out_path.name}")
            gch4i_flux_data_dict[year] = year_ds

    def write_outputs(self):
        self.create_final_netcdfs()
        self.create_montly_scaling_files()


# %%
file_writer = CreateFinalNetCDFs()
file_writer.write_outputs()


# For reference, we can look at the attributes (and other features) of the v2 data
# # %%
# from gch4i.config import V3_DATA_PATH

# v2_scale_file = V3_DATA_PATH / "Gridded_GHGI_Methane_v2_Monthly_Scale_Factors_2012.nc"
# v2_flux_file = V3_DATA_PATH / "Gridded_GHGI_Methane_v2_2012.nc"
# # %%
# v2_scale_ds = xr.open_dataset(v2_scale_file)
# v2_scale_ds.attrs
# # %%
# v2_flux_ds = xr.open_dataset(v2_flux_file)
# v2_flux_ds.attrs
# # %%
