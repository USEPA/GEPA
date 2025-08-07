"""
Name:                   create_final_netcdfs.py
Date Last Modified:     2025-08-4
Authors Name:           Nick Kruskamp (RTI International)
Purpose:                This File is used to format the emission flux data into
                        the final netCDF files.
                        The output is a gridded methane emissions product that can be
                        used for further analysis.
Notes:

"""

# %%
# %load_ext autoreload
# %autoreload 2
# %%
# import re

import numpy as np
import pandas as pd
import xarray as xr

from gch4i.config import final_gridded_dir, global_data_dir_path, prelim_gridded_dir
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
        self.units = "molec cm-2 s-1"
        # attributes written to the final files
        self.attrs = {
            "title": "Gridded U.S. Methane Anthropogenic Greenhouse Gas Inventory (Version 3)",
            "publication": "A gridded inventory of annual 2012-2022 U.S. anthropogenic methane emissions",
            "authors": "Nicholas Kruskamp, Hannah Lohman, Julie Powers, Shane Coffield, Yasmine Farhat, Chris Coxen, Andrew Burnette, Nathan Ellermeier, John Bollenbacher, Joannes D. Maasakkers",
            "history": "August 10, 2025",
            "conventions": "COARDS",
            "version": "1.0",
            "contact": "nkruskamp@rti.org",
            # NOTE: this is a placeholder for the year of the data. Year will be
            # replaced with the year of the data during processing.
            "year": "xxxx",
        }
        self._get_month_scale_attrs()
        self.flux_data_files = list(self.input_dir.glob("*ch4_emi_flux.tif"))
        self.monthly_scale_files = list(
            self.monthly_scaling_dir.glob("*_monthly_scaling.tif")
        )
        self.area_matrix_path = global_data_dir_path / "gridded_area_01_cm2.tif"

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

    def _get_area_matrix(self):
        """
        Load the area matrix from the global data directory.
        The area matrix is used to calculate the total emissions for each grid cell.
        """
        self.area_matrix = xr.open_dataset(self.area_matrix_path)
        self.area_matrix = (
            self.area_matrix.rename({"x": "lon", "y": "lat", "band": "time"})
            .rename_vars({"band_data": "grid_cell_area"})
            .reset_coords("spatial_ref", drop=True)
        )

    def create_final_netcdfs(self):
        self.gch4i_flux_dict = {}
        for i, year in enumerate(YEARS):
            # TODO: remove draft when final final.
            out_path = final_gridded_dir / f"Gridded_GHGI_Methane_v3_{year}_AugTest.nc"

            year_data_dict = {}
            for in_path in self.flux_data_files:
                # Get the file name and extract the source category and long name
                source_cat = in_path.stem.split("_")[0]
                name_parts = in_path.stem.split("_")[:-3]
                long_name = f"""{year} Methane emissions from IPCC source category {' '.join(name_parts)}"""
                var_name = f"emi_ch4_{'_'.join(name_parts)}"
                group_ds = (
                    xr.open_dataset(in_path)
                    .sel(band=i + 1)
                    .rename(
                        {"band_data": var_name, "band": "time", "x": "lon", "y": "lat"}
                    )
                    .expand_dims({"time": 1})
                    .assign_coords({"time": [0.0]})
                    .set_coords(["time", "lon", "lat"])
                    .reset_coords("spatial_ref", drop=True)
                )

                year_data_dict[var_name] = group_ds
            year_ds = xr.merge(
                year_data_dict.values()
            )  # note that xarray merge will drop variable attributes if they are
            # different from each other
            # setting the global file attributes
            year_ds.attrs = self.attrs.copy()
            year_ds.attrs["year"] = year  # update attributes to the current year
            # adjusting the global attributes
            year_ds.coords["time"].attrs["long_name"] = "time"
            year_ds.coords["time"].attrs["units"] = f"hours since {year}-01-01 00:00:00"
            year_ds.coords["time"].attrs["calendar"] = "standard"
            year_ds.coords["time"].attrs["axis"] = "T"
            year_ds.coords["lat"].attrs["long_name"] = "Latitude"
            year_ds.coords["lat"].attrs["units"] = "degrees_north"
            year_ds.coords["lat"].attrs["standard_name"] = "latitude"
            year_ds.coords["lat"].attrs["axis"] = "Y"
            year_ds.coords["lon"].attrs["long_name"] = "Longitude"
            year_ds.coords["lon"].attrs["units"] = "degrees_east"
            year_ds.coords["lon"].attrs["standard_name"] = "longitude"
            year_ds.coords["lon"].attrs["axis"] = "X"
            self.area_matrix["time"] = year_ds["time"]
            # add the area matrix to the dataset
            year_ds = xr.merge([year_ds, self.area_matrix])
            # add a moderate amount of compression and set the individua variable
            # attributes
            for var in year_ds:
                year_ds[var].encoding.update(dict(zlib=True, complevel=4))
                year_ds[var] = year_ds[var].fillna(0)
                year_ds[var].attrs = {}  # remove all dataset and variable attributes
                if var == "grid_cell_area":
                    year_ds[var].attrs = {
                        "standard_name": "grid_cell_area",
                        "long_name": "Grid cell areas to convert to absolute emissions",
                        "units": "cm^2",
                    }
                else:
                    source_cat = year_ds[var].name.split("_")[2]
                    name_parts = year_ds[var].name.split("_")[2:]
                    long_name = f"""{year} Methane emissions from IPCC source category 
                    {' '.join(name_parts)}"""
                    year_ds[var].attrs["source_category"] = source_cat
                    year_ds[var].attrs["standard_name"] = "annual_emissions"
                    year_ds[var].attrs["long_name"] = long_name
                    year_ds[var].attrs["units"] = "molec cm-2 s-1"
            # save the dataset to a netCDF file
            year_ds["time"] = [0.0]
            year_ds.to_netcdf(out_path, mode="w", format="NETCDF4")
            self.gch4i_flux_dict[year] = year_ds
            print(f"Saved {out_path.name}")

    def _calc_time_index(self, year):
        time_index = pd.date_range(
            f"{year}-01-01", periods=12, freq="MS"
        ) - pd.to_datetime(f"{year}-01-01")
        time_index_h = time_index / np.timedelta64(1, "h")
        return time_index_h

    def create_monthly_scaling_files(self):
        self.gch4i_month_scale_dict = {}
        # this creates a netCDF file for each year with all the monthly scaling factors
        # for that year by indexing the values.
        for i_month, year in zip(np.arange(0, (len(YEARS) * 12), 12), YEARS):
            # get the time index in hours since the start of the year
            time_index_h = self._calc_time_index(year)
            month_data_dict = {}
            out_path = final_gridded_dir / f"Gridded_GHGI_Methane_v3_{year}_draft.nc"
            for in_path in file_writer.monthly_scale_files:
                # Get the file name and extract the source category and long name
                # source_cat = in_path.stem.split("_")[0]
                name_parts = in_path.stem.split("_")[:-2]
                var_name = f"monthly_scale_factor_{'_'.join(name_parts)}"

                group_ds = (
                    xr.open_dataset(in_path)
                    .isel(band=slice(i_month, i_month + 12))
                    .rename(
                        {"band_data": var_name, "band": "time", "x": "lon", "y": "lat"}
                    )
                    .drop_vars(["spatial_ref"])
                    # .assign_attrs(var_attrs)
                    .assign_coords(
                        {"time": time_index_h.values}
                        # {"time": pd.date_range(f"{year}-01-01", periods=12, freq="MS")}
                    )
                    .set_coords(["time", "lon", "lat"])
                    # .reset_coords("spatial_ref", drop=True)
                )
                month_data_dict[var_name] = group_ds
            year_ds = xr.merge(month_data_dict.values())
            year_ds.attrs = self.scale_attrs.copy()
            year_ds.attrs["year"] = year  # update attributes to the current year
            # adjusting the global attributes
            year_ds.coords["time"].attrs["long_name"] = "time"
            year_ds.coords["time"].attrs["units"] = f"hours since {year}-01-01 00:00:00"
            year_ds.coords["time"].attrs["calendar"] = "standard"
            year_ds.coords["time"].attrs["axis"] = "T"
            year_ds.coords["lat"].attrs["long_name"] = "Latitude"
            year_ds.coords["lat"].attrs["units"] = "degrees_north"
            year_ds.coords["lat"].attrs["standard_name"] = "latitude"
            year_ds.coords["lat"].attrs["axis"] = "Y"
            year_ds.coords["lon"].attrs["long_name"] = "Longitude"
            year_ds.coords["lon"].attrs["units"] = "degrees_east"
            year_ds.coords["lon"].attrs["standard_name"] = "longitude"
            year_ds.coords["lon"].attrs["axis"] = "X"
            # add a moderate amount of compression and set the individua variable attributes
            for var in year_ds:
                year_ds[var].encoding.update(dict(zlib=True, complevel=4))
                year_ds[var] = year_ds[var].fillna(0)
                year_ds[var].attrs = {}  # remove all dataset and variable attributes
                source_cat = year_ds[var].name.split("_")[3]
                name_parts = year_ds[var].name.split("_")[3:]
                long_name = f"""{year} Monthly scale factors for IPCC source category {' '.join(name_parts)}"""
                year_ds[var].attrs["source_category"] = source_cat
                year_ds[var].attrs["standard_name"] = "monthly scale factor"
                year_ds[var].attrs["long_name"] = long_name
                year_ds[var].attrs["units"] = "1"
            year_ds.to_netcdf(out_path, mode="w", format="NETCDF4")
            self.gch4i_month_scale_dict[year] = year_ds

    def write_outputs(self):
        self._get_area_matrix()
        self.create_final_netcdfs()
        self.create_monthly_scaling_files()

    # TODO: plotting function to visualize the data
    def plot_data(self):
        pass


# %%
file_writer = CreateFinalNetCDFs()
file_writer.write_outputs()
# file_writer.plot_data()
# %%
# For reference, we can look at the attributes (and other features) of the v2 data
from gch4i.config import V3_DATA_PATH

v2_flux_file = V3_DATA_PATH / "Gridded_GHGI_Methane_v2_2012.nc"
v2_scale_file = V3_DATA_PATH / "Gridded_GHGI_Methane_v2_Monthly_Scale_Factors_2012.nc"
v3_flux_file = final_gridded_dir / "Gridded_GHGI_Methane_v3_2012_AugTest.nc"
v3_scale_file = final_gridded_dir / f"Gridded_GHGI_Methane_v3_2018_draft.nc"
# %%
v2_scale_ds = xr.open_dataset(v2_scale_file)
v2_scale_ds.close()
v2_scale_ds
# %%
v2_flux_ds = xr.open_dataset(v2_flux_file)
v2_flux_ds.close()
v2_flux_ds
# %%
v3_flux_ds = xr.open_dataset(v3_flux_file)
v3_flux_ds.close()
v3_flux_ds
# %%
v3_scale_ds = xr.open_dataset(v3_scale_file)
v3_scale_ds.close()
v3_scale_ds
# %%
