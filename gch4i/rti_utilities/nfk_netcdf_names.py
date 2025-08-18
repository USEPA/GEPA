# %%
# %load_ext autoreload
# %autoreload 2

from pathlib import Path

import numpy as np
import rioxarray
import xarray as xr

from gch4i.config import proxy_data_dir_path, tmp_data_dir_path
from tqdm.auto import tqdm

# %%
proxy_paths = proxy_data_dir_path.glob("**/roads*.nc")
proxy_paths = sorted(proxy_paths)
proxy_paths
# %%
for proxy_path in tqdm(proxy_paths):
    try:
        ds = xr.open_dataset(proxy_path)
        out_ds = ds.copy(deep=True)
        ds.close()
        var_names = list(out_ds.data_vars.keys())
        if "rel_emi" not in var_names:
            print(proxy_path.name)
            var_names = list(out_ds.data_vars.keys())
            print("old var names: ", var_names)
            out_ds = out_ds.rename_vars({"road_emissions": "rel_emi"})
            var_names = list(out_ds.data_vars.keys())
            print("new var names: ", var_names)
            print()
            try:
                out_ds.to_netcdf(tmp_data_dir_path / proxy_path.name)
            except Exception as e:
                print(f"Error processing {proxy_path.name}: {e}")
    except Exception as e:
        print(f"Error processing {proxy_path.name}: {e}\n")
        continue
# %%