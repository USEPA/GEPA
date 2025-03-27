# %%
%load_ext autoreload
%autoreload 2
# %%
from pathlib import Path

import pandas as pd
from IPython.display import display
from tqdm.auto import tqdm

from gch4i.config import V3_DATA_PATH, logging_dir, prelim_gridded_dir
from gch4i.gridding_utils import GriddingMapper, GroupGridder, get_status_table

# %%
# The path to the status database for all v3 emi/proxy pairs
status_db_path: Path = logging_dir / "gridding_status.db"
# The path to the v2/v3 name crosswalk to QC v2/v3 flux data
v2_data_path: Path = V3_DATA_PATH.parents[1] / "v2_v3_comparison_crosswalk.csv"
# The data guide file for all emi/proxy pairs
mapping_file_path: Path = V3_DATA_PATH.parents[1] / "gch4i_data_guide_v3.xlsx"

# get the emi/proxy data guide, status data, and v2/v3 crosswalk data
g_mapper = GriddingMapper(mapping_file_path)
status_df = get_status_table(status_db_path, save=False)
v2_df = pd.read_csv(v2_data_path).rename(columns={"v3_gch4i_name": "gch4i_name"})

# merge the status data with the mapping data
status_df = status_df.merge(
    g_mapper.mapping_df, on=["gch4i_name", "emi_id", "proxy_id"]
)

# get the status of each gridding group
group_ready_status = (
    status_df.groupby("gch4i_name")["status"]
    .apply(lambda x: x.eq("complete").all())
    .to_frame()
)

# filter the emi/proxy data to only those groups that are ready for gridding
# NOTE: not all v3 gridding groups have a v2 product
ready_for_gridding_df = (
    group_ready_status[group_ready_status["status"] == True]
    .join(v2_df.set_index("gch4i_name"))
    .fillna({"v2_key": ""})
    .astype({"v2_key": str})
).merge(g_mapper.mapping_df, on="gch4i_name", how="left")

# display the progress of the emi/proxy pairs
print("percent of emi/proxy pairs by status")
display(status_df["status"].value_counts(normalize=True).multiply(100).round(2))

# display the progress of the gridding groups
print("percent of gridding groups ready")
display(
    group_ready_status["status"].value_counts(normalize=True).multiply(100).round(2)
)

# %%
# for each gridding group that is ready, perform final gridding
for gch4i_name, gridding_data in tqdm(
    ready_for_gridding_df.groupby("gch4i_name"),
    # gridding_ready_df.sample(4).groupby("gch4i_name"),
    total=ready_for_gridding_df.gch4i_name.nunique(),
    desc="gridding groups",
):
    try:
        print(f"Running {gch4i_name}")
        group_gridder = GroupGridder(gch4i_name, gridding_data, prelim_gridded_dir)
        group_gridder.run_gridding()
    except Exception as e:
        print(f"Error with {gch4i_name}")
        print(e)
        continue
    print()
# %%
# gch4i_name = '1B1a_abandoned_coal'
gch4i_name = '1A_mobile_combustion'
gridding_data = ready_for_gridding_df.query(f"gch4i_name == '{gch4i_name}'")
group_gridder = GroupGridder(gch4i_name, gridding_data, prelim_gridded_dir)
group_gridder.run_gridding()
# %%
group_gridder.annual_flux_da.sel(time=2012).where(lambda x: x>0).plot()
group_gridder.annual_flux_da.sel(time=2012).where(lambda x: x>0).plot()
# %%
group_gridder.v2_flux_da.sel(time=2012).where(lambda x: x>0).plot()
# %%
group_gridder.annual_mass_da.where(lambda x: x>0).sel(time=2012).plot()
# %%
group_gridder.flux_diff_da.sel(time=2012).plot()
# group_gridder.flux_diff_da.sel(time=2012).where(lambda x: x>0).plot()
# %%
group_gridder.mass_diff_da.sel(time=2012).where(lambda x: x>0).plot()
# %%
group_gridder.annual_mass_da.sel(time=2012).where(lambda x: x>0).plot()


# %%
(group_gridder.annual_mass_da.x == group_gridder.flux_diff_da.x).all()
# %%
(group_gridder.annual_mass_da.y == group_gridder.flux_diff_da.y).all()
# group_gridder.flux_diff_da
# %%
(group_gridder.annual_mass_da.y == group_gridder.mass_diff_da.y).all()
# %%

(group_gridder.annual_mass_da.x == group_gridder.mass_diff_da.x).all()

# %%

