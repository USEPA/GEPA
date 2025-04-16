# %%
# %load_ext autoreload
# %autoreload 2
# %%
from IPython.display import display
from tqdm.auto import tqdm

from gch4i.config import prelim_gridded_dir
from gch4i.gridding_utils import GriddingInfo, GroupGridder

# %%
# get the emi/proxy data guide, status data, and v2/v3 crosswalk data
g_info = GriddingInfo()
g_info.display_group_status()
# %%
# example for running all gridding groups
# for each gridding group that is ready, perform final gridding
for gch4i_name, gridding_group_data in tqdm(
    g_info.ready_groups_df.groupby("gch4i_name"),
    total=g_info.ready_groups_df.gch4i_name.nunique(),
    desc="gridding groups",
):
    try:
        print(f"Running {gch4i_name}")
        gg = GroupGridder(gch4i_name, gridding_group_data, prelim_gridded_dir)
        gg.run_gridding()
    except Exception as e:
        print(f"Error with {gch4i_name}")
        print(e)
        continue
    print()
# %%
# Example for running a single group
gch4i_name = "1B1a_coal_mining_underground"
gridding_group_data = g_info.ready_groups_df.query(f"gch4i_name == '{gch4i_name}'")
gg = GroupGridder(gch4i_name, gridding_group_data, prelim_gridded_dir)
gg.run_gridding()
# %%
