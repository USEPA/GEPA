# %%
from IPython.display import display
from tqdm.auto import tqdm

from gch4i.config import v4_prelim_gridded_dir
from gch4i.gridding_utils import GriddingInfo, GroupGridder

# %%
# get the emi/proxy data guide, status data, and v2/v3 crosswalk data
g_info = GriddingInfo(update_mapping=True, save_file=True)
g_info.display_all_group_statuses()
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
        gg = GroupGridder(gch4i_name, gridding_group_data, v4_prelim_gridded_dir)
        gg.run_gridding()
    except Exception as e:
        print(f"Error with {gch4i_name}")
        print(e)
        continue
    print()
# %%
# Example for running a single group
# gch4i_name = "1B2aii_petroleum_production"
gch4i_name = "3F4_fbar"
# gch4i_name = "3B_manure_management"
# gch4i_name = "1A_stationary_combustion"
# gch4i_name = "3A_enteric_fermentation"
# gch4i_name = "4A1_4A2_Forest_land_remaining_forest_land"
# gch4i_name = "1B2biv_ng_transmission_storage"
# gch4i_name = "5A_industrial_landfills"
gridding_group_data = g_info.ready_groups_df.query(f"gch4i_name == '{gch4i_name}'")
gg = GroupGridder(gch4i_name, gridding_group_data, v4_prelim_gridded_dir)
gg.run_gridding()
# %%
gg.__class__ = GroupGridder
gg.calculate_monthly_scaling()
gg.month_scale_check
# %%
