"""
Author:     Nick Kruskamp
Date:       October 6, 2025
Purpose:    The primary purpose of this script is fill in the data year 2023 using v3
            proxy files that contain data for years 2012-2022.

            This script is designed to standarize the express update for v4 proxies.
            These proxies are ones that cannot be quickly updated due to data that is
            not openly available or requires significant processing. This process will
            follow a fixed approach for each proxy:

            The steps are:

            1. Pull the original v3 proxy files that contains data for years 2012-2022
            2. Replicate the 2022 data to 2023 using the v3 proxy data
            3. if this does not satify QC, look for individual state/year combinations
               that are missing data and fill 2023 with that data.

Inputs:

            proxy_path: Path. Path to the v3 (!!) proxy file that needs to be updated.
            emi_path: Path. Path to the v4 (!!) emissions file.

Outputs:
            output_path: Path. Path to write the updated v4 proxy file.
"""

# %%
from networkx import display
from tqdm.auto import tqdm

from gch4i.gridding_utils import GriddingInfo, V4ExpressUpdater

# %%
g_info = GriddingInfo()
g_info.get_ready_pairs()
g_info.get_ready_groups()
g_info.display_all_group_statuses()
g_info.display_all_pair_statuses()

SKIP = True
SKIP_THESE = [
    "complete",
    "emi file not found",
    # "monthly failed, annual failed",
    # "express update failed",
    # "monthly failed, annual complete",
    # "failed state/year QC",
    # "error reading proxy",
    # "emi missing years",
    # "proxy file does not exist",
    # "proxy has emtpy geometries",
    # "failed raster QC",
    # "failed allocation QC",
    # "proxy year has NAs",
    # "failed annual raster QC",
    # "failed monthly raster QC",
    # "failed",
]

# %%
for row in tqdm(
    g_info.pairs_ready_for_gridding_df.itertuples(index=False),
    total=len(g_info.pairs_ready_for_gridding_df),
):
    if SKIP and row.status in SKIP_THESE:
        # print(f"Skipping: {row.gch4i_name} {row.emi_id} {row.proxy_id} ({row.status})")
        continue
    try:
        exp_updater = V4ExpressUpdater(
            gch4i_name=row.gch4i_name,
            emi_id=row.emi_id,
            proxy_id=row.proxy_id,
        )
        exp_updater.run_update()
    except Exception as e:
        print(f"ERROR: {row.gch4i_name} {row.emi_id} {row.proxy_id} {e}")
        # break
# %%
g_info.get_status_table(save=True)
# %%
exp_updater.read_emi_file()
display(exp_updater.emi_df.head())
exp_updater.read_proxy_file()
display(exp_updater.proxy_gdf.head())
exp_updater.get_rel_emi_col()
# %%
exp_updater.copy_2022_to_2023()
# %%
exp_updater.proxy_gdf.year.value_counts().sort_index()
# %%
2023 in exp_updater.proxy_gdf.year.unique()
# %%
exp_updater.proxy_gdf.state_code.value_counts().sort_index()
# %%
