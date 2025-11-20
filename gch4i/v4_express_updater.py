"""
Author:     Nick Kruskamp
Date:       October 6, 2025
Purpose:    The primary purpose of this script is fill in the data year 2023 using v3
            proxy files that contain data for years 2012-2022.

            Express proxies are those that could not be quickly updated due to data that is
            not openly available or requires significant processing.

            This script is designed to standarize the express update for v4 proxies.
            This process will follow a fixed approach for each proxy:

            The steps are:

            Pull the original v3 proxy files that contains data for years 2012-2022
            1.  Copy the year 2022 data to 2023.
            2.  if this does not satify QC, look for individual state/year combinations
                that are missing data and fill 2023 with that data.
            3.  if there are no state data at all, fill with the entire state
                geometry.

            The development creates a new class called V4ExpressUpdater that handles
            the update process. It inherits from the original v3 EmiProxyGridder class.

Inputs:

            proxy_path: Path. Path to the v3 (!!) proxy file that needs to be updated.
            emi_path: Path. Path to the v4 (!!) emissions file.

Outputs:
            output_path: Path. Path to write the updated v4 proxy file.
"""

# %%
from IPython.display import display
from tqdm.auto import tqdm

from gch4i.gridding_utils import GriddingInfo, V4ExpressUpdater, EmiProxyGridder

# %%
g_info = GriddingInfo()
g_info.get_ready_pairs()
g_info.get_ready_groups()
g_info.display_all_group_statuses()
g_info.display_all_pair_statuses()

SKIP = True
SKIP_THESE = [
    "complete",
    # "emi file not found",
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
express_grid_these = g_info.pairs_ready_for_gridding_df.query(
    "(express == True) & (~status.isin(@SKIP_THESE))"
)
express_grid_these
# %%
regular_grid_these = g_info.pairs_ready_for_gridding_df.query(
    # "(express == False)"
    "(express == False) & (~status.isin(@SKIP_THESE))"
)
regular_grid_these

# %%
for row in tqdm(
    express_grid_these.itertuples(index=False),
    total=len(express_grid_these),
):
    if SKIP and row.status in SKIP_THESE:
        print(f"Skipping: {row.gch4i_name} {row.emi_id} {row.proxy_id} ({row.status})")
        continue
    # if row.emi_id in ["trans_onshore_emi", "trans_refining_emi"]:
    #     print(f"Skipping transport: {row.gch4i_name} {row.emi_id} {row.proxy_id}")
    #     continue
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
for row in tqdm(
    regular_grid_these.itertuples(index=False),
    total=len(regular_grid_these),
):
    # base_name = f"{row.gch4i_name}-{row.emi_id}-{row.proxy_id}"
    # out_path = v4_logging_dir / row.gch4i_name / f"{base_name}.tif"
    # if not out_path.exists():
    # print(f"Gridding: {row.gch4i_name} {row.emi_id} {row.proxy_id}")

    # if row.emi_id in ["trans_onshore_emi", "trans_refining_emi"]:
    #     print(f"Skipping transport: {row.gch4i_name} {row.emi_id} {row.proxy_id}")
    #     continue

    try:
        gridder = EmiProxyGridder(
            gch4i_name=row.gch4i_name,
            emi_id=row.emi_id,
            proxy_id=row.proxy_id,
        )
        gridder.run_gridding()
    except Exception as e:
        print(f"ERROR: {row.gch4i_name} {row.emi_id} {row.proxy_id} {e}")
        # break


# %%
g_info.get_status_table()
# %%
g_info.display_all_group_statuses()
# %%
