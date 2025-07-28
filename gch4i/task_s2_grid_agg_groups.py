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
        gg = GroupGridder(gch4i_name, gridding_group_data, prelim_gridded_dir)
        gg.run_gridding()
    except Exception as e:
        print(f"Error with {gch4i_name}")
        print(e)
        continue
    print()
# %%
# Example for running a single group
# gch4i_name = "1B2aii_petroleum_production"
gch4i_name = "3B_manure_management"
# gch4i_name = "3A_enteric_fermentation"
gridding_group_data = g_info.ready_groups_df.query(f"gch4i_name == '{gch4i_name}'")
gg = GroupGridder(gch4i_name, gridding_group_data, prelim_gridded_dir)
gg.run_gridding()
# %%
gg.plot_timeseries_comparison()
gg.plot_timeseries_comparison("mass")
# %%
plotting_data = gg.month_scale_ds["band_data"].where(lambda x: x["year"] == 2020)
# %%
fg = plotting_data.isel(band=123).plot.imshow(
    # col="band",
    # col_wrap=3,
    # # cmap=self.emi_custom_colormap,
    # # transform=ccrs.PlateCarree(),  # remember to provide this!
    # # subplot_kws={"projection": ccrs.PlateCarree()},
    # # interpolation=None,
    # # vmin=10**-15,
    # # vmax=10,
    # cbar_kwargs={
    #     "orientation": "horizontal",
    #     "shrink": 0.8,
    #     "aspect": 40,
    #     "extend": "neither",
    #     "label": "methane emissions (Mg a$^{-1}$ km$^{-2}$)",
    # },
    # robust=True,
    # figsize=(20, 20),
)
# plt.show()
# %%
gg.month_scale_ds["band_data"].isel(band=123).plot.imshow()
# %%
