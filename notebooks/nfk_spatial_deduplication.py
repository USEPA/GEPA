# NOTE: So at this point, there are clearly duplicates in the dataset that need to be
# removed. Unfortunately for us there is no out of the box geometric way to do this.
# so we resolve this in a loop.
facility_concat_2_gdf["has_sim_geo"] = [
    (facility_concat_2_gdf.geom_equals_exact(x, tolerance=0.001).sum() > 1)
    for x in facility_concat_2_gdf.geometry.values
]
# %%
facility_concat_2_gdf[facility_concat_2_gdf["has_sim_geo"]].sort_values(
    ["geometry", "formatted_fac_name"]
)