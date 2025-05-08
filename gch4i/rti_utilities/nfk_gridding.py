            fig, axs = plt.subplots(1, 4, figsize=(12, 6))
            axs[0].imshow(proxy_arr.values, cmap="Spectral")
            axs[1].imshow(res.values, cmap="Spectral")
            tmp_gdf = county_gdf.query("fips == @fips").join(
                emi_df.query("fips == @fips").sample(1).set_index("fips")
            )
            tmp_raster = rasterize(
                [
                    (geom, value)
                    for geom, value in zip(tmp_gdf.geometry, tmp_gdf.ghgi_ch4_kt)
                ],
                out_shape=proxy_arr.shape,
                transform=transform,
                fill=0,
                dtype="float32",
            )
            axs[2].imshow(tmp_raster, cmap="Spectral")
            tmp_gdf.plot(ax=axs[3])
            plt.show()

            emi_xr = (
                emi_df[["year_month", "fips", "ghgi_ch4_kt"]]
                .set_index(["year_month", "fips"])
                .to_xarray()
            )
            emi_xr


# %%
# Create a connection to the SQLite database

conn = sqlite3.connect(status_db_path)
cursor = conn.cursor()

# Create a table to store the status of each row
cursor.execute(
    """
CREATE TABLE IF NOT EXISTS gridding_status (
    gch4i_name TEXT,
    emi_id TEXT,
    proxy_id TEXT,
    status TEXT,
    PRIMARY KEY (gch4i_name, emi_id, proxy_id)
)
"""
)
conn.commit()
# %%
# ======================================================================================
# prepare and clean up the logging and QC output directory
# ======================================================================================
for gch4i_name in mapping_df.gch4i_name:
    all_group_files = list(logging_dir.rglob(f"{gch4i_name}*"))
    out_qc_dir = logging_dir / gch4i_name
    out_qc_dir.mkdir(exist_ok=True, parents=True)
    all_group_files = [x for x in all_group_files if not x.is_dir()]
    all_group_files = [x for x in all_group_files if x.parent != out_qc_dir]
    if all_group_files:
        print(f"Moving {len(all_group_files)} files to {out_qc_dir}")
        for file in all_group_files:
            destination = out_qc_dir / file.name
            if destination.exists():
                destination.unlink()
            file.rename(destination)

# %%