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