# %%
# %load_ext autoreload
# %autoreload 2
# %%
from calendar import month
from pathlib import Path

import duckdb
import geopandas as gpd
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from gch4i.config import emi_data_dir_path, global_data_dir_path, tmp_data_dir_path

sns.set_theme(style="darkgrid")
# %%
county_path = Path(global_data_dir_path, "tl_2020_us_county.zip")


# %%
class EntericEmiCheck:
    def __init__(self, emi_name, county_path):
        self.emi_name = emi_name
        self.county_path = county_path

    def read_county_data(self):
        self.county_gdf = (
            gpd.read_file(
                self.county_path, columns=["GEOID", "NAME", "geometry", "STATEFP"]
            )
            .astype({"GEOID": str, "STATEFP": int})
            .query("(STATEFP < 60) & (STATEFP != 2) & (STATEFP != 15)")
            .set_index("GEOID")
        )

    def read_emi_data(self):
        emi_paths = list(emi_data_dir_path.glob(f"{self.emi_name}*.csv"))
        emi_dfs = []
        for in_path in emi_paths:
            in_name = in_path.name.split("_")[-2]
            tmp_df = pd.read_csv(
                in_path, usecols=["state_code", "fips", "year", "month", "ghgi_ch4_kt"]
            ).assign(source=in_name)
            emi_dfs.append(tmp_df)

        self.emi_df = (
            pd.concat(emi_dfs, ignore_index=True)
            .assign(
                GEOID=lambda df: df["fips"].astype(int).astype(str).str.zfill(5),
                month_int=lambda df: pd.to_datetime(df["month"], format="%B").dt.month,
            )
            .drop(columns=["fips"])
        )

    def calc_monthly_yearly_emi(self):
        self.monthly_emi_df = (
            self.emi_df.groupby(["year", "month_int", "GEOID"])
            .agg(dict(state_code="first", ghgi_ch4_kt="sum"))
            .reset_index()
            .sort_values(["year", "month_int", "GEOID"])
        )

        self.yearly_emi_df = (
            self.emi_df.groupby(["year", "GEOID"])
            .agg(dict(state_code="first", ghgi_ch4_kt="sum"))
            .reset_index()
            .sort_values(["year", "GEOID"])
        )

    def merge_with_county(self):
        self.monthly_emi_gdf = self.county_gdf.merge(
            self.monthly_emi_df, left_index=True, right_on="GEOID", how="right"
        )

        self.yearly_emi_gdf = self.county_gdf.merge(
            self.yearly_emi_df, left_index=True, right_on="GEOID", how="right"
        )

    def plot_monthly_timeseries(self):
        g = sns.relplot(
            data=self.monthly_emi_df,
            x="month_int",
            y="ghgi_ch4_kt",
            hue="year",
            kind="line",
            palette="Dark2",
            height=6,
            aspect=1.6,
        )
        g.figure.suptitle("Monthly CH4 Emissions from Enteric Fermentation", y=1.05)
        g.set_axis_labels("Month", "CH4 Emissions (kt)")
        g.set_titles(col_template="{col_name}")
        g.set(xticks=range(1, 13))
        g.set_xticklabels(
            [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
        )
        sns.despine()
        plt.show()
# %%

emi_check = EntericEmiCheck("enteric_fermentation", county_path)
emi_check.read_county_data()
emi_check.read_emi_data()
emi_check.calc_monthly_yearly_emi()
emi_check.merge_with_county()
# %%
emi_check.__class__ = EntericEmiCheck
emi_check.plot_monthly_timeseries()
# %%
# plot_gdf = emi_gdf.query("year == 2020 and month == 'April'").copy()
plot_gdf = emi_check.yearly_emi_gdf.query("year == 2020").copy()

plot_gdf.to_parquet(tmp_data_dir_path / "enteric_fermentation_2020.parquet")

# %%

_, ax = plt.subplots(1, 1, figsize=(12, 8))

plot_gdf.plot(
    column="ghgi_ch4_kt",
    cmap="magma",
    scheme="natural_breaks",
    lw=0,
    k=6,
    legend=True,
    ax=ax,
)
sns.despine()
plt.show()
# %%
g = sns.relplot(
    data=emi_check.emi_df,
    x="month_int",
    y="ghgi_ch4_kt",
    hue="year",
    col="source",
    kind="line",
    palette="Dark2",
    col_wrap=2,
    facet_kws={"sharey": False, "sharex": True},
)
g.set_axis_labels("Month", "CH4 Emissions (kt)")
g.set_titles(col_template="{col_name}")
g.set_xticklabels(
    ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
)

# g.set(ylim=(0, 100))
g.figure.suptitle("Monthly CH4 Emissions from Enteric Fermentation", y=1.05)
g.add_legend(title="Year")
# %%

# %%
