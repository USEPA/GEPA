# %%
from gch4i.config import V3_DATA_PATH
import xarray as xr
import pandas as pd
import seaborn as sns

# %%
res_list = []
v2_month_files = list(V3_DATA_PATH.glob("*Monthly_Scale_Factors_*.nc"))
for in_path in v2_month_files:
    year = int(in_path.stem.split("_")[-1])
    ds = xr.open_dataset(in_path)
    res_df = (
        ds.sum(dim="time")
        .where(lambda x: x > 0)
        .to_dataframe()
        .dropna(axis=0, how="all")
        .describe()
        .loc[["min", "max", "mean", "std"]]
        .T.assign(year=year)
        .reset_index()
    )
    res_list.append(res_df)
    ds.close()
res_df = pd.concat(res_list, ignore_index=True).rename(columns={"index": "gch4i_name"})
# %%


sns.relplot(
    data=res_df,
    x="year",
    y="mean",
    hue="gch4i_name",
    kind="line",
    height=5,
    aspect=2,
    facet_kws={"sharey": False},
)
# %%
sns.relplot(
    data=res_df,
    x="year",
    y="max",
    hue="gch4i_name",
    kind="line",
    height=5,
    aspect=2,
    facet_kws={"sharey": False},
)
# %%
sns.relplot(
    data=res_df,
    x="year",
    y="min",
    hue="gch4i_name",
    kind="line",
    height=5,
    aspect=2,
    facet_kws={"sharey": False},
)
# %%
v2_files = list(V3_DATA_PATH.glob("Gridded_GHGI_Methane_v2_*.nc"))
v2_files = [f for f in v2_files if "Monthly_Scale_Factors" not in f.stem]
v2_files.sort(key=lambda x: int(x.stem.split("_")[-1]))
key_list = []
for in_path in v2_files:
    ds = xr.open_dataset(in_path)
    key_list.append(list(ds.keys()))
    ds.close()
key_list
# %%
