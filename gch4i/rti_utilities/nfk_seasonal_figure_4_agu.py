# %%
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from gch4i.config import emi_data_dir_path
import duckdb

# %%
list(emi_data_dir_path.glob("*.parquet"))
# %%
# Load the data
in_path = emi_data_dir_path / "enteric fermentation_beef_emi.csv"
emi_df = pd.read_csv(in_path).assign(
    date=lambda df: pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%B"
    ),
    month=lambda df: pd.to_datetime(df["month"], format="%B").dt.month,
    year=lambda df: pd.to_datetime(df["year"].astype(str) + "-07", format="%Y-%m"),
)
emi_df
# %%
all_ef_df = duckdb.execute(
    f"SELECT * FROM read_csv('{str(emi_data_dir_path)}/manure*.csv')"
).fetchdf()
all_ef_df


# %%
tmp_df = all_ef_df.assign(
    date=lambda df: pd.to_datetime(
        df["year"].astype(str) + "-" + df["month"].astype(str), format="%Y-%B"
    ),
    month=lambda df: pd.to_datetime(df["month"], format="%B").dt.month,
    year=lambda df: pd.to_datetime(df["year"].astype(str) + "-01", format="%Y-%m"),
)
tmp_df
# %%
yearly_sums = tmp_df.groupby("year")["ghgi_ch4_kt"].sum().reset_index()
yearly_sums
# %%
monthly_sums = tmp_df.groupby("date")["ghgi_ch4_kt"].sum().reset_index()
monthly_sums
# %%

fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot yearly emissions on the primary y-axis
sns.lineplot(data=yearly_sums, x="year", y="ghgi_ch4_kt", ax=ax1, color="xkcd:teal")
ax1.set_xlabel("Date")
ax1.set_ylabel("Yearly Emissions (kt ch4)", color="xkcd:teal")
ax1.tick_params(axis="y", labelcolor="xkcd:teal")

ax1.set_xticks(yearly_sums["year"])


# Create a secondary y-axis for monthly emissions
ax2 = ax1.twinx()

# Plot monthly emissions on the secondary y-axis
sns.lineplot(data=monthly_sums, x="date", y="ghgi_ch4_kt", ax=ax2, color="xkcd:lavender")
ax2.set_ylabel("Monthly Emissions (kt ch4)", color="xkcd:lavender")
ax2.tick_params(axis="y", labelcolor="xkcd:lavender")

ax1.grid(axis="x")

# Show the plot
plt.title("Enteric Fermentation:\nYearly vs Monthly Emissions")
fig.tight_layout()  # Adjust layout to make room for the secondary y-axis
plt.show()
# %%
# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot yearly emissions on the primary y-axis
sns.barplot(data=emi_df, x="year", y="ghgi_ch4_kt", ax=ax1, label="Yearly Emissions")
# sns.lineplot(data=emi_df, x="year", y="ghgi_ch4_kt", ax=ax1, label="Yearly Emissions")
ax1.set_xlabel("Date")
ax1.set_ylabel("Yearly Emissions (kt ch4)", color="xkcd:teal")
ax1.tick_params(axis="y", labelcolor="xkcd:teal")

# Create a secondary y-axis for monthly emissions
ax2 = ax1.twinx()

# Plot monthly emissions on the secondary y-axis
sns.lineplot(
    data=emi_df,
    x="date",
    y="ghgi_ch4_kt",
    ax=ax2,
    color="xkcd:lavender",
    label="Monthly Emissions",
)
ax2.set_ylabel("Monthly Emissions (kt ch4)", color="xkcd:lavender")
ax2.tick_params(axis="y", labelcolor="xkcd:lavender")

# TODO: add major and minor ticks for years

# Show the plot
plt.title("Enteric Fermentation:\nYearly vs Monthly Emissions")
fig.tight_layout()  # Adjust layout to make room for the secondary y-axis
plt.show()
# %%
# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot yearly emissions on the primary y-axis
sns.lineplot(data=emi_df, x="year", y="ghgi_ch4_kt", ax=ax1, label="Yearly Emissions")
ax1.set_xlabel("Date")
ax1.set_ylabel("Emissions (kt ch4)")
ax1.tick_params(axis="y")
ax1.set_xticks(yearly_sums["year"])

# Plot monthly emissions on the secondary y-axis
sns.lineplot(
    data=emi_df,
    x="date",
    y="ghgi_ch4_kt",
    ax=ax1,
    color="xkcd:lavender",
    label="Monthly Emissions",
)

ax1.grid(axis="x")

# Show the plot
plt.title("Manure Management:\nAverage Yearly vs Monthly Emissions")
fig.tight_layout()  # Adjust layout to make room for the secondary y-axis
plt.show()
# %%
