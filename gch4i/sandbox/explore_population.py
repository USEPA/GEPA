'''
Explore population_proxy.nc

'''

#%%
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

#%%
pop_path = r'C:\Users\nellermeier\Environmental Protection Agency (EPA)\Gridded CH4 Inventory - RTI 2024 Task Order\Task 2\ghgi_v3_working\v3_data\proxy\population_proxy.nc'

ds = xr.open_dataset(pop_path)

#%%
# get raveled area of non-na values
pop_rav = ds['population'].values.ravel()
pop_rav = pop_rav[~np.isnan(pop_rav)]

print(f"Number of non-na values: {len(pop_rav)}")
print(f"Mean population: {np.mean(pop_rav)}")
print(f"Max population: {np.max(pop_rav)}")
print(f"Min population: {np.min(pop_rav)}")

#%%
ds.population.isel(year=0).plot()

#%%
# plot histogram of values less than 0.1
plt.hist(pop_rav[pop_rav < 0.1], bins=100)