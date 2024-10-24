# EPA U.S. Gridded Methane Emissions Inventory (gridded GHGI)

This code accompanies the peer-reviewed manuscript _Maasakkers, J. D., et al., A gridded inventory of 2012-2018 U.S. anthropogenic methane emissions, ES&T (2023)_ [[link]](https://pubs.acs.org/doi/full/10.1021/acs.est.3c05138). This product is an update to gridded methane GHGIv1, described previously in _Maasakkers et al. (2016)_ [[link]](https://pubs.acs.org/doi/10.1021/acs.est.6b02878). More information is available on the following U.S. EPA website: https://www.epa.gov/ghgemissions/us-gridded-methane-emissions

Please cite the original manuscript when using the code or data generated by this repository: _Maasakkers et al. (2023)_ [[link]](https://pubs.acs.org/doi/full/10.1021/acs.est.3c05138)


********
## Background
The gridded EPA U.S. anthropogenic methane greenhouse gas inventory (gridded methane GHGI) includes spatially and temporally resolved (gridded) maps of annual anthropogenic methane emissions (0.1°×0.1°) for the contiguous United States (CONUS). Total gridded methane emissions for each emission source sector are consistent with national annual U.S. anthropogenic methane emissions reported in the _U.S. EPA Inventory of U.S. Greenhouse Gas Emissions and Sinks_ [[link]](https://www.epa.gov/ghgemissions/inventory-us-greenhouse-gas-emissions-and-sinks) (U.S. GHGI).

This code produces the main gridded GHGI v2, which includes gridded annual U.S. anthropogenic methane emissions for 2012-2018 for 26 source categories. This dataset is developed to be consistent with the national U.S. GHGI published in 2020 (_U.S. EPA, Inventory of U.S. Greenhouse Gas Emissions and Sinks: 1990 - 2020. U.S. Environmental Protection Agency, 2020, EPA 430-R-22-003_) [[link]](https://www.epa.gov/ghgemissions/inventory-us-greenhouse-gas-emissions-and-sinks-1990-2018)

This code also extends the version 2 methodology to produce 2012-2020 gridded methane emissions consistent with the 2022 U.S. GHGI. In this process, the source-specific spatial patterns from the 2012-2018 are used to downscale more recent national methane emission estimates from the 2022 GHGI Report.

Final gridded data are available on Zenodo: https://zenodo.org/records/8367082

*********
## Code & Repository Overview

The gridded GHGI (v2 and express extension) code is written in Python in Jupyter Notebooks. Each sector-based script is designed to read in sector-specific U.S. national emissions from the 2020 GHGI Report and allocate those emissions to a CONUS grid using available spatial information in the underlying inventory activity datasets. These datasets are not provided in the public version of this repository. References to the relevant data sources are provided in the _Maasakkers et. al., (2023)(https://pubs.acs.org/doi/full/10.1021/acs.est.3c05138)_ manuscript. Some of these underlying spatial proxy datasets include proprietary information. Others are available upon request. 


### Repository Contents

#### Code
This repository includes:
- /code/GEPA_2022_Supplement - includes a Jupyter Notebook for the express extension for Natural Gas Post-Meter Emissions
- /code/Global_Functions - includes global functions used across all sector-specific Jupyter Notebooks.
- /code/Final_Gridded_Data/ - folder where the annual source-specific methane flux files are output.
- /code/File_Processing/ - folder that contains the scripts to process the source-specific raw output flux files, into annual flux files published to the Zenodo repository
- /code/...  - each folder corresponds to an inventory source category, including the Jupyter Notebooks for both the main v2 and the express extension




#### Emissions
Source categories include:

Agriculture
- Enteric Fermentation
- Manure Management
- Rice Cultivation
- Field Burning of Agricultural Residues

Natural Gas Systems
- Exploration
- Production
- Transmission & Storage
- Processing
- Distribution
- Post-Meter (express extension only)

Petroleum Systems
- Exploration
- Production
- Transport
- Refining

Waste
- Municipal Solid Waste (MSW) Landfills
- Industrial Landfills
- Domestic Wastewater Treatment & Discharge
- Industrial Wastewater Treatment & Discharge
- Composting

Coal Mines
- Underground Coal Mining
- Surface Coal Mining
- Abandoned Underground Coal Mines

Other
- Stationary combustion
- Mobile Combustion
- Abandoned Oil and Gas Wells
- Petrochemical Production
- Ferroalloy Production

Note: The gridded methane GHGI does not include emissions from the Land Use, Land Use Change, or Forestry (LULUCF) category of the inventory. Methane emissions from these sources include but are not limited to emissions from fires and flooded lands.

## Contributing
To report an error, please open an issue on GitHub

## Guidance for setting up your local environment and .env file

If you’re starting fresh, I recommend the bare-bones install of conda that by uses the conda-forge channel for downloading packages, called miniforge:
https://github.com/conda-forge/miniforge

If you want to go even deeper, look at micromamba: https://mamba.readthedocs.io/en/latest/installation/micromamba-installation.html

Once you have conda installed, you need to create the environment.
- open PowerShell or bash.
- if you have just installed conda, run:
```conda init```

close and reopen your shell to complete the changes.
create your environment
- Navigate to the GEPA folder by running (update your path):
```cd 'C:\Users\nkruskamp\Research Triangle Institute\EPA Gridded Methane - Task 2\code\GEPA'```
- you can use the command ls to list the files in that directory and make sure you see environment.yml
- then create the environment, run:
```conda env create --file environment.yml```

This will install the conda environment “gch4i” into you conda environments with the packages needed to run the notebooks.

Each user will need to add a .env file to your local version of the repo. In VS Code, right click in the explorer (generally located on the left side of your VS Code window) and select "new file." Name this file ".env". 

Inside your .env file, input this text:

```V3_DATA_PATH = "YOUR_USER_PATH_TO_V3_DATA"```

Replace the text YOUR_USER_PATH_TO_V3_DATA with your actual local path. For example,

```"/Users/username/Library/CloudStorage/OneDrive-SharedLibraries-EnvironmentalProtectionAgency(EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/v3_data"```

Save your updated .env file and everything should now point to your personal v3 data path.
