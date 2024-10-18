from pathlib import Path
livestock_dir_path = Path("C:/Users/nkruskamp/Environmental Protection Agency (EPA)/Gridded CH4 Inventory - Task 2/ghgi_v3_working/GEPA_Source_Code/GEPA_Livestock/InputData/USDA_Census")

livestock_files = list(livestock_dir_path.glob("*_county*"))

livestock_types = set([x.stem.split("_")[2] for x in livestock_files])

dict(enumerate(livestock_types, 1))