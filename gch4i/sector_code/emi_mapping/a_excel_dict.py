from pathlib import Path
import os
from dotenv import load_dotenv, find_dotenv
import pandas as pd
import ast

load_dotenv(find_dotenv(usecwd=True))

# Set path to location of input data files (same directory where output files are saved)
# Define your own path to the data directory in the .env file
v3_data = os.getenv("V3_DATA_PATH")
V3_DATA_PATH = Path(v3_data)

figures_data_dir_path = V3_DATA_PATH / "figures"
global_data_dir_path = V3_DATA_PATH / "global"
ghgi_data_dir_path = V3_DATA_PATH / "ghgi"
tmp_data_dir_path = V3_DATA_PATH / "tmp"
emi_data_dir_path = V3_DATA_PATH / "emis"
proxy_data_dir_path = V3_DATA_PATH / "proxy"

# INPUT PATHS
file_path = ghgi_data_dir_path / "wastewater/Dictionary_Mapping_Test.xlsx"


def read_excel_dict(file_path, sheet):
    # Read in Excel Mapping file
    df = pd.read_excel(file_path,
                       sheet_name=sheet)
    # Convert the DataFrame to a dictionary with lists as values
    data_dict = df.set_index('Key').T.to_dict('list')

    return data_dict


def read_excel_dict2(file_path, sheet):
    # Read in Excel Mapping file
    df = pd.read_excel(file_path, sheet_name=sheet)

    # Convert the DataFrame to a dictionary with lists as values
    data_dict = df.set_index('Key').T.to_dict('list')

    # Convert floats to integers or leave strings/NaNs unchanged
    for key, values in data_dict.items():
        data_dict[key] = [
            int(value) if isinstance(value, float) and not pd.isnull(value) else value
            for value in values
        ]

    return data_dict



def read_excel_dict_cell(file_path, emission, sheet='Single_Cell'):
    """
    Reads an Excel file and returns a dictionary corresplonding with a title.
    """

    # Read in Excel File
    df = pd.read_excel(file_path, sheet_name=sheet)

    # Filter for Emissions Dictionary
    df = df.loc[df['Emission'] == emission].drop(columns=['Emission'])

    # Assign to object
    # result = ast.literal_eval(df['Dictionary'][0])
    result = ast.literal_eval(df.iloc[0, 0])

    return result


def read_excel_params(file_path, subsector, emission, sheet='testing'):
    """
    Reads add_param column from gch4i_data_guide_v3.xlsx and returns a dictionary.
    """
    # Read in Excel File
    df = (pd.read_excel(file_path, sheet_name=sheet)
            .assign(
                ghgi_group=lambda x: x['ghgi_group'].str.strip().str.casefold()
            ))

    # Edit emission
    # emission = emission.strip().casefold()

    # Filter for Emissions Dictionary
    df = df.loc[df['gch4i_name'] == subsector]

    df = df.loc[df['ghgi_group'] == emission, 'add_params']

    # Convert to dictionary
    result = ast.literal_eval(df.iloc[0])

    return result
