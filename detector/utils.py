import pandas as pd
import os
from pathlib import Path
import numpy as np

def get_path_data() -> Path:
    #abs_path = Path(__file__).parents[1].absolute()
    return Path(os.environ.get("PATH_DATA"))

def load_data(source: str="xl-1542M",
              truncation: bool=True,
              n_rows: int=500_000) -> dict[pd.DataFrame]:
    '''Load the data in dictionary of pandas Dataframes.
    ---
    source: specifies the outputs of a GPT-2 model

    ---
    truncation: specifies if Top-K 40 truncation data is used

    ---
    n_rows: specifies the fraction of train-data loaded. Smaller values for testing the code.'''
    path_data = get_path_data()
    final_data={}
    for split in ["train", "valid", "test"]:
        data={}
        if truncation:
            file_path = path_data / f"{source}-k40.{split}.csv"
        else:
            file_path = path_data / f"{source}.{split}.csv"

        if split == "train":
            data['fake'] = pd.read_csv(file_path, usecols=["text"], nrows=n_rows//2) # nrows to have balanced dataset
        else:
            data['fake'] = pd.read_csv(file_path, usecols=["text"])
        data['fake']["AI"] = 1 # AI written

        file_path = path_data / f"webtext.{split}.csv"

        if split == "train":
            data['true'] = pd.read_csv(file_path, usecols=["text"], nrows=n_rows//2) # nrows to have balanced dataset
        else:
            data['true'] = pd.read_csv(file_path, usecols=["text"])
        data['true']["AI"] = 0 # not AI written

        final_data[split] = pd.concat([data["true"], data["fake"]])

    return final_data

# create a function to split datasets into subsets of small, medium and large texts
def divide_frame(df: pd.DataFrame) -> tuple[pd.DataFrame]:
    '''Return tuple of DataFrames with small, medium and large texts
    split according to 1. and 3. quartile of text length distribution.'''
    q = np.percentile(df.text_length, [25, 75])
    q_1, q_3 = q[0], q[1]
    df_small = df[df.text_length < q_1].reset_index(drop=True)
    df_medium = df[df.text_length.between(q_1, q_3)].reset_index(drop=True)
    df_large = df[df.text_length > q_3].reset_index(drop=True)
    return df_small, df_medium, df_large
