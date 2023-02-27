import os
import re

import numpy as np
import pandas as pd

an_files = []
for root, dirs, files in os.walk('../analyze_these'):
    for file in files:
        if file == "analysis.csv":
            an_files.append(f"{root}/{file}")

frames = []

for filename in an_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    tomo = re.search(".*(tomo_([0-9]{8})-[0-9]{2})", filename)
    df['tomo'] = tomo[1]
    if "ncr1" in filename:
        df['type'] = "ncr1"
    elif "npc2" in filename:
        df['type'] = "npc2"
    else:
        df['type'] = "wt"
    frames.append(df)

frame = pd.concat(frames, axis=0, ignore_index=True)

for cell_type in ["ncr1", "npc2", "wt"]:
    filtered = frame.where(frame["type"] == cell_type).dropna()
    print(f"% consumed by {cell_type}: {np.sum(filtered['consumed']) / len(filtered['consumed'])}")
    print(f"Mean volume for {cell_type}: {np.mean(filtered['volume'])}")
    print(f"Mean distance for {cell_type}: {np.mean(filtered['distance'])}")

types = ["wt", "ncr1", "npc2"]


def get_type(ctype) -> pd.DataFrame:
    return frame.where(frame["type"] == ctype).dropna()


def get_consumed(arr) -> pd.DataFrame:
    # ugly hack
    if type(arr) is str:
        arr = get_type(arr)
    new_arr = arr[(arr['consumed'] == 1)]
    return new_arr

def get_bound(arr) -> pd.DataFrame:
    # ugly hack
    if type(arr) is str:
        arr = get_type(arr)
    new_arr = arr[(arr['consumed'] == 0) & (arr['distance'] == 0)]
    return new_arr

def get_not_free(arr) -> pd.DataFrame:
    # ugly hack
    if type(arr) is str:
        arr = get_type(arr)
    new_arr = arr[(arr['consumed'] == 1) | (arr['distance'] == 0)]
    return new_arr

def get_free(arr) -> pd.DataFrame:
    if type(arr) is str:
        arr = get_type(arr)
    new_arr = arr[(arr['consumed'] == 0) & (arr['distance'] > 0)]
    return new_arr


dest = "../analysis_results"
for ctype in types:
    cons = get_free(ctype)
    cons['distance'].to_csv(f"{dest}/{ctype}_free_distance_selected.csv")
    cons['volume'].to_csv(f"{dest}/{ctype}_free_volume_selected.csv")
    get_consumed(ctype)['volume'].to_csv(f"{dest}/{ctype}_consumed_volume_selected.csv")
    get_not_free(ctype)['volume'].to_csv(f"{dest}/{ctype}_notfree_volume_selected.csv")
    get_bound(ctype)['volume'].to_csv(f"{dest}/{ctype}_bound_volume_selected.csv")
    get_type(ctype)['volume'].to_csv(f"{dest}/{ctype}_both_volume_selected.csv")
