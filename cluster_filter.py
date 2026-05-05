import pandas as pd
import numpy as np

def ImportPicke(path):
    return(pd.read_pickle(path))

def stdev_of_events(events):
    start_times = [pair[0] for pair in events]
    return(np.std(start_times))

table_k = ImportPicke("/Volumes/pool-miblab1/users/steen/00_imaging/combi_paint/26_02_12_A549_EGFR-GFP_quenchedMP/w1c1_R4-Cy3B_300pM_30mW_1/results/table_k_R4_Cy3B_300pM_30mW.pkl")

table_k["stdevs"] = table_k["events"].apply(stdev_of_events)

filterfactor=0.2
n_frames=20000

stdgood = filterfactor*n_frames

table_k["stdgood"] = np.where(table_k["stdevs"] >= stdgood, 1, -1)

print(table_k.head())

counts = table_k["stdgood"].value_counts()
print("stdgood = 1:", counts.get(1, 0))
print("stdgood = -1:", counts.get(-1, 0))