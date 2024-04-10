########################################################
#########           Philipp R. Steen           #########
#########             Jungmann Lab             #########
######### Max Planck Institute of Biochemistry #########
#########     Ludwig Maximilian University     #########
#########                 2024                 #########
########################################################

import numpy as np
import pandas as pd
import yaml
import time
import h5py

def picasso_hdf5(df, hdf5_fname, hdf5_oldname, path):
    """
    This function recieves a pandas data frame coming from a Picasso .hdf5
    file but that has been modified somehow (e.g. coordinates rotated, 
    groups edited, etc) and formats it to be compatible with Picasso. 
    It also creates the necessary .yaml file.
    
    It is meant to be used in a folder that contains the original Picasso
    .hdf5 file and the .yaml file.
    
    - df: pandas data frame object
    - hdf5_fname: the desired filename for the Picasso-compatible .hdf5 file
    - hdf5_oldname: name of the original Picasso file that was modified
    - path: the absolute path containing the path to the file's folder
    
    Note: include ".hdf5" in the file names
    Warning: the .yaml file is basically a copy of the old one. If important
    information should be added to the new .yaml file this function should be
    modified
    """
    labels = list(df.keys())
    df_picasso = df.reindex(columns=labels, fill_value=1)
    locs = df_picasso.to_records(index = False)

    # Saving data
    
    hf = h5py.File(path + hdf5_fname, 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()

    # YAML saver
    yaml_oldname = path + hdf5_oldname.replace('.hdf5', '.yaml')
    yaml_newname = path + hdf5_fname.replace('.hdf5', '.yaml')
    
    yaml_file_info = open(yaml_oldname, 'r')
    yaml_file_data = yaml_file_info.read()
    
    yaml_newfile = open(yaml_newname, 'w')
    yaml_newfile.write(yaml_file_data)
    yaml_newfile.close()   
    
    print('New Picasso-compatible .hdf5 file and .yaml file successfully created.')
    
def save_picks(x_coord, y_coord, file_name, pick_diameter = 1.0):
    yaml.default_flow_style = None
    picks = {"Centers": [[float(x_coord[i]),float(y_coord[i])] for i in range(0,len(y_coord))],"Diameter": pick_diameter,"Shape": "Circle"}
    with open(str(file_name) + '_jup.yaml', "w") as f:
        yaml.dump(picks, f,default_flow_style=None)

def CountI(group): 
#Counts the number of localizations per cluster
    number = group['group'].nunique()
    index = len(group)
    return(number, index)

class Measurement:
    def __init__(self,
                 path = ""):
        self.path = path
        self.current_fulltable = ""

    def Import(self):
    #Imports a Picasso hdf5
        fulltable = pd.read_hdf(self.path, key = 'locs')
        self.current_fulltable = fulltable.sort_values(by=['group', 'frame'])

    def CountSiteNumber(self): 
    #Counts the number of clusters per original group (i.e. per origami)
        fulltable = self.current_fulltable
        all_numbers = []
        for i in range(max(fulltable["group_input"])+1):
            now = fulltable[fulltable["group_input"] == (i) ]
            if not now.empty:
                Count, length = CountI(now)
                numbers = [Count] * length
                all_numbers = np.append(all_numbers, numbers)
        fulltable['sites_per_origami'] = all_numbers
        self.current_fulltable = fulltable
    
    def SaveCenterYaml(self, number, name):
    #Saves the center of all localizations comprising the original group
        fulltable = self.current_fulltable.loc[self.current_fulltable["sites_per_origami"] == number]
        fulltable.head()
        x_coord = []
        y_coord = []
        for i in range(max(fulltable["group_input"])+1):
            xy = fulltable[fulltable["group_input"] == (i)]
            if not xy.empty:
                x_coord.append(xy["x"].mean())
                y_coord.append(xy["y"].mean())
        x_coord = np.asarray(x_coord)
        y_coord = np.asarray(y_coord)
        
        path_parts = self.path.split(".")
        new_path = path_parts[:-1]+[name]+path_parts[-1:]
        final_path = "."
        final_path = final_path.join(new_path)
        save_picks(x_coord, y_coord, final_path)
            
    def PrintTable(self):
        print(self.current_fulltable.head())

###############################################################################################
###############################################################################################
###############################################################################################

t1 = time.time()
crosshairs_path = "/.../S3_align_1/eval/S3_align_1_MMStack_Pos0.ome_locs_clustered.hdf5"
five = Measurement(path = crosshairs_path)
five.Import()
five.CountSiteNumber()
five.SaveCenterYaml(5, "5_sbs")
t2 = time.time()
print("Time elapsed: ", np.round(t2-t1, 2), " seconds")