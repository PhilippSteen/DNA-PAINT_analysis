########################################################
#########           Philipp R. Steen           #########
#########             Jungmann Lab             #########
######### Max Planck Institute of Biochemistry #########
#########     Ludwig Maximilian University     #########
#########              2024-2026               #########
########################################################

import math
import pandas as pd
import yaml

def Import(input_path):
    """Imports clustered .hdf5 from Picasso. Requires group info."""
    fulltable = pd.read_hdf(input_path, key = 'locs')
    fulltable.sort_values(by=['group', 'frame'])
    yaml_path = input_path[:-4]+"yaml"
    with open(yaml_path) as f:
        data = yaml.safe_load_all(f)
        for x in data:
            try: 
                #pick_diameter = x.get('Pick Diameter')
                pick_area = x.get('Area (um^2)')
            except:
                continue
    return(fulltable, pick_area) #return(fulltable, pick_diameter)

'''
def LocsPerSquareMicrometer(table, diameter, pixelsize = 0.13):
    """
    Calculates number of localizations per square micrometer.
    Requires picked data (nucleus, cytoplasm, membrane or slide) with >= 1 pick.
    Extracts the number of picks and adjusts the total area correspondingly. 
    
    Parameters
    ----------
    table : dataframe
        Dataframe containing all locs within pick(s)
    diameter : int
        The pick diameter (in pixels, set in Picasso)
    pixelsize : int (optional, default = 0.13)
        Pixel size in micrometers

    Returns
    -------
    locs_per_um_sq : float
        Number of localizations per square micrometer
    """

    number_of_picks = table["group"].nunique()
    area = (math.pi * ( (0.5*diameter) * (pixelsize) )**2)*number_of_picks
    locs = len(table.index)
    return(locs/area)
'''
def LocsPerSquareMicrometer(table, area):
    locs = len(table.index)
    return(locs/area)

def SBR_comparison(nup_cy3b, cyt_cy3b, nup_doi, cyt_doi, destination):
    lpsum_npc_cy3b = LocsPerSquareMicrometer(*Import(nup_cy3b))
    lpsum_cyt_cy3b = LocsPerSquareMicrometer(*Import(cyt_cy3b))
    lpsum_npc_doi = LocsPerSquareMicrometer(*Import(nup_doi))
    lpsum_cyt_doi = LocsPerSquareMicrometer(*Import(cyt_doi))

    SBR_Cy3b = lpsum_npc_cy3b / lpsum_cyt_cy3b
    SBR_DOI = lpsum_npc_doi / lpsum_cyt_doi

    Relative_SBR = SBR_DOI / SBR_Cy3b

    results_string =   ("Locs per square um, NPCs, Cy3b: \t" + str(lpsum_npc_cy3b) + "\n"+
                        "Locs per square um, cytoplasm, Cy3b: \t" + str(lpsum_cyt_cy3b) + "\n"+
                        "SBR, Cy3b: \t\t\t\t" + str(SBR_Cy3b) + "\n\n"+
                        "Locs per square um, NPCs, DOI: \t\t" + str(lpsum_npc_doi) + "\n"+
                        "Locs per square um, cytoplasm, DOI: \t" + str(lpsum_cyt_doi) + "\n"+
                        "SBR, DOI: \t\t\t\t" + str(SBR_DOI) + "\n\n"+
                        "Relative SBR value: \t\t\t" + str(Relative_SBR))
    
    with open(destination+'/stick_results.txt', 'w') as f:
        f.write(results_string)
    
    print(results_string)
    return(results_string)

###############################################################################################
###############################################################################################
###############################################################################################

SBR_comparison("/Volumes/pool-miblab1/users/kellerer/z_raw/260422_NUP_DyeAnalysis/R1_Atto643_100pM_40mW_1/dye_a/R1_Atto643_100pM_40mW_1_MMStack_Pos0.ome_locs_undrifted_NPC-picks.hdf5",
               "/Volumes/pool-miblab1/users/kellerer/z_raw/260422_NUP_DyeAnalysis/R1_Atto643_100pM_40mW_1/dye_a/R1_Atto643_100pM_40mW_1_MMStack_Pos0.ome_locs_undrifted_cytoplasm_picks.hdf5",
               "/Volumes/pool-miblab1/users/kellerer/z_raw/260422_NUP_DyeAnalysis/R1_Cy5_NDA_100pM_40mW_2/dye_a/R1_Cy5_NDA_100pM_40mW_2_MMStack_Pos0.ome_locs_render_NPC-picks.hdf5",
               "/Volumes/pool-miblab1/users/kellerer/z_raw/260422_NUP_DyeAnalysis/R1_Cy5_NDA_100pM_40mW_2/dye_a/R1_Cy5_NDA_100pM_40mW_2_MMStack_Pos0.ome_locs_render_cytoplasm_picks.hdf5",
               destination="/Volumes/pool-miblab1/users/kellerer/z_raw/260422_NUP_DyeAnalysis/R1_Cy5_NDA_100pM_40mW_2/dye_a/eval")

#SBR_comparison("/.../C3_Cy3b_100pM_adTx_20mW_NUP-PICKS.hdf5",
#               "/.../C3_Cy3b_100pM_adTx_20mW_CYTOPLASM-PICKS.hdf5",
#               "/.../C3_JF585_100pM_Tx_20mW_NUP-PICKS.hdf5",
#               "/.../C3_JF585_100pM_Tx_20mW_CYTOPLASM-PICKS.hdf5",
#               destination = "/.../C3_JF585_100pM_Tx_20mW_1/eval")