########################################################
#########           Philipp R. Steen           #########
#########             Jungmann Lab             #########
######### Max Planck Institute of Biochemistry #########
#########     Ludwig Maximilian University     #########
#########                 2024                 #########
########################################################

import time
from main_functions import *
#from extra_functions import *

core_params = {
    "path":"/.../sbs.hdf5",
    "save_path":"/.../savefolder",
    'save_code':"name",
    "ignore_dark": 1,
    "plot_title": "plot title",
    "origami": True,
    "colorcode": "green", #green, red, blue
    "number_sbs": 1,
    "n_frames": 20000, 
    "concentration_pM": 500,
    "exposure_time": 0.1
}
fulltable = Import(core_params["path"])

### Calculations

t1 = time.time()

enhanced_df = EnhanceDataFrame(fulltable, core_params)
bindingevent_info_df = Build_df_bindingevents(enhanced_df)
group_info_df = Build_df_groups(bindingevent_info_df, core_params["n_frames"])
all_first, all_center, all_last = AllPhotons(enhanced_df)
all_center = SBR(all_center)
edge_photons = np.concatenate([np.asarray(all_first["first_photons"]), np.asarray(all_last["last_photons"])])
center_photons = np.asarray(all_center["center_photons"])

FileSaver(enhanced_df, "_enhanced_df", core_params)
FileSaver(bindingevent_info_df, "_binding_event_info", core_params)
FileSaver(group_info_df, "_group_info", core_params)
FileSaver(all_center, "_all_center_photons", core_params)

t2 = time.time()
print("Time elapsed: ", np.round(t2-t1, 2), " seconds")

### Plotting results

save = True

plotting_params = {
    #Binding site destruction fit type ("bi" for biexponential fit, anything else for exponential fit)
    #'exp_fit_type' : "x",
    #Plot dark times
    'dark_range' : [0,250],
    'dark_cutoff' : [20,6000],
    'dark_binwidth' : 10,
    #Plot bright times
    'bright_range' : [0,.6],
    'bright_cutoff' : [.01,1000],
    'bright_binwidth' : 0.02,
    #Initial fit parameters for photon histogram
    #'PhotoPlot_fit_params' : [200, 3000, 500, 200, 10000, 1500],
    #Plot photon histogram
    #'bin_width_photons' : 1000,
    #'fit_style_photons' : "single",
    #'x_factor_photons' : 2.6,
    #Plot background over time
    'average_frames' : 200,
    #Plot background
    'binwidth_bg' : 20,
    'plot_range_bg' : [0,1200],
    #Plot signal to background ratio
    'binwidth_sbr' : 1,
    'x_factor_sbr' : 2.0
}

fig, ax = ImprovedSiteDestruction(group_info_df, core_params, "any", save)
LocPlot2(fulltable, core_params, plotting_params, NumberOfGroups(fulltable), save)
PlotAllKinetics(group_info_df, plotting_params["dark_range"], plotting_params["dark_cutoff"], "dark", plotting_params["dark_binwidth"], [core_params["plot_title"]+"\nDark times", "Dark times (s)", "Number of binding sites"], core_params, save)
PlotAllKinetics(group_info_df, plotting_params["bright_range"], plotting_params["bright_cutoff"], "bright", plotting_params["bright_binwidth"], [core_params["plot_title"]+"\nBright times", "Bright times (s)", "Number of binding sites"], core_params, save)
SimplePhotons(center_photons, core_params, 20, 3, save)
LocPlot(fulltable, core_params, NumberOfGroups(fulltable), save)
BackGroundPlot(fulltable, core_params, plotting_params["binwidth_bg"], plotting_params["plot_range_bg"], save)
PlotSBR(all_center, core_params, plotting_params["binwidth_sbr"], plotting_params["x_factor_sbr"], save)

print("All plots displayed successfully!")