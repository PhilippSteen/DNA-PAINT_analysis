########################################################
#########           Philipp R. Steen           #########
#########             Jungmann Lab             #########
######### Max Planck Institute of Biochemistry #########
#########     Ludwig Maximilian University     #########
#########                 2024                 #########
########################################################

from core_functions import *
from plotting_functions import *
import time

t1 = time.time()

Test1 = Measurement(in_path="/.../SBS.hdf5",
                    save_path = "/.../results",
                    saving_name = "Sequence_Name_Buffer",
                    total_n_frames = 20000)
Test1.Begin()
Test1.FileSaver()

t2 = time.time()
print("Time elapsed during analysis (incl. file saving): ", np.round(t2-t1, 2), " seconds")

t3 = time.time()

Test1b = Plotting(table_g = Test1.table_g,
                 table_k = Test1.table_k,
                 show = False,
                 save = True,
                 save_path = "/.../results",
                 saving_name = "Sequence_Name_Buffer",
                 total_n_frames = 20000)

Test1b.Plot_photons()
Test1b.Plot_bg()
Test1b.Plot_sbr()
Test1b.Plot_tb()
Test1b.Plot_td()
Test1b.Plot_r()
Test1b.Plot_locs()
Test1b.SaveAllResults()

t4 = time.time()
print("Time elapsed during plotting / evaluation (incl. file saving): ", np.round(t4-t3, 2), " seconds")