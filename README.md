# DNA-PAINT_analysis

Get performance metrics for your DNA-PAINT measurement. 

Associated publication: 

Please cite XXX if you use this software. 

## Requirements

A python environment must be created, for instance with Anaconda. It must satisfy the following dependencies: 

### Python Standard Library

- python 3.9.7
- itertools 
- math
- os
- time 
- configparser
- datetime
- multiprocessing

### Packages

- numpy 1.21.2
- scipy 1.7.1
- pandas 1.4.4
- matplotlib 3.4.3
- yaml 0.2.5
- h5py 3.7.0
- lmfit 1.0.3
- pandarallel 1.6.3

The software was tested on Mac OS 14.4.1

## Instructions

### Main.py

<b>First, define a Measurement object with the following parameters: </b>

- in_path: The path to the .hdf5 file containing your single binding sites (or picked NPCs) to be analyzed (string). The .hdf5 file must contain groups (i.e. be picked sites). 
- save_path: The folder where you want your output to be saved (string).
- saving_name: The unique identifier for all output files (string). 
- total_n_frames: The number of frames for the measurement (integer).

MeasurementName.Begin() will perform the calculations. 

MeasurementName.FileSaver() will save the results. 

<b>Next, define a Plotting object with the following parameters: </b>

- table_g: The table containing binding events, photon counts and SBR (calculated by MeasurementName.Begin()).
- table_k: The table containing kinetics (bright times, dark times, mean bright and dark times, time from last frame to end of measurement and ratio of time from last frame to end of measurement over mean dark time, calculated by MeasurementName.Begin()).
- show: Whether to display the plots (Boolean).
- save: Whether to save the plots (Boolean).
- save_path: The folder where you want your output plots to be saved (string).
- saving_name: The unique identifier for all output plots (string).
- total_n_frames: The number of frames for the measurement (integer).

MeasurementPlotName.Plot_photons() will plot photon distributions. 

MeasurementPlotName.Plot_bg() will plot the background photon counts. 

MeasurementPlotName.Plot_sbr() will plot the signal to background ratio. 

MeasurementPlotName.Plot_tb() will plot the bright times.

MeasurementPlotName.Plot_td() will plot the dark times. 

MeasurementPlotName.Plot_r() will plot the ratios of time from last frame to end of measurement over mean dark time for all picked sites. 

MeasurementPlotName.Plot_locs() will plot localizations over time. 

MeasurementPlotName.SaveAllResults() will save the plots. 

### Count_docking_sites.py

This file counts the number of clusters (i.e. docking sites) within picks. This is used to identify barcode DNA origami. 

User input is required at the bottom of the file:

Define a Measurement object with a path to the .hdf5 file containing picked and clustered data. The file needs to contain "group" as well as "group_input" columns. 

MeasurementName.Import() will import the file. 

MeasurementName.CountSiteNumber() will count the number of clusters in each pick (i.e. DNA origami)

MeasurementName.SaveCenterYaml(integer, "string") will save the pick locations for all picks containing "integer" number of clusters with the filename appended with "string". 

The output is a .yaml file that can be loaded into Picasso render as load pick regions. 

### Unspecific_binding_NPC.py

This file quantifies the relative unspecific binding in NPC measurements. 

User input is required at the bottom of the file:

Call SBR_comparison("a","b","c","d", destination = "e") with five input parameters:

- a: Path to the .hdf5 file containing NPC picks in the reference imaging round (e.g. Cy3B), string.
- b: Path to the .hdf5 file containing cytoplasm picks in the reference imaging round (e.g. Cy3B), string.
- c: Path to the .hdf5 file containing NPC picks in the dye-of-interest imaging round, string.
- d: Path to the .hdf5 file containing cytoplasm picks in the dye-of-interest imaging round, string.
- e: Path to the folder where the results .txt file should be saved, string. 
