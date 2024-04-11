########################################################
#########           Philipp R. Steen           #########
#########             Jungmann Lab             #########
######### Max Planck Institute of Biochemistry #########
#########     Ludwig Maximilian University     #########
#########                 2024                 #########
########################################################

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import h5py
import configparser
from datetime import datetime
from scipy.optimize import curve_fit
from lmfit import Model
from pandarallel import pandarallel
import itertools
from scipy.signal import savgol_filter
import os

def Import(input_path):
    """Imports clustered .hdf5 from Picasso. Requires group info."""
    fulltable = pd.read_hdf(input_path, key = 'locs')
    fulltable.sort_values(by=['group', 'frame'])
    return(fulltable)

def DetermineBursts(group, core_params):
    """Determines binding events and associated photon counts"""
    event_number = 0
    previous_frame = -10000
    indices = [] #Index the binding events
    index = [1]
    binding_times = [] #Binding time (in frames) of each binding event
    bursts = [] #List the frame numbers, clustered by binding event
    burst = []
    photo_bursts = [] #List the photon counts, clustered by binding event
    photo_burst = []
    first_photons = []
    center_photons = []
    last_photons = []
    previous_last_photons = -1
    previous_first_photons = group["photons"].iloc[0]
    current_index = -1
    for i, row in group.iterrows():
        frame = row["frame"]
        photons = row["photons"]
        if frame - previous_frame <= core_params["ignore_dark"]:
            index.append(current_index)
            burst.append(frame)
            photo_burst.append(photons)
            previous_last_photons = photons
        else:
            event_number,previous_frame,indices,index,binding_times,bursts,burst,photo_bursts,photo_burst,first_photons,center_photons,last_photons,previous_last_photons,previous_first_photons,current_index = BasicCalc(frame, photons, event_number,previous_frame,indices,index,binding_times,bursts,burst,photo_bursts,photo_burst,first_photons,center_photons,last_photons,previous_last_photons,previous_first_photons,current_index)
        previous_frame = frame
    event_number,previous_frame,indices,index,binding_times,bursts,burst,photo_bursts,photo_burst,first_photons,center_photons,last_photons,previous_last_photons,previous_first_photons,current_index = BasicCalc(frame, photons, event_number,previous_frame,indices,index,binding_times,bursts,burst,photo_bursts,photo_burst,first_photons,center_photons,last_photons,previous_last_photons,previous_first_photons,current_index)
    return(indices[1:], bursts[1:], photo_bursts[1:], binding_times[1:], first_photons[1:], center_photons[1:], last_photons[1:])

def BasicCalc(frame, photons, event_number,previous_frame,indices,index,binding_times,bursts,burst,photo_bursts,photo_burst,first_photons,center_photons,last_photons,previous_last_photons,previous_first_photons,current_index):
    """Performs additional calculations for DetermineBursts"""
    event_length = len(index)
    binding_times.append([event_length]*event_length)
    current_index += 1
    indices.append(index)
    index = []
    index.append(current_index)    
    bursts.append(burst)
    burst = []
    burst.append(frame)
    if event_length == 1:
        first_photons.append([previous_first_photons])
        center_photons.append([-1])
        last_photons.append([-1])
    elif event_length == 2:
        first_photons.append([previous_first_photons,-1])
        center_photons.append([-1,-1])
        last_photons.append([-1, previous_last_photons])
    else:
        first_photons.append([previous_first_photons, *(event_length-1)*[-1]])
        center_photons.append([-1, *photo_burst[1:-1], -1])
        last_photons.append([*(event_length-1)*[-1], previous_last_photons])
    photo_bursts.append(photo_burst)
    photo_burst = []
    photo_burst.append(photons)
    previous_first_photons = photons
    return(event_number,previous_frame,indices,index,binding_times,bursts,burst,photo_bursts,photo_burst,first_photons,center_photons,last_photons,previous_last_photons,previous_first_photons,current_index)

def EnhanceDataFrame(df, core_params): 
    """Generates dataframe with additional columns (binding events, binding times, photon info) for further use"""
    enhanced_df = pd.DataFrame(columns = df.keys())
    group_ids = df["group"].unique()
    pd.options.mode.chained_assignment = None
    for i in group_ids:
        group = df[df["group"] == i]
        indices, bursts, photo_bursts, binding_times, first_photons, center_photons, last_photons = DetermineBursts(group, core_params)
        group["binding_event"] = list(itertools.chain.from_iterable(indices))
        group["binding_times"] = list(itertools.chain.from_iterable(binding_times))
        group["first_photons"] = list(itertools.chain.from_iterable(first_photons))
        group["center_photons"] = list(itertools.chain.from_iterable(center_photons))
        group["last_photons"] = list(itertools.chain.from_iterable(last_photons))
        enhanced_df = pd.concat([enhanced_df, group], ignore_index=True)
    pd.options.mode.chained_assignment = 'warn'
    return(enhanced_df)

def AllPhotons(enhanced_df):
    """Returns photon counts"""
    all_first = enhanced_df[enhanced_df["first_photons"] >-1]
    all_center = enhanced_df[enhanced_df["center_photons"] >-1]
    all_last = enhanced_df[enhanced_df["last_photons"] >-1]
    return(all_first, all_center, all_last)

def Build_df_center_frames_only(enhanced_df):
    """Builds dataframe with center frames only"""
    center_only = enhanced_df[enhanced_df["center_photons"] > 0]
    return(center_only)

def PicassoSaveHdf5(center_only, core_params):
    df = center_only.astype(np.float64)
    df['frame'] = df['frame'].astype(int)
    df['group'] = df['group'].astype(int)
    labels = list(df.keys())
    df_picasso = df.reindex(columns=labels, fill_value=1)
    df_picasso = df_picasso.reset_index()
    df_picasso = df_picasso.drop(columns=['index'])
    locs = df_picasso.to_records(index = False)
    hf = h5py.File(core_params["save_path"] + "/center_f_only", 'w')
    hf.create_dataset('locs', data=locs)
    hf.close()

def Build_df_bindingevents(enhanced_df): 
    """Builds dataframe with bright and dark times for each group and binding event"""
    xy_df = enhanced_df.groupby(['group', 'binding_event']).mean() #XY position of each binding event is found via mean
    bindingevent_info_df = xy_df.filter(["x", "y"], axis=1)
    first_frames_df = enhanced_df.groupby(['group', 'binding_event']).first()
    bindingevent_info_df["start_frame"] = first_frames_df["frame"]
    brights_df = enhanced_df.groupby(['group', 'binding_event']).first() #Bright time of each binding event is already known, info in "binding_times"
    bindingevent_info_df["bright_time"] = brights_df["binding_times"]
    darks_df = brights_df.diff() #Dark time of each binding event is found via differences between first frames of binding events, info in "frame"
    darks_df.loc(axis=0)[:, [0.0]] = np.nan #The "first" dark time entry is not valid (compares with a different spot)
    bindingevent_info_df["dark_time"] = darks_df["frame"]
    n_locs = enhanced_df.groupby(['group', 'binding_event']).size()
    bindingevent_info_df["n_locs"] = n_locs
    return(bindingevent_info_df)

def Build_df_groups(bindingevent_info_df, n_frames): 
    """Builds dataframe with mean bright and mean dark times for each group as well as x and y locations. Additionally calculates the ratio of the final dark segment over the mean dark time."""
    indices, darks_mean, darks_fit = Kinetics(bindingevent_info_df, "dark_time")
    indices2, brights_mean, brights_fit = Kinetics(bindingevent_info_df, "bright_time")
    if indices != indices2:
        print("Indices do not match")
        return()
    group_info_df = pd.DataFrame()
    group_info_df["group"] = indices
    group_info_df = group_info_df.set_index('group')
    xy_df = bindingevent_info_df.groupby("group").mean()
    group_info_df["x"] = xy_df["x"]
    group_info_df["y"] = xy_df["y"]
    group_info_df["bright_times_mean"] = brights_mean
    group_info_df["bright_times_fit"] = brights_fit
    group_info_df["dark_times_mean"] = darks_mean
    group_info_df["dark_times_fit"] = darks_fit
    n_locs = bindingevent_info_df.groupby('group').sum()
    group_info_df["n_locs"] = n_locs["n_locs"]
    n_binding_events = bindingevent_info_df.groupby(['group']).size()
    group_info_df["n_binding_events"] = n_binding_events
    last_frame_per_event = bindingevent_info_df["start_frame"] + bindingevent_info_df["bright_time"]
    last_frame = last_frame_per_event.groupby(['group']).max()
    group_info_df["last_frame"] = last_frame
    ratio = (n_frames-group_info_df["last_frame"])/group_info_df["dark_times_fit"]
    group_info_df["end_ratio"] = ratio
    return(group_info_df)

def MeanMedian(ax, darks_mean, darks_fit, bright_or_dark, colors):
    """Calculates the mean and median of the mean dark and bright times"""
    darks_mean_mean = np.mean(darks_mean)
    darks_mean_median = np.median(darks_mean)
    darks_fit_mean = np.mean(darks_fit)
    darks_fit_median = np.median(darks_fit)
    ax.axvline(x=darks_fit_mean, color = colors[1], label = r"$\tau_{"+bright_or_dark+", mean}$ = "+str(round(darks_fit_mean,2))+" s")
    ax.axvline(x=darks_fit_median, color = colors[2], label = r"$\tau_{"+bright_or_dark+", median}$ = "+str(round(darks_fit_median,2))+" s")
    return(darks_mean_mean, darks_mean_median, darks_fit_mean, darks_fit_median)

def PlotAllKinetics(group_info_df, plot_range, cutoff_range, bright_or_dark, binwidth, text, core_params, save):
    """Plots bright or dark times"""
    mpl.style.use('seaborn-poster')
    fig, ax = plt.subplots(1, figsize = (8, 5))
    fig.tight_layout()
    if bright_or_dark == "dark":
        darks_mean = group_info_df["dark_times_mean"]
        darks_fit = group_info_df["dark_times_fit"]
        colors = ["blue", "darkblue", "indigo"]
    elif bright_or_dark == "bright":
        darks_mean = group_info_df["bright_times_mean"]
        darks_fit = group_info_df["bright_times_fit"]
        colors = ["orange", "darkorange", "goldenrod"]
    else:
        print("Invalid specification of bright or dark")
        return()
    binwidth = binwidth
    min_cutoff = cutoff_range[0]
    max_cutoff = cutoff_range[1]
    darks_mean = np.multiply(darks_mean,core_params["exposure_time"])
    darks_fit = np.multiply(darks_fit,core_params["exposure_time"])
    filter_mean = darks_mean[(darks_mean >= min_cutoff) & (darks_mean <= max_cutoff)]
    filter_fit = darks_fit[(darks_fit >= min_cutoff) & (darks_fit <= max_cutoff)]
    ax.set_title(text[0])
    ax.set_xlabel(text[1])
    ax.set_ylabel(text[2])
    filter_mean_std = np.std(filter_mean)
    filter_fit_std = np.std(filter_fit)
    ax = plt.gca()
    if core_params["number_sbs"] != 1:
        text = str(len(filter_fit)*core_params["number_sbs"])+" binding sites\n"+str(len(filter_fit))+" DNA origami"
    elif core_params["number_sbs"] == 1:
        text = str(len(filter_fit))+" binding sites"
    plt.text(0.03, 0.03, text, ha='left', va='bottom', fontsize = 14, transform=ax.transAxes)
    darks_mean_mean, darks_mean_median, darks_fit_mean, darks_fit_median = MeanMedian(ax, filter_mean, filter_fit, bright_or_dark, colors)
    entries, bins, patches = ax.hist(filter_fit, bins = np.arange(0, max(filter_fit) + binwidth, binwidth), alpha = 0.5, color = colors[0], edgecolor=colors[1], linewidth=1, label = "Mean "+bright_or_dark+" times\nSTD = "+str(np.round(filter_fit_std, 3))+" s")
    plt.ylim([0, max(entries)*1.1])
    ax.set_xlim(plot_range)

    ax.legend()
    if save==True:
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_"+bright_or_dark+"_kinetics.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'darks/brights_mean_mean': str(darks_mean_mean),
            'darks/brights_mean_median': str(darks_mean_median),
            'darks/brights_mean_std': str(filter_mean_std),
            'darks/brights_fit_mean': str(darks_fit_mean),
            'darks/brights_fit_median': str(darks_fit_median),
            'darks/brights_fit_std': str(filter_fit_std),
            'min_cutoff': min_cutoff,
            'max_cutoff': max_cutoff,
            'binwidth': binwidth,
            'file_name': bright_or_dark+"_kinetics.pdf"}
        with open(os.path.join(core_params["save_path"], core_params["save_code"]+"_"+bright_or_dark+"_kinetics_params.txt"), 'w') as configfile:
            config.write(configfile)
    plt.show()

def Kinetics(bindingevent_info_df, bright_or_dark):
    """Calculates bright or dark times, calls CDF_kinetics"""
    indices = []
    ks_mean = []
    ks_fit = []
    group_ids = bindingevent_info_df.index.unique(0)
    for i in group_ids:
        if bright_or_dark == "dark_time":
            group = bindingevent_info_df.loc(axis=0)[i, 1:] #The first entry is always NaN for dark times
        elif bright_or_dark == "bright_time":
            group = bindingevent_info_df.loc(axis=0)[i, :]
        else:
            print("Invalid Entry")
            return()
        k = np.asarray(group[bright_or_dark])
        k_mean = np.mean(k)
        k_fit = CDF_kinetics(k)
        ks_mean.append(k_mean)
        ks_fit.append(k_fit)
        indices.append(i)
    ks_mean = np.asarray(ks_mean)
    ks_fit = np.asarray(ks_fit)
    return(indices, ks_mean, ks_fit)

def expfunc(t,tau): 
    """Used for fitting CDF_kinetics"""
    return((1 - np.exp(-(t/tau)))) 

def PlotKineticsFitting():
    """Prepares the plotting for CDF_kinetics"""
    fig, ax = plt.subplots(1, figsize = (8,6))
    fig.tight_layout()
    ax.set_xscale('log')
    ax.set_xlabel("Dark time (s)")
    ax.set_ylabel("Probability")
    return(fig, ax)

def PlotALine(fig, ax, dark_sorted, tau_d, p): 
    """Plotting function for CDF_kinetics"""
    plottingaxis = np.linspace(0, dark_sorted[-1], 5000)
    ax.plot(plottingaxis, expfunc(plottingaxis, tau_d), linewidth = 0.8, color = 'darkorange')
    ax.axvline(x=tau_d, linewidth = 0.6, color = 'purple')
    ax.plot(dark_sorted, p, linewidth = "0.5", color = "black")
    return(ax)

def CDF_kinetics(dark): 
    """calculates mean bright or dark times based on an exponential CDF fit"""
    plot_or_not = False
    if plot_or_not:
        fig, ax = PlotKineticsFitting()
    if len(dark) >= 2:
        dark_sorted = np.sort(dark)
        p = 1. * np.arange(len(dark_sorted)) / (len(dark_sorted) - 1)
        d_model = Model(expfunc)
        try:
            d_result = d_model.fit(p, t=dark_sorted, tau=200)
            tau_d = d_result.params['tau'].value
        except:
            tau_d = 0
        
        if plot_or_not:
            ax = PlotALine(fig, ax, dark_sorted, tau_d, p)
    else:
        tau_d=0
    if plot_or_not:
        plt.show()
    return(tau_d)

def NumberOfGroups(fulltable): 
    """The number of groups (picks) in the dataframe"""
    return(fulltable["group"].nunique())

def expfunc1(x,a1,m1):
    return a1*np.exp(-x*(1/m1))

def expfunc2(x, a1, m1, a2, m2):
    return a1*np.exp(-x*(1/m1)) + a2*np.exp(-x*(1/m2))

def ImprovedSiteDestruction(group_info_df, core_params, which_a, save):
    non_inf_sites = group_info_df[group_info_df["end_ratio"]!=np.inf]
    non_zero_sites = non_inf_sites[non_inf_sites["end_ratio"]>0]
    binwidth=0.5
    hist, bin_edges = np.histogram(non_zero_sites["end_ratio"], bins = np.arange(min(non_zero_sites["end_ratio"]), max(non_zero_sites["end_ratio"]) + binwidth, binwidth))
    bin_middles = bin_edges+(0.5*binwidth)
    xvals, yvals = bin_middles[:-1], hist

    popt_1e, pcov_1e = curve_fit(expfunc1, xvals, yvals, p0=[100, 1])
    natural_portion_over_four_m_1e = ((popt_1e[0]*max(popt_1e[1], 1))/np.exp(4))/(popt_1e[0]*popt_1e[1])
    cutter = 4*max(popt_1e[1], 1)
    apparently_destroyed_sites = non_zero_sites[non_zero_sites["end_ratio"]>=cutter]
    apparently_good_sites = non_zero_sites[non_zero_sites["end_ratio"]<cutter]
    ratio_of_ratios = len(apparently_destroyed_sites)/(len(apparently_good_sites)+len(apparently_destroyed_sites))
    final_ratio = ratio_of_ratios-natural_portion_over_four_m_1e

    print("sigma is ", popt_1e[1], "\n sigma used ", max(popt_1e[1], 1))
    print("portion of integral over 4 sigma ", natural_portion_over_four_m_1e, "\n ratio of destroyed sites without correction ", ratio_of_ratios, "\n ratio of destroyed sites with correction", final_ratio)
    mpl.style.use('seaborn-poster')
    fig, ax = plt.subplots(1, figsize = (8, 5))
    fig.tight_layout()

    binwidth_display = 0.5
    n, bins, patches = ax.hist(non_zero_sites["end_ratio"], bins = np.arange(min(non_zero_sites["end_ratio"]), max(non_zero_sites["end_ratio"]) + binwidth_display, binwidth_display), color = "grey")
    ax.plot(xvals, expfunc1(xvals, *popt_1e), color = "black", linewidth = 0.5)

    ax.set_xlim([0, 10])

    if save==True:
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_binding_site_destruction_new.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'sigma_used_1fit': str(cutter/4),
            "portion of integral over 4 sigma ":  str(natural_portion_over_four_m_1e),
            "ratio of destroyed sites without correction ":  str(ratio_of_ratios),
            "ratio of destroyed sites with correction":  str(final_ratio),
            'file_name': '_binding_site_destruction_new.pdf'}
        with open(os.path.join(core_params["save_path"], core_params["save_code"]+"_binding_site_destruction_new.txt"), 'w') as configfile:
            config.write(configfile)
    plt.show()
    return(fig, ax)

def GetHistColor(colorcode):
    """Defines the color of the photon histogram"""
    if colorcode == "green":
        bincolor = "lightgreen"
        linecolor = "darkgreen"
    elif colorcode == "blue":
        bincolor = "skyblue"
        linecolor = "darkblue"
    elif colorcode == "red":
        bincolor = "salmon"
        linecolor = "darkred"
    else:
        bincolor = "black"
        linecolor = "black"
    return(bincolor, linecolor)

def SimplePhotons(all_center, core_params, n_bins, x_factor, save):
    all_center = np.asarray(all_center)
    center_mean = np.mean(all_center)
    display_width = int(x_factor*center_mean)
    binwidth = int(display_width/n_bins)
    bins=range(0, display_width + binwidth, binwidth)
    bincolor, linecolor = GetHistColor(core_params["colorcode"])
    
    mpl.style.use('seaborn-poster')
    fig, ax = plt.subplots(1, figsize = (8, 8))
    fig.tight_layout()
    entries, bins, patches = ax.hist(all_center, bins=bins, density=True, color = bincolor, edgecolor=linecolor, linewidth=1, alpha = 0.6, label = "Mean photons:\n"+str(int(np.round(center_mean))))
    ax.tick_params(axis="both", direction="in")
    ax.set_xbound(0,display_width)

    ax.legend()

    if save==True:
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_photons_clean.pdf"), bbox_inches="tight", pad_inches=0.2)
    
    return(center_mean)

def LocPlot(df, core_params, origami_number, save):
    """Plots localizations over time"""
    mpl.style.use('seaborn-poster')
    fig = plt.figure(figsize=(8, 5))
    fig.tight_layout()
    
    n_bins = 20
    
    n, bins, patches = plt.hist(df["frame"], bins = n_bins, color = "orange", edgecolor='darkorange', linewidth=2, alpha = 0.6)

    n_frames = max(core_params["n_frames"], float(df["frame"].max()+1))
    upto_20 = df[df["frame"]<=0.2*n_frames]
    over_80 = df[df["frame"]>=0.8*n_frames]
    upto_20_len = upto_20["frame"].size
    over_80_len = over_80["frame"].size

    avg_start = upto_20_len/(0.2*n_bins)

    avg_end = over_80_len/(0.2*n_bins)
    
    plt.ylim([0, max(n)*1.15])

    drop = (1-avg_end/avg_start)*100
    
    plt.axhline(y = avg_start, color = "black", linewidth = 0.8)
    plt.axhline(y = avg_end, color = "black", linewidth = 0.8, label = "Change: "+str(np.round(-drop, 1))+"%")
    
    ax = plt.gca()
    plt.text(0.03, 0.97, str(origami_number*core_params["number_sbs"])+" binding sites", ha='left', va='top', fontsize = 14, transform=ax.transAxes)
    
    plt.title(core_params["plot_title"]+"\nLocalizations over time")
    #plt.xlabel("Time (s)")
    plt.ylabel("Localizations")
    
    ax.tick_params(axis="both", direction="in")
    ax.set_xticks(np.arange(0, core_params["n_frames"]+1, step=(core_params["n_frames"]/4)))
    #fig.canvas.draw()
    #labels = [item.get_text() for item in ax.get_xticklabels()]
    #labels = (int(x) * core_params["exposure_time"] for x in labels) #change x labels from frames to seconds
    #labels = (int(x) for x in labels)
    #ax.set_xticklabels(labels)
    plt.xlabel("Frames")
    
    ax.ticklabel_format(axis='y', style='', scilimits=(0,0))
    
    plt.legend()
    
    if save==True:
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_frames.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'avg_start': str(avg_start),
            'avg_end': str(avg_end),
            'n_bins': str(n_bins),
            'n_frames': str(n_frames),
            'file_name': "frames.pdf"}
        with open(os.path.join(core_params["save_path"], core_params["save_code"]+"_frames_params.txt"), 'w') as configfile:
            config.write(configfile)
        
    plt.show()

def SogSegments(segments):
    seg1 = int(0.5*len(segments)+1)
    if (seg1%2) == 0:
        seg1 = int(0.5*len(segments))
    return(seg1)
    
def LocPlot2(df, core_params, plotting_params, origami_number, save):
    """Plots localizations over time"""
    mpl.style.use('seaborn-poster')
    fig = plt.figure(figsize=(8, 5))
    fig.tight_layout()
    plt.title(core_params["plot_title"]+"\nLocalizations over time")
    df = df[df["frame"]>0]
    diff = 200
    timestep_s = int(diff*core_params["exposure_time"])
    segments = np.arange(0, df["frame"].max(), diff)
    xvals = segments+0.5*diff
    xvals_seconds = xvals*core_params["exposure_time"]
    simps = []
    for step in segments:
        simp = df[(df["frame"]>=step) & (df["frame"]<(step+diff))]
        simpl = len(simp)
        simps.append(simpl)
    plt.plot(xvals_seconds, simps, color = "orange", linewidth = 1, label = "Number of localizations\nper "+str(timestep_s)+" seconds")
    plt.ylim(0,max(simps)+0.1*max(simps))
    
    seg1 = SogSegments(segments)    
    yhat = savgol_filter(simps, seg1, 3)
    plt.plot(xvals_seconds, yhat, color = "orange", linewidth = 2, label = "Smoothed n. loc.")
    ax = plt.gca()
    ax.ticklabel_format(axis='y', style='', scilimits=(0,0))
    plt.text(0.03, 0.97, str(origami_number*core_params["number_sbs"])+" binding sites", ha='left', va='top', fontsize = 14, transform=ax.transAxes)
    plt.xlabel("Time (s)")
    plt.ylabel("Localizations per "+str(timestep_s)+" seconds")
    
    ##### Analysis part #####
    
    len_of_parts = int(len(segments) * 0.15) #15% of locs for first / last segment
    
    first_part = simps[:len_of_parts]
    first_mean = np.mean(first_part)
    
    windows = np.convolve(simps,np.ones(len_of_parts,dtype=int),'valid')
    max_window = np.argmax(windows)
    biggest_part = simps[max_window:max_window+len_of_parts]
    biggest_mean = np.mean(biggest_part)
    
    last_part = simps[-len_of_parts:]
    last_mean = np.mean(last_part)
    
    plt.axvspan((max_window)*timestep_s, (max_window+len_of_parts)*timestep_s, alpha=0.05, color='red', label = "15% of measurement with\nmax. number of localizations")
    
    if (first_mean>last_mean): prefac=-1 
    else: prefac=1
    start_to_end = (1-(min(first_mean, last_mean) / max(first_mean, last_mean))) * prefac
    if (biggest_mean>last_mean): prefac=-1 
    else: prefac=1
    max_to_end = (1-(min(biggest_mean, last_mean) / max(biggest_mean, last_mean))) * prefac
    prefac=1
    start_to_max = (1-(min(first_mean, biggest_mean) / max(first_mean, biggest_mean))) * prefac
    
    start_to_end *=100
    max_to_end *=100
    start_to_max *=100
    
    print("change from start to end: \t", start_to_end)
    print("change from maximum to end: \t", max_to_end)
    print("change from start to maximum: \t", start_to_max)
    
    plt.plot([], [], ' ', label="Change from max. to end: "+str(np.round(max_to_end, 1))+"%")
    
    plt.legend()
    if save==True:
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_frames_n.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'avg_start': str(first_mean),
            'avg_max': str(biggest_mean),
            'avg_end': str(last_mean),
            'start_of_max_part': str((max_window)*timestep_s),
            'change_from_start_to_end': str(start_to_end),
            'change_from_maximum_to_end': str(max_to_end),
            'change_from_start_to_maximum': str(start_to_max),
            'segment_length_seconds': str(timestep_s),
            'file_name': "frames_n.pdf"}
        with open(os.path.join(core_params["save_path"], core_params["save_code"]+"_frames_n_params.txt"), 'w') as configfile:
            config.write(configfile)
    plt.show()
    
    
def BackGroundPlot(df, core_params, binwidth, display_range, save):
    """Plots the background photon count"""
    mpl.style.use('seaborn-poster')
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    
    bincolor, linecolor = GetHistColor(core_params["colorcode"])
    
    df = df[df["bg"]>0]
    number = df.shape[0]
    
    bins = np.arange(0, max(df["bg"]) + binwidth, binwidth)
    n, bins, patches = plt.hist(df["bg"], bins = bins, color = bincolor, edgecolor=linecolor, linewidth=1, alpha = 0.3, density=True)
    plt.ylim([0, max(n)*1.15])
    
    maxi = bins[n.argmax()] + binwidth/2
    plt.axvline(x=maxi, color = "black", linewidth = 1, label = "Hist. max: "+str(round(maxi,2)))
    mean = df["bg"].mean()
    plt.axvline(x=mean, color = linecolor, linewidth = 1, label = "Mean: "+str(round(mean,2)))
    
    plt.title(core_params["plot_title"]+"\nBackground photons counts")
    plt.xlabel("Background photons per pixel")
    plt.ylabel("Normalized counts")
    ax = plt.gca()
    plt.text(0.03, 0.97, str(number)+" localizations", ha='left', va='top', fontsize = 14, transform=ax.transAxes)
    
    ax.ticklabel_format(axis='y', style='', scilimits=(0,0))
    plt.xlim(display_range)
    
    plt.legend()
    
    if save==True:
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_background.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'mean': str(mean),
            'hist_max': str(maxi),
            'binwidth': str(binwidth),
            'n_localizations': str(number),
            'display_range': str(display_range),
            'file_name': "background.pdf"}
        with open(os.path.join(core_params["save_path"], core_params["save_code"]+"_background_params.txt"), 'w') as configfile:
            config.write(configfile)
            
    plt.show()

def two_d_gaus(x, y, sx, sy):
    """2D Gaussian"""
    return(np.exp(-((x**2 / (2*sx**2))+(y**2 / (2*sy**2)))))
    
def CalcMaxPhotonsPixel(row):
    """Calculates the photons collected over the 1x1 pixel central area of a given localization"""
    from scipy.special import erf
    import math
    N = row["center_photons"]
    sx = row["sx"]
    sy = row["sy"]
    bounds = 0.5
    result = 2*math.pi*sx*sy*erf(bounds/(np.sqrt(2)*sx))*erf(bounds/(np.sqrt(2)*sy))
    entire = 2*math.pi*sx*sy
    scale_factor = N/entire
    peak_pixel_value = scale_factor*result
    return(peak_pixel_value)
    
def SBR(all_center):
    """Calculates the signal to background ratio by dividing the CalcMaxPhotonsPixel by the background per pixel value"""
    import multiprocessing
    num_cpus = multiprocessing.cpu_count()
    if num_cpus > 30:
        pandarallel.initialize(nb_workers=30)
    else:
        pandarallel.initialize()
    all_center["peak_pixel_value"] = all_center.parallel_apply(CalcMaxPhotonsPixel, axis=1)
    all_center["sbr"] = (all_center["peak_pixel_value"]/all_center["bg"])
    return(all_center)

def PlotSBR(all_center, core_params, binwidth, x_factor, save):
    """Plots a histogram of the signal to background ratios"""
    mpl.style.use('seaborn-poster')
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    bincolor, linecolor = GetHistColor(core_params["colorcode"])
    df = all_center[all_center["sbr"]>0]
    df = df[df["sbr"]!=np.inf]
    number = df.shape[0]
    mean = df["sbr"].mean()
    bins = np.arange(0, max(df["sbr"]) + binwidth, binwidth)
    n, bins, patches = plt.hist(df["sbr"], bins = bins, color = bincolor, edgecolor=linecolor, linewidth=1, alpha = 0.3, density=True)
    plt.xlim([0, mean*x_factor])
    maxi = bins[n.argmax()] + binwidth/2
    plt.axvline(x=maxi, color = "black", linewidth = 1, label = "Hist. max: "+str(round(maxi,2)))
    plt.axvline(x=mean, color = linecolor, linewidth = 1, label = "Mean: "+str(round(mean,2)))
    plt.title(core_params["plot_title"]+"\nSignal to Background Ratio")
    plt.xlabel("Signal to Background Ratio")
    plt.ylabel("Normalized counts")
    ax = plt.gca()
    plt.text(0.03, 0.97, str(number)+" localizations", ha='left', va='top', fontsize = 14, transform=ax.transAxes)
    ax.ticklabel_format(axis='y', style='', scilimits=(0,0))
    plt.legend()
    if save==True:
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_sbr.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'mean': str(mean),
            'hist_max': str(maxi),
            'binwidth': str(binwidth),
            'n_localizations': str(number),
            'file_name': "sbr.pdf"}
        with open(os.path.join(core_params["save_path"], core_params["save_code"]+"_sbr_params.txt"), 'w') as configfile: 
            config.write(configfile)
    plt.show()
    
def YamlSaver(file_name):
    """Saves pick files for Picasso"""
    pick_diameter = 1.0
    def save_picks_1(name):
        yaml.default_flow_style = None
        picks = {"Centers": [[float(very_large_x[i]),float(very_large_y[i])] for i in range(0,len(very_large_y))],"Diameter": pick_diameter,"Shape": "Circle"}
        with open(str(name) + '_jup.yaml', "w") as f:
            yaml.dump(picks, f,default_flow_style=None)
    SaveEverything = save_picks_1(file_name)

def save_picks(x_coord, y_coord, file_name, pick_diameter = 1.0):
    """Saves pick files for Picasso"""
    yaml.default_flow_style = None
    picks = {"Centers": [[float(x_coord[i]),float(y_coord[i])] for i in range(0,len(y_coord))],"Diameter": pick_diameter,"Shape": "Circle"}
    with open(str(file_name) + '_jup.yaml', "w") as f:
        yaml.dump(picks, f,default_flow_style=None)

def FileSaver(df, file_name, core_params):
    """Saves a dataframe for later use"""
    df.to_csv(os.path.join(core_params["save_path"], core_params["save_code"]+file_name+".csv"), index = True)
    df.to_pickle(os.path.join(core_params["save_path"], core_params["save_code"]+file_name+".pkl"))
    return()