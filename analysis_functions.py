########################################################
#########           Philipp R. Steen           #########
#########             Jungmann Lab             #########
######### Max Planck Institute of Biochemistry #########
#########     Ludwig Maximilian University     #########
#########                 2022                 #########
########################################################

import numpy as np
import scipy as sp
import pandas as pd
from scipy import integrate
import matplotlib.pyplot as plt
import matplotlib as mpl
import yaml
import time
import math
import h5py
import configparser
from datetime import datetime
from scipy import stats
from scipy.stats import norm
from scipy.optimize import curve_fit
from lmfit import Model
from lmfit.models import ExponentialModel
from pandarallel import pandarallel
import itertools

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
    #ax.axvline(x=darks_mean_mean, color = "red", label = "mean mean: t = "+str(round(darks_mean_mean,2)))
    #ax.axvline(x=darks_mean_median, color = "orange", label = "mean median: t = "+str(round(darks_mean_median,2)))
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
    
    #if bright_or_dark == "dark":
    #    plt.text(0.03, 0.97, text, ha='left', va='top', fontsize = 14, transform=ax.transAxes)
    #elif bright_or_dark == "bright":
    #    plt.text(0.03, 0.03, text, ha='left', va='bottom', fontsize = 14, transform=ax.transAxes)
    plt.text(0.03, 0.03, text, ha='left', va='bottom', fontsize = 14, transform=ax.transAxes)
    
    darks_mean_mean, darks_mean_median, darks_fit_mean, darks_fit_median = MeanMedian(ax, filter_mean, filter_fit, bright_or_dark, colors)
    #ax.hist(filter_mean, bins = np.arange(0, max(filter_mean) + binwidth, binwidth), alpha = 0.5, color = "red", label = "mean, datapoints: "+str(len(filter_mean))+"\nSTD = "+str(np.round(filter_mean_std, 2)))
    #ax.hist(filter_fit, bins = np.arange(0, max(filter_fit) + binwidth, binwidth), alpha = 0.3, color = "blue", label = "CDF fit, datapoints: "+str(len(filter_fit))+"\nSTD = "+str(np.round(filter_fit_std, 2)))
    entries, bins, patches = ax.hist(filter_fit, bins = np.arange(0, max(filter_fit) + binwidth, binwidth), alpha = 0.5, color = colors[0], edgecolor=colors[1], linewidth=1, label = "Mean "+bright_or_dark+" times\nSTD = "+str(np.round(filter_fit_std, 3))+" s")
    plt.ylim([0, max(entries)*1.1])
    ax.set_xlim(plot_range)
    ax.legend()
    if save==True:
        plt.savefig(FigurePathFinder(core_params, bright_or_dark+"_kinetics.pdf"), bbox_inches="tight", pad_inches=0.2)
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
            'file_name': text[1]+"_kinetics.pdf"}
        with open(FigurePathFinder(core_params, bright_or_dark+"_kinetics_params.txt"), 'w') as configfile:
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
        #popt, pcov = curve_fit(expfunc, dark_sorted, p)
        
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

def FigurePathFinder(core_params, name): 
    """Finds the path to save the figures in based on the input path"""
    path = core_params["path"]
    path_parts = path.split(".")
    new_path = path_parts[:-1]+[name]
    final_path = "."
    final_path = final_path.join(new_path)
    return(final_path)

def NumberOfGroups(fulltable): 
    """The number of groups (picks) in the dataframe"""
    return(fulltable["group"].nunique())

def SiteDestructionAnalysis(group_info_df, path, plot, save): 
    """Checks for binding sites that appear to be destroyed over the course of the measurement"""
    non_inf_sites = group_info_df[group_info_df["end_ratio"]!=np.inf]
    non_zero_sites = non_inf_sites[non_inf_sites["end_ratio"]>0]
    ratio_np = np.asarray(non_zero_sites["end_ratio"])
    binwidth=1
    hist, bin_edges = np.histogram(non_zero_sites["end_ratio"], bins = np.arange(min(non_zero_sites["end_ratio"]), max(non_zero_sites["end_ratio"]) + binwidth, binwidth))
    bin_middles = bin_edges+(0.5*binwidth)
    xvals, yvals = bin_middles[:-1], hist
    model_e = ExponentialModel()
    params_e = model_e.make_params(amplitude=1, decay=1)
    result_e = model_e.fit(yvals, params_e, x=xvals)
    cutter = result_e.params["decay"].value * 4
    apparently_destroyed_sites = non_zero_sites[non_zero_sites["end_ratio"]>=cutter]
    apparently_good_sites = non_zero_sites[non_zero_sites["end_ratio"]<cutter]

    if plot == True:
        mpl.style.use('seaborn-poster')
        fig = plt.figure(figsize=(8, 8))
        fig.tight_layout()
        plt.hist(non_zero_sites["end_ratio"], bins = np.arange(min(non_zero_sites["end_ratio"]), max(non_zero_sites["end_ratio"]) + binwidth, binwidth), color = "grey")
        plt.plot(xvals, result_e.best_fit, color = "black")
        plt.axvline(x=cutter, color = "red")
        plt.xlim([0, 2*cutter])

    d_x = np.asarray(apparently_destroyed_sites["x"])
    d_y = np.asarray(apparently_destroyed_sites["y"])
    g_x = np.asarray(apparently_good_sites["x"])
    g_y = np.asarray(apparently_good_sites["y"])
    if save==True:
        save_picks(d_x, d_y, path+"/apparently_destroyed_sites")
        save_picks(g_x, g_y, path+"/apparently_good_sites")
    ratio_of_ratios = len(apparently_destroyed_sites)/(len(apparently_good_sites)+len(apparently_destroyed_sites))
    return(ratio_of_ratios)

def _2gaussian(x_array, amp1,cen1,sigma1, amp2,cen2,sigma2):
    """Double Gaussian for fitting"""
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2))) + \
            amp2*(1/(sigma2*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen2)/sigma2)**2)))

def _1gaussian(x_array, amp1,cen1,sigma1):
    """Gaussian for fitting"""
    return amp1*(1/(sigma1*(np.sqrt(2*np.pi))))*(np.exp((-1.0/2.0)*(((x_array-cen1)/sigma1)**2)))

def StrangeFit(bin_middles, center_entries, xax, p0, linecolor):
    """Performs double Gaussian fitting"""
    popt_2gauss, pcov_2gauss = sp.optimize.curve_fit(_2gaussian, bin_middles, center_entries, p0=p0)
    perr_2gauss = np.sqrt(np.diag(pcov_2gauss))
    pars_1 = popt_2gauss[0:3]
    pars_2 = popt_2gauss[3:6]
    gauss_peak_1 = _1gaussian(xax, *pars_1)
    gauss_peak_2 = _1gaussian(xax, *pars_2)
    plt.plot(xax, gauss_peak_1, linecolor, linewidth = 0.8, label = "Mean photons (1):\n"+str(int(np.round(pars_1[1]))))
    plt.plot(xax, gauss_peak_2, linecolor, linewidth = 0.8, label = "Mean photons (2):\n"+str(int(np.round(pars_2[1]))))
    return(pars_2)

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

def PhotoPlot(all_edge, all_center, core_params, origami_number, binwidth, fit_style, p0, x_factor, save):
    """Plots the photon count histogram"""
    all_edge = np.asarray(all_edge)
    all_center = np.asarray(all_center)
    
    mpl.style.use('seaborn-poster')
    bins=range(int(all_edge.min()), int(all_center.max()) + binwidth, binwidth)
    fig = plt.figure(figsize=(8, 8))
    fig.tight_layout()
    
    bincolor, linecolor = GetHistColor(core_params["colorcode"])

    edge_entries, edge_bins, edge_patches = plt.hist(all_edge, bins=bins, density=True, color = "grey", edgecolor='black', linewidth=1, alpha = 0.3, label = "First / last frame,\nN$_{loc}$="+str(len(all_edge)))

    center_entries, center_bins, center_patches = plt.hist(all_center, bins=bins, density=True, color = bincolor, edgecolor=linecolor, linewidth=1, alpha = 0.6, label = "\"Center\" frame(s),\nN$_{loc}$="+str(len(all_center)))
    
    bin_middles = 0.5 * (center_bins[1:] + center_bins[:-1])

    xax = np.linspace(0, max(all_center)+1000, 10000)

    if fit_style == "double":
        pars = StrangeFit(bin_middles, center_entries, xax, p0, linecolor)
        center = pars[1]
    elif fit_style == "single":
        popt1, pcov1 = sp.optimize.curve_fit(_1gaussian, bin_middles, center_entries, p0=p0[3:6])
        gauss_peak_3 = _1gaussian(xax, *popt1)
        center = popt1[1]
        plt.plot(xax, gauss_peak_3, linecolor, linewidth = 0.8, label = "Mean photons:\n"+str(int(np.round(center))))
    else:
        print("fit type not specified")
        return()

    ax = plt.gca()
    ax.tick_params(axis="both", direction="in")
    plt.text(0.03, 0.97, str(origami_number*core_params["number_sbs"])+" binding sites", ha='left', va='top', fontsize = 14, transform=ax.transAxes)
    
    #Add labels in preferred order
    handles, labels = plt.gca().get_legend_handles_labels()
    
    if fit_style == "double":
        order = [2,3,0,1]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 
    elif fit_style == "single":
        order = [1,2,0]
        plt.legend([handles[idx] for idx in order],[labels[idx] for idx in order]) 

    plt.title(core_params["plot_title"]+"\nPhoton counts")
    plt.xlim([0, x_factor*center])
    plt.ylim([0, max(max(center_entries), max(edge_entries))*1.15])
    plt.xlabel("Photons per localization-frame")
    plt.ylabel("Normalized counts")
    ax.ticklabel_format(axis='y', style='', scilimits=(0,0))
    
    if save==True:
        plt.savefig(FigurePathFinder(core_params, "photons.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'fit_used': fit_style,
            'fit_mean': str(center),
            'number_center_frames': str(len(all_center)),
            'number_edge_frames': str(len(all_edge)),
            'origami_number': str(origami_number),
            'binwidth': binwidth,
            'file_name': "photons.pdf"}
        with open(FigurePathFinder(core_params, "photons_params.txt"), 'w') as configfile:
            config.write(configfile)
    plt.show()
    return(center)

def LocPlot(df, core_params, origami_number, save):
    """Plots localizations over time"""
    mpl.style.use('seaborn-poster')
    fig = plt.figure(figsize=(8, 5))
    fig.tight_layout()
    
    n_bins = 20
    
    n, bins, patches = plt.hist(df["frame"], bins = n_bins, color = "orange", edgecolor='darkorange', linewidth=2, alpha = 0.6)
    
    n_frames = core_params["n_frames"]
    upto_15 = df[df["frame"]<=0.15*n_frames]
    over_85 = df[df["frame"]>=0.85*n_frames]
    upto_15_len = upto_15["frame"].size
    over_85_len = over_85["frame"].size

    avg_start = upto_15_len/(0.15*n_bins)
    avg_end = over_85_len/(0.15*n_bins)
    
    plt.ylim([0, max(n)*1.15])
    
    drop = (1-avg_end/avg_start)*100
    
    plt.axhline(y = avg_start, color = "black", linewidth = 0.8)
    plt.axhline(y = avg_end, color = "black", linewidth = 0.8, label = "Reduction: "+str(np.round(drop, 1))+"%")
    
    ax = plt.gca()
    plt.text(0.03, 0.97, str(origami_number*core_params["number_sbs"])+" binding sites", ha='left', va='top', fontsize = 14, transform=ax.transAxes)
    
    plt.title(core_params["plot_title"]+"\nLocalizations over time")
    plt.xlabel("Frame number")
    plt.ylabel("Localizations")
    
    ax.tick_params(axis="both", direction="in")
    ax.set_xticks(np.arange(0, 20001, step=5000))
    
    plt.legend()
    
    if save==True:
        plt.savefig(FigurePathFinder(core_params, "frames.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'avg_start': str(avg_start),
            'avg_end': str(avg_end),
            'n_bins': str(n_bins),
            'n_frames': str(n_frames),
            'file_name': "frames.pdf"}
        with open(FigurePathFinder(core_params, "frames_params.txt"), 'w') as configfile:
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
        plt.savefig(FigurePathFinder(core_params, "background.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'mean': str(mean),
            'hist_max': str(maxi),
            'binwidth': str(binwidth),
            'n_localizations': str(number),
            'display_range': str(display_range),
            'file_name': "background.pdf"}
        with open(FigurePathFinder(core_params, "background_params.txt"), 'w') as configfile:
            config.write(configfile)
            
    plt.show()

def two_d_gaus(x, y, sx, sy):
    """2D Gaussian"""
    return(np.exp(-((x**2 / (2*sx**2))+(y**2 / (2*sy**2)))))
    
def CalcMaxPhotonsPixel(row):
    """Calculates the photons collected over the 1x1 pixel central area of a given localization"""
    from scipy import integrate
    import math
    N = row["center_photons"]
    sx = row["sx"]
    sy = row["sy"]
    ROI_size=1
    ROI_bound = ROI_size/2
    result = integrate.dblquad(two_d_gaus, -ROI_bound, ROI_bound, -ROI_bound, ROI_bound, args=(sx, sy))
    entire = 2*math.pi*sx*sy
    scale_factor = N/entire
    peak_pixel_value = scale_factor*result[0]
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
    #all_center["peak_pixel_value"] = all_center.swifter.apply(CalcMaxPhotonsPixel, axis=1)
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
        plt.savefig(FigurePathFinder(core_params, "sbr.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'mean': str(mean),
            'hist_max': str(maxi),
            'binwidth': str(binwidth),
            'n_localizations': str(number),
            'file_name': "sbr.pdf"}
        with open(FigurePathFinder(core_params, "sbr_params.txt"), 'w') as configfile:
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
    """Saves a dataframe to a csv file"""
    df.to_csv(FigurePathFinder(core_params, file_name+".csv"), index = True)
    return()

###############################################################################################
###############################################################################################
###############################################################################################

class Measurement:
    def __init__(self,
                 core_params = []):
        self.core_params = core_params
        self.group_info_df = []
        self.edge_photons = []
        self.center_photons = []
        self.fulltable = []
        self.all_center = []
    
    def HeavyCalcs(self):
        t1 = time.time()
        fulltable = Import(self.core_params["path"])
        enhanced_df = EnhanceDataFrame(fulltable, self.core_params)
        bindingevent_info_df = Build_df_bindingevents(enhanced_df)
        group_info_df = Build_df_groups(bindingevent_info_df, self.core_params["n_frames"])
        FileSaver(bindingevent_info_df, "binding_event_info", self.core_params)
        FileSaver(group_info_df, "group_info", self.core_params)
        all_first, all_center, all_last = AllPhotons(enhanced_df)
        all_center = SBR(all_center)
        edge_photons = np.concatenate([np.asarray(all_first["first_photons"]), np.asarray(all_last["last_photons"])])
        center_photons = np.asarray(all_center["center_photons"])
        t2 = time.time()
        print("Time elapsed: ", np.round(t2-t1, 2), " seconds")
        self.group_info_df = group_info_df
        self.edge_photons = edge_photons
        self.center_photons = center_photons
        self.fulltable = fulltable
        self.all_center = all_center
        
    def Plots(self, plotting_params, save):
        print("Percentage of sites destroyed:", SiteDestructionAnalysis(self.group_info_df, self.core_params["save_path"], False, save))
        #Plot dark times
        PlotAllKinetics(self.group_info_df, plotting_params["dark_range"], plotting_params["dark_cutoff"], "dark", plotting_params["dark_binwidth"], [self.core_params["plot_title"]+"\nDark times", "Dark times (s)", "Number of binding sites"], self.core_params, save)
        #Plot bright times
        PlotAllKinetics(self.group_info_df, plotting_params["bright_range"], plotting_params["bright_cutoff"], "bright", plotting_params["bright_binwidth"], [self.core_params["plot_title"]+"\nBright times", "Bright times (s)", "Number of binding sites"], self.core_params, save)
        #Plot photon histogram
        PhotoPlot(self.edge_photons, self.center_photons, self.core_params, NumberOfGroups(self.fulltable), plotting_params["bin_width_photons"], plotting_params["fit_style_photons"], plotting_params["PhotoPlot_fit_params"], plotting_params["x_factor_photons"], save)
        #Plot localizations over time
        LocPlot(self.fulltable, self.core_params, NumberOfGroups(self.fulltable), save)
        #Plot background
        BackGroundPlot(self.fulltable, self.core_params, plotting_params["binwidth_bg"], plotting_params["plot_range_bg"], save)
        #Plot signal to background ratio
        PlotSBR(self.all_center, self.core_params, plotting_params["binwidth_sbr"], plotting_params["x_factor_sbr"], save)



