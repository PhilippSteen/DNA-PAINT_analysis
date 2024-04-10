########################################################
#########           Philipp R. Steen           #########
#########             Jungmann Lab             #########
######### Max Planck Institute of Biochemistry #########
#########     Ludwig Maximilian University     #########
#########                 2024                 #########
########################################################

import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import matplotlib as mpl
import configparser
from datetime import datetime
from scipy.optimize import curve_fit
from lmfit.models import ExponentialModel
from scipy.signal import savgol_filter
import os

def _2exponential(x, a, k1, b, k2):
    return a*np.exp(x*k1) + b*np.exp(x*k2)

def SiteDestructionAnalysis(group_info_df, core_params, which_a, save): 
    """Checks for binding sites that appear to be destroyed over the course of the measurement"""
    mpl.style.use('seaborn-poster')
    fig, ax = plt.subplots(1, figsize = (8, 5))
    fig.tight_layout()
    
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

    if which_a=="bi":
        popt_2exponential, pcov_2exponential = curve_fit(_2exponential, xvals, yvals, p0=[100,-0.1,200,-0.1])
        m1 = -1/popt_2exponential[1]
        m2 = -1/popt_2exponential[3]
        print(popt_2exponential)
        print("first mean = ", m1)
        print("second mean = ", m2)
        ax.plot(xvals, _2exponential(xvals, *popt_2exponential), color = "blue")
        realmean = min(m1, m2)
        cutter = realmean*4
        print("Used bi-exponential fit")
        
    if cutter < 4:
        cutter = 4
    
    apparently_destroyed_sites = non_zero_sites[non_zero_sites["end_ratio"]>=cutter]
    apparently_good_sites = non_zero_sites[non_zero_sites["end_ratio"]<cutter]

    d_x = np.asarray(apparently_destroyed_sites["x"])
    d_y = np.asarray(apparently_destroyed_sites["y"])
    g_x = np.asarray(apparently_good_sites["x"])
    g_y = np.asarray(apparently_good_sites["y"])
    
    ratio_of_ratios = len(apparently_destroyed_sites)/(len(apparently_good_sites)+len(apparently_destroyed_sites))

    ax.hist(non_zero_sites["end_ratio"], bins = np.arange(min(non_zero_sites["end_ratio"]), max(non_zero_sites["end_ratio"]) + binwidth, binwidth), color = "grey")
    ax.plot(xvals, result_e.best_fit, color = "black")
    ax.axvline(x=cutter, color = "red", label = "Cutoff at "+str(np.round(cutter, 2)))
    ax.set_xlim([0, 4*cutter])
    plt.title(core_params["plot_title"]+"\nDestruction Analysis")
    ax.set_xlabel("Ratio")
    ax.set_ylabel("Number of binding sites")
    ax.plot([], [], ' ', label="Sites destroyed: "+str(np.round(100*ratio_of_ratios,1))+" %")
    ax.legend()
    
    if save==True:
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_binding_site_destruction.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'portion_destroyed': str(ratio_of_ratios),
            'number_destroyed': str(len(apparently_destroyed_sites)),
            'number_survived': str(len(apparently_good_sites)),
            'sigma': str(cutter/4),
            'cutoff': str(cutter),
            'file_name': 'binding_site_destruction.pdf'}
        with open(os.path.join(core_params["save_path"], core_params["save_code"]+"_binding_site_destruction.txt"), 'w') as configfile:
            config.write(configfile)
        save_picks(d_x, d_y, os.path.join(core_params["save_path"], core_params["save_code"]+"_apparently_destroyed_sites"))
        save_picks(g_x, g_y, os.path.join(core_params["save_path"], core_params["save_code"]+"_apparently_good_sites"))
    plt.show()
    return(ratio_of_ratios)

def BackgroundOverTime(df, core_params, plotting_params, save):
    """Plots the average background photon count over the course of the measurement"""
    mpl.style.use('seaborn-poster')
    fig = plt.figure(figsize=(8, 5))
    fig.tight_layout()
    plt.title(core_params["plot_title"]+"\nBackground over time")
    plt.xlabel("Time (s)")
    plt.ylabel("Background photons per pixel")
    bincolor, linecolor = GetHistColor(core_params["colorcode"])
    df = df[df["bg"]>0]
    number = df.shape[0]
    diff = plotting_params["average_frames"]
    timestep_s = int(diff*core_params["exposure_time"])
    segments = np.arange(0, df["frame"].max(), diff)
    xvals = segments+0.5*diff
    xvals_seconds = xvals*core_params["exposure_time"]
    simps = []
    for step in segments:
        simp = df[(df["frame"]>=step) & (df["frame"]<(step+diff))].mean()
        simps.append(simp["bg"])
    plt.plot(xvals_seconds, simps, color = bincolor, linewidth = 1, label = str(timestep_s)+" second average\nbackground photons")
    
    seg1 = SogSegments(segments)
    yhat = savgol_filter(simps, seg1, 3)
    plt.plot(xvals_seconds, yhat, color = linecolor, linewidth = 2, label = "Smoothed bg. ph.")
    plt.legend()
    if save==True:
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_bg_time.pdf"), bbox_inches="tight", pad_inches=0.2)
        config = configparser.ConfigParser()
        config['params'] = {
            'Date and time': str(datetime.now()),
            'avg_timestep_frames': str(diff),
            'file_name': "bg_time.pdf"}
        with open(os.path.join(core_params["save_path"], core_params["save_code"]+"_bg_time_params.txt"), 'w') as configfile: 
            config.write(configfile)
    plt.show()
    
def SogSegments(segments):
    seg1 = int(0.5*len(segments)+1)
    if (seg1%2) == 0:
        seg1 = int(0.5*len(segments))
    return(seg1)

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
        plt.savefig(os.path.join(core_params["save_path"], core_params["save_code"]+"_photons.pdf"), bbox_inches="tight", pad_inches=0.2)
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
        with open(os.path.join(core_params["save_path"], core_params["save_code"]+"_photons_params.txt"), 'w') as configfile:
            config.write(configfile)
    plt.show()
    return(center)

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