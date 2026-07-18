# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 11:40:14 2026

@author: LukSu
"""
#%% Imports
import os
import sys
import numpy as np
import pandas as pd
import json
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter, MultipleLocator
from matplotlib.backends.backend_pdf import PdfPages
import mne

plt.switch_backend('pdf') # To prevent figures from opening

#%% Parameters
SaveFile = "F:\\finalfigures.pdf"
DataLocationLFP = "F:\\saveddata\\260520-All"
DataLocationEEG = "F:\\savedEEG\\260525-All"
StatInputFile = "stats\\inputvariables.json"
with open(StatInputFile, 'r') as f:
    StatInput = json.load(f)
rowcountsceeg = [4,4,8,8]

# CSD
DataFiles = {
    "Primed": {"Contra": "TargetPrimed.npy",
               "Ipsi": "DistractorPrimed.npy"},
    "Unprimed": {"Contra": "TargetUnprimed.npy",
                 "Ipsi": "DistractorUnprimed.npy"}
    }
ProbeFile = "ProbeDistances.csv"
TimeFile = "TimeArray.npy"
TimeWindow = [-0.1, 0.23] # ms

# Dipole moment
CategoryColor = {"Primed": "purple", "Unprimed": "green"}

# cEEG
LeadFieldSel = ["Plg142V4"]
Electrode = "CP3"
ExcludedSessionscEEG = StatInput["Statistics"]["Excluded session indices cEEG"]
ZoomedTime = np.array(StatInput["Plotting"]["Zoomed in plot time window"])/1000
JointPlotTimes = [0.12]
JointPlotSphere = [0, 0.025, 0, 0.09]
TickSpacing = 0.05

TimeClip = StatInput["Plotting"]["Time point to clip data at"]/1000

# EEG
ExcludedSessionsEEG = StatInput["Statistics"]["Excluded session indices EEG"]

#%% Check save location
if os.path.isfile(SaveFile):
    confirm = input(f'\n\n\033[91mFile \033[96m{SaveFile} \033[91malready exists\033[0m, overwrite? [y/n] ')
    if confirm.upper() != 'Y':
        sys.exit()
    else:
        print("Overwriting...")
os.makedirs(f'{"\\".join(SaveFile.split("\\")[:-1])}', exist_ok = True)

#%% Functions
def find_nearest(array, value):
    indx = (np.abs(array - value)).argmin()
    return indx, array[indx]

def RoundHalfUp(number, decimal = 0):
    return np.floor(number * (10**decimal) + 0.5)/(10**decimal)

def GaussianFilter(data, zs, gauss_sigma, filter_range):
    step = zs[1] - zs[0]
    filter_positions = np.arange(-filter_range/2, filter_range/2+step, step)
    gaussianfilter = 1/(gauss_sigma * np.sqrt(2 * np.pi)) * np.exp(-filter_positions**2/(2*gauss_sigma**2))
    filterlength = len(gaussianfilter)
    (m,n) = data.shape
    tmp_data = np.zeros((m+2*filterlength, n))
    tmp_data[filterlength:filterlength+m,:] = data[:,:]
    scalingfactor = sum(gaussianfilter)
    tmp_data = lfilter(gaussianfilter/scalingfactor, 1, tmp_data, axis = 0)
    filtered_data = tmp_data[int(RoundHalfUp(1.5*filterlength)):int(RoundHalfUp(1.5*filterlength))+m,:]
    return filtered_data

def SmoothCSD_2D(data, resolutionfactor):
    totchan = (data.shape[0]+2)/10
    el_pos = np.arange(0.1, totchan+0.1, 0.1)
    npoints = resolutionfactor*data.shape[0]
    le = len(el_pos)-3
    first_z = el_pos[0] - (el_pos[1] - el_pos[0])/2
    last_z = el_pos[le] + (el_pos[le] - el_pos[le-1])/2
    zs = np.arange(first_z, last_z+(last_z - first_z)/npoints, (last_z - first_z)/npoints)
    el_pos[le+1] = el_pos[le] + (el_pos[le] - el_pos[le-1])
    smooth_csd = np.empty((len(zs), data.shape[1]))
    jj = 0
    for ii in range(len(zs)):
        if zs[ii] > (el_pos[jj] + (el_pos[jj+1] - el_pos[jj])/2):
            jj = min(jj+1, le)
        smooth_csd[ii,:] = data[jj,:]
    gauss_sigma = 0.1
    filter_range = 5 * gauss_sigma
    return GaussianFilter(smooth_csd, zs, gauss_sigma, filter_range)

def PlotCSD(ax, data, time, xlimit = [], xlimitlower = None, xlimitupper = None, xlbl = "Time from event (s)", ylbl = "Lower<-Cx (Depth)->Upper", title = "", colbarlbl = "CSD (nA/mm$^3$)"):
    resolution = 10
    if xlimit: xlimitlower, xlimitupper = xlimit
    csd = np.nanmean(data, axis = 2)
    fs_csd = SmoothCSD_2D(np.vstack((csd[0,:], csd, csd[-1,:])), resolution)
    if xlimitupper: ax.set_xlim(right = xlimitupper)
    if xlimitlower: ax.set_xlim(left = xlimitlower)
    if xlimitlower and xlimitupper:
        lowerindx, lowerbound = find_nearest(time, xlimitlower)
        upperindx, upperbound = find_nearest(time, xlimitupper)
        climit = np.nanmax(np.abs(fs_csd[:,lowerindx:upperindx]))
    else:
        lowerindx = 0
        upperindx = -1
        lowerbound = time[0]
        upperbound = time[-1]
        climit = np.nanmax(np.abs(fs_csd))
    im = ax.imshow(fs_csd[:,lowerindx:upperindx], cmap = "jet_r", extent = [lowerbound, upperbound, data.shape[0]+0.5, 0.5], interpolation = "none", clim = [-climit, climit], aspect = "auto")
    colbar = ax.figure.colorbar(im, ax=ax)
    if colbarlbl: colbar.set_label(colbarlbl)
    if title: ax.set_title(title)
    if xlbl: ax.set_xlabel(xlbl)
    if ylbl: ax.set_ylabel(ylbl)
    ax.axes.get_yaxis().set_ticks(np.arange(1, data.shape[0]+1, 1))

#%% Load data
CSDData = {}
SessionInformation = {}
ProbeDict = {}
DipoleData = {}
cEEGData = {}
EEGData = {}

# CSD
for category, files in DataFiles.items():
    CSDData.setdefault(category, {})
    SessionInformation.setdefault(category, {})
    for hemifield, filename in files.items():
        CSDData[category][hemifield] = np.load(f'{DataLocationLFP}\\{filename}')
        SessionInformation[category][hemifield] = pd.read_csv(f'{DataLocationLFP}\\{filename[:-4]}.csv', index_col = 0)
ElectrodeCount = CSDData[category][hemifield].shape[0]
TimeArray = np.load(f'{DataLocationLFP}\\{TimeFile}')/1000 # s
ProbeInformation = pd.read_csv(f'{DataLocationLFP}\\{ProbeFile}', index_col = [0,1])
for (session, key), valuedict in ProbeInformation.T.to_dict().items():
    ProbeDict.setdefault(session, {"Indices": np.array(list(valuedict.keys())).astype(np.int_)})
    try:
        ProbeDict[session][key] = np.array(list(valuedict.values())).astype(np.float64)
    except ValueError:
        ProbeDict[session][key] = np.array(list(valuedict.values()))
# Dipole moment
for filename in os.listdir(f'{DataLocationEEG}\\npy_dipole'):
    if filename[-4:]==".npy" and filename[:6]=="dipole":
        _, category = filename[:-4].split("_")
        DipoleData[category] = np.load(f'{DataLocationEEG}\\npy_dipole\\{filename}')

# cEEG
for filename in os.listdir(f'{DataLocationEEG}\\mne'):
    if filename[-4:]==".fif" and filename[:4]=="cEEG":
        _, condition, hemifield, loc, _ = filename[:-4].split("_")
        cEEGData.setdefault(condition, {})
        cEEGData[condition].setdefault(hemifield, {})
        data = mne.read_epochs(f'{DataLocationEEG}\\mne\\{filename}', proj = False, verbose = False)
        cEEGData[condition][hemifield][loc] = data

# EEG
for filename in os.listdir(f'{DataLocationEEG}\\npy'):
    if filename[-4:]==".npy" and filename[:3]=="EEG":
        _, condition, hemifield = filename[:-4].split("_")
        EEGData.setdefault(condition, {})
        data = np.load(f'{DataLocationEEG}\\npy\\{filename}')
        EEGData[condition][hemifield] = data

#%% Create cEEG plots
cEEGlayout = [["CSD A", "Dipole A", "Wave P",  "Wave P" ],
              ["CSD A", "Dipole A", "Wave U",  "Wave U" ],
              ["CSD B", "Dipole B", "Wave D",  "Wave D" ],
              ["CSD B", "Dipole B", "Wave D", "Wave D"],
              ["CSD C", "Dipole C", "Topo P",  "Topo PL"],
              ["CSD C", "Dipole C", "Ts P",    "Ts P"   ],
              ["CSD C", "Dipole C", "Ts P",    "Ts P"   ],
              ["CSD D", "Dipole D", "Topo U",  "Topo UL"],
              ["CSD D", "Dipole D", "Ts U",    "Ts U"   ],
              ["CSD D", "Dipole D", "Ts U",    "Ts U"   ]]
cEEGlayoutw = [0.33, 0.33, 0.33, 0.01]
cEEGlayouth = [0.125, 0.125, 0.125, 0.125, 0.155, 0.05, 0.025, 0.15, 0.05, 0.025]
cEEGdict = {
            "CSD": {
                "Primed": {
                    "Contra": "CSD A",
                    "Ipsi": "CSD B"
                    },
                "Unprimed": {
                    "Contra": "CSD C",
                    "Ipsi": "CSD D"
                    }
                },
            "Dipole": {
                "Primed": {
                    0: "Dipole A",
                    1: "Dipole B"
                    },
                "Unprimed": {
                    0: "Dipole C",
                    1: "Dipole D"
                    }
                },
            "Wave": {
                "Primed": "Wave P",
                "Unprimed": "Wave U",
                "Diff": "Wave D",
                "Diff zoomed": "Wave DZ"
                },
            "Joint": {
                "Primed": {
                    "Topo": "Topo P",
                    "Topo legend": "Topo PL",
                    "Ts": "Ts P"
                    },
                "Unprimed": {
                    "Topo": "Topo U",
                    "Topo legend": "Topo UL",
                    "Ts": "Ts U"
                    }
                }
            }
supfigceeg = plt.figure(layout = "constrained", figsize = (25,15))
supaxesceeg = supfigceeg.subplot_mosaic(cEEGlayout, width_ratios = cEEGlayoutw, height_ratios = cEEGlayouth)
supaxesceeg[cEEGdict["Wave"]["Diff"]].invert_yaxis()
# ceeginsetax = supaxesceeg[cEEGdict["Wave"]["Diff"]].inset_axes([0.2, 0.4, 0.4, 0.65], xlim = ZoomedTime)
# ceeginsetax.invert_yaxis()

# CSD
for cond, conddata in CSDData.items():
    indx = 0 if cond == "Primed" else len(CSDData)
    for ii, (hemifield, hemidata) in enumerate(conddata.items()):
        PlotCSD(supaxesceeg[cEEGdict["CSD"][cond][hemifield]], hemidata, TimeArray, xlimit = TimeWindow)

# Dipole moment
for cond, conddata in DipoleData.items():
    indx = 0 if cond == "Primed" else len(DipoleData)
    for hemiindx, hemidata in enumerate(conddata):
        # hemiindx 0 : Contra, hemiindx 1 : ipsi
        timeindx = (find_nearest(TimeArray, TimeWindow[0])[0], find_nearest(TimeArray, TimeWindow[1])[0])
        supaxesceeg[cEEGdict["Dipole"][cond][hemiindx]].plot(TimeArray[timeindx[0]:timeindx[1]], hemidata[timeindx[0]:timeindx[1]], color = CategoryColor[cond])
        supaxesceeg[cEEGdict["Dipole"][cond][hemiindx]].set_xlabel("Time (s)")
        supaxesceeg[cEEGdict["Dipole"][cond][hemiindx]].set_ylabel("Current dipole moment (nA*m)")
        supaxesceeg[cEEGdict["Dipole"][cond][hemiindx]].invert_yaxis()


# cEEG
for indx, (cond, conddata) in enumerate(cEEGData.items()):
    for leadfieldname in LeadFieldSel:
        ipsidata = conddata[leadfieldname]["Ipsi"]
        contradata = conddata[leadfieldname]["Contra"]
        diffdata = conddata[leadfieldname]["Diff"]
        supaxesceeg[cEEGdict["Joint"][cond]["Ts"]].invert_yaxis()
        diffdata.average().plot_joint(times = JointPlotTimes, title = None, show = True,
                                      ts_args = dict(proj = False, sphere = JointPlotSphere, highlight = None, axes = supaxesceeg[cEEGdict["Joint"][cond]["Ts"]]),
                                      topomap_args = dict(proj = False, sphere = JointPlotSphere, axes = [supaxesceeg[cEEGdict["Joint"][cond]["Topo"]],
                                                                                                          supaxesceeg[cEEGdict["Joint"][cond]["Topo legend"]]]))
        supaxesceeg[cEEGdict["Joint"][cond]["Topo legend"]].yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        supaxesceeg[cEEGdict["Joint"][cond]["Ts"]].xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        supaxesceeg[cEEGdict["Joint"][cond]["Ts"]].xaxis.set_major_locator(MultipleLocator(TickSpacing))
        npipsi = ipsidata.get_data(picks = Electrode).squeeze()*1e6
        npcontra = contradata.get_data(picks = Electrode).squeeze()*1e6
        npdiff = diffdata.get_data(picks = Electrode).squeeze()*1e6
        nptime = ipsidata.times
        cliptimeindx = find_nearest(nptime, TimeClip)[0]
        npipsi = npipsi[:,:cliptimeindx]
        npcontra = npcontra[:,:cliptimeindx]
        npdiff = npdiff[:,:cliptimeindx]
        nptime = nptime[:cliptimeindx]
        supaxesceeg[cEEGdict["Wave"][cond]].plot(nptime, np.nanmean(np.delete(npipsi, ExcludedSessionscEEG, axis = 0), axis = 0), color = "blue", label = "Ipsi")
        supaxesceeg[cEEGdict["Wave"][cond]].plot(nptime, np.nanmean(np.delete(npcontra, ExcludedSessionscEEG, axis = 0), axis = 0), color = "red", label = "Contra")
        supaxesceeg[cEEGdict["Wave"][cond]].set_xlabel("Time (s)")
        supaxesceeg[cEEGdict["Wave"][cond]].set_ylabel("Voltage (μV)")
        supaxesceeg[cEEGdict["Wave"][cond]].invert_yaxis()
        supaxesceeg[cEEGdict["Wave"][cond]].legend()
        npdiffmean = np.nanmean(np.delete(npdiff, ExcludedSessionscEEG, axis = 0), axis = 0)
        npdiffste = np.std(np.delete(npdiff, ExcludedSessionscEEG, axis = 0), axis = 0) / np.sqrt(np.delete(npdiff, ExcludedSessionscEEG, axis = 0).shape[0])
        supaxesceeg[cEEGdict["Wave"]["Diff"]].plot(nptime, np.nanmean(np.delete(npdiff, ExcludedSessionscEEG, axis = 0), axis = 0), color = CategoryColor[cond], label = cond)
        supaxesceeg[cEEGdict["Wave"]["Diff"]].fill_between(nptime, np.subtract(npdiffmean, npdiffste*2), np.add(npdiffmean, npdiffste*2), alpha = 0.3, color = CategoryColor[cond])
        supaxesceeg[cEEGdict["Wave"]["Diff"]].set_xlabel("Time (s)")
        supaxesceeg[cEEGdict["Wave"]["Diff"]].set_ylabel("Voltage (μV)")
        timeindx = (find_nearest(nptime, ZoomedTime[0])[0], find_nearest(nptime, ZoomedTime[1])[0])
        # ceeginsetax.plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npdiff, ExcludedSessionscEEG, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = CategoryColor[cond])
        # ceeginsetax.fill_between(nptime[timeindx[0]:timeindx[1]], np.subtract(npdiffmean, npdiffste*2)[timeindx[0]:timeindx[1]], np.add(npdiffmean, npdiffste*2)[timeindx[0]:timeindx[1]], alpha = 0.3, color = CategoryColor[cond])
        # supaxesceeg[cEEGdict["Wave"]["Diff zoomed"]].set_xlabel("Time (s)")
        # supaxesceeg[cEEGdict["Wave"]["Diff zoomed"]].set_ylabel("Voltage (μV)")
# supaxesceeg[cEEGdict["Wave"]["Diff"]].indicate_inset_zoom(ceeginsetax)
supaxesceeg[cEEGdict["Wave"]["Diff"]].legend()

#%% Create EEG plots
supfigeeg = plt.figure(constrained_layout = True, figsize = (12, 7))
eegsubfigs = supfigeeg.subfigures(nrows = 3, ncols = 1)
eegaxes = [subfig.subplots(nrows = 1, ncols = 1) for subfig in eegsubfigs]
[ax.invert_yaxis() for ax in eegaxes]
# eegfigz = plt.figure(constrained_layout = True)
# eegsubfigsz = eegfigz.subfigures(nrows = 3, ncols = 1)
# eegaxesz = [subfig.subplots(nrows = 1, ncols = 1) for subfig in eegsubfigsz]
# [ax.invert_yaxis() for ax in eegaxesz]

for onsindx, (cond, conddata) in enumerate(EEGData.items()):
    npipsi = conddata["Ipsi"].squeeze().T
    npcontra = conddata["Contra"].squeeze().T
    cliptimeindx = find_nearest(nptime, TimeClip)[0]
    npipsi = npipsi[:,:cliptimeindx]
    npcontra = npcontra[:,:cliptimeindx]
    nptime = nptime[:cliptimeindx]
    npdiff = npcontra - npipsi
    eegaxes[onsindx].plot(nptime, np.nanmean(np.delete(npipsi, ExcludedSessionsEEG, axis = 0), axis = 0), color = "blue", label = "Ipsi")
    eegaxes[onsindx].plot(nptime, np.nanmean(np.delete(npcontra, ExcludedSessionsEEG, axis = 0), axis = 0), color = "red", label = "Contra")
    eegaxes[onsindx].set_xlabel("Time (s)")
    eegaxes[onsindx].set_ylabel("Voltage (μV)")
    eegaxes[onsindx].legend()
    timeindx = (find_nearest(nptime, ZoomedTime[0])[0], find_nearest(nptime, ZoomedTime[1])[0])
    # eegaxesz[onsindx].plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npipsi, ExcludedSessionsEEG, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = "blue")
    # eegaxesz[onsindx].plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npcontra, ExcludedSessionsEEG, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = "red")
    # eegaxesz[onsindx].set_xlabel("Time (s)")
    # eegaxesz[onsindx].set_ylabel("Voltage (μV)")
    npdiffmean = np.nanmean(np.delete(npdiff, ExcludedSessionsEEG, axis = 0), axis = 0)
    npdiffste = np.std(np.delete(npdiff, ExcludedSessionsEEG, axis = 0), axis = 0) / np.sqrt(np.delete(npdiff, ExcludedSessionsEEG, axis = 0).shape[0])
    eegaxes[2].plot(nptime, np.nanmean(np.delete(npdiff, ExcludedSessionsEEG, axis = 0), axis = 0), color = CategoryColor[cond], label = cond)
    eegaxes[2].fill_between(nptime, np.subtract(npdiffmean, npdiffste*2), np.add(npdiffmean, npdiffste*2), alpha = 0.3, color = CategoryColor[cond])
    eegaxes[2].set_xlabel("Time (s)")
    eegaxes[2].set_ylabel("Voltage (μV)")
    # eegaxesz[2].plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npdiff, ExcludedSessionsEEG, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = CategoryColor[cond])
    # eegaxesz[2].fill_between(nptime[timeindx[0]:timeindx[1]], np.subtract(npdiffmean, npdiffste*2)[timeindx[0]:timeindx[1]], np.add(npdiffmean, npdiffste*2)[timeindx[0]:timeindx[1]], alpha = 0.3, color = CategoryColor[cond])
    # eegaxesz[2].set_xlabel("Time (s)")
    # eegaxesz[2].set_ylabel("Voltage (μV)")
eegaxes[2].legend()

#%% Save figures
with PdfPages(SaveFile) as pdf:
    pdf.savefig(supfigceeg)
    pdf.savefig(supfigeeg)

plt.close("all")
plt.switch_backend('QtAgg')
