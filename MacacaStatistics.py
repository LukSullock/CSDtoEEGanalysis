# -*- coding: utf-8 -*-
"""
CSD to EEG conversion. Scripts used to convert CSD to EEG and analyse resulting data.

Copyright (C) 2026 Luk Sullock Enzlin

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
#%% Imports
import os
import sys
import numpy as np
import pandas as pd
import json
import scipy.stats
import shutil
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib.transforms as tfrms
from matplotlib.backends.backend_pdf import PdfPages
import time
import mne

#%%
plt.switch_backend('pdf') # To prevent figures from opening
plt.rcParams.update({'figure.max_open_warning': 40})

#%% Variable settings

DataLocationLFP = "F:\\saveddata\\260520-All"
DataLocationEEG = "F:\\savedEEG\\260525-All"
SaveLocation = "F:\\SavedStatistics\\260719-All"

# CSD
DataFiles = {
    "Primed": {"Contra": "TargetPrimed.npy",
               "Ipsi": "DistractorPrimed.npy"},
    "Unprimed": {"Contra": "TargetUnprimed.npy",
                 "Ipsi": "DistractorUnprimed.npy"}
    }
ProbeFile = "ProbeDistances.csv"
TimeFile = "TimeArray.npy"
TimeWindow = [-100, 230] # ms

# cEEG
LeadFieldSel = ["Plg142V4"]
Electrode = "CP3"

# Visualise
pltsize = (6,4)
CategoryColor = {"Primed": "purple", "Unprimed": "green"}
TimeClip = 230 # ms
ZoomedTime = [100, 200] # ms
BaselineTime = [-100, 0] # ms
SNRThresh = 2.0

# Statistics
# Exclusion criteria: both ipsi and contra SNR less than SNRThresh in either primed or unprimed
ExclusionSessionIndx = [3, 4, 12] # [3, 4, 12]
ExclusionSessionIndxEEG = [4, 10, 12, 18] # [3, 4, 6, 10, 12, 18]
MinTimePoints = 1
SearchInterval = (0.07, 0.25) # s
PercentMax = 0.25



#%% Check save location
if os.path.exists(SaveLocation):
    confirm = input(f'\n\n\033[91mFolder \033[96m{SaveLocation} \033[91malready exists\033[0m, overwrite? [y/n] ')
    if confirm.upper() != 'Y':
        sys.exit()
    else:
        print("Overwriting...")
        try:
            shutil.rmtree(SaveLocation)
        except OSError as e:
            time.sleep(0.05)
            if not os.listdir(SaveLocation):
                pass
            else:
                raise e
os.makedirs(f'{SaveLocation}', exist_ok = True)
# os.makedirs(f'{SaveLocation}\\cEEG', exist_ok = True)
# os.makedirs(f'{SaveLocation}\\CSD', exist_ok = True)
# os.makedirs(f'{SaveLocation}\\EEG', exist_ok = True)

#%% General functions
def find_nearest(array, value):
    indx = (np.abs(array - value)).argmin()
    return indx, array[indx]

def calculate_snr(array, baselineindx, axis = 0, ddof = 0):
    mean = array[baselineindx:].mean(axis)
    std = array[:baselineindx].std(axis = axis, ddof = ddof)
    return abs(np.where(std == 0, 0, mean/std))

#%% CSD load data
CSDData = {}
SessionInformation = {}
for category, files in DataFiles.items():
    CSDData.setdefault(category, {})
    SessionInformation.setdefault(category, {})
    for hemifield, filename in files.items():
        CSDData[category][hemifield] = np.load(f'{DataLocationLFP}\\{filename}')
        SessionInformation[category][hemifield] = pd.read_csv(f'{DataLocationLFP}\\{filename[:-4]}.csv', index_col = 0)
ElectrodeCount = CSDData[category][hemifield].shape[0]
TimeArray = np.load(f'{DataLocationLFP}\\{TimeFile}')/1000 # s
ProbeInformation = pd.read_csv(f'{DataLocationLFP}\\{ProbeFile}', index_col = [0,1])
ProbeDict = {}
for (session, key), valuedict in ProbeInformation.T.to_dict().items():
    ProbeDict.setdefault(session, {"Indices": np.array(list(valuedict.keys())).astype(np.int_)})
    try:
        ProbeDict[session][key] = np.array(list(valuedict.values())).astype(np.float64)
    except ValueError:
        ProbeDict[session][key] = np.array(list(valuedict.values()))

#%% LFP visualise
lfpfig = plt.figure(figsize = (5, 15), constrained_layout = True)
lfpfig.suptitle("LFPs")
lfpsubfigs = lfpfig.subfigures(nrows = 4, ncols = 1)
lfpaxes = [subfig.subplots(nrows = 1, ncols = 1) for subfig in lfpsubfigs]
indx = 0
for category, categorydata in CSDData.items():
    for hemifield, hemifielddata in categorydata.items():
        timeindx = (find_nearest(TimeArray, TimeWindow[0]/1000)[0], find_nearest(TimeArray, TimeWindow[1]/1000)[0])
        plotdf = pd.DataFrame()
        plotdf["Time"] = TimeArray[timeindx[0]:timeindx[1]]*1000
        plotdf.set_index("Time")
        meandata = np.nanmean(hemifielddata[:,timeindx[0]:timeindx[1],:], axis = 2)
        probepos = ProbeDict[list(ProbeDict.keys())[0]]["position_from_L4_boundary_mm"]
        dy = abs(meandata.min() - meandata.max()) * 0.7
        for electrodeindx, electrodedata in enumerate(meandata):
            plotdf[probepos[electrodeindx]] = electrodedata + dy * electrodeindx
        cmap = colors.ListedColormap(CategoryColor[category])
        plotdf.plot(x = "Time", legend = False, colormap = cmap, ax = lfpaxes[indx], title = f'{category} {hemifield}', xlabel = "Time (ms)", ylabel = "Position from L4 boundary (mm)")
        trans = tfrms.blended_transform_factory(lfpaxes[indx].transData, lfpaxes[indx].transAxes)
        ylims = lfpaxes[indx].get_ylim()
        ysize = abs(ylims[0]) + abs(ylims[1])
        
        lfpaxes[indx].errorbar(1.01*TimeWindow[1], dy/ysize, yerr = 100/ysize, capsize = 0, transform = trans) # 100/ysize -> total bar of 200
        lfpaxes[indx].set_ylim(-dy, meandata.shape[0] * dy)
        lfpaxes[indx].set_yticks([ii * dy for ii in range(meandata.shape[0])], labels = plotdf.columns[1:].astype(np.float64))
        indx += 1

#%% Dipole load data
DipoleData = {}
for filename in os.listdir(f'{DataLocationEEG}\\npy_dipole'):
    if filename[-4:]==".npy" and filename[:6]=="dipole":
        _, category = filename[:-4].split("_")
        DipoleData[category] = np.load(f'{DataLocationEEG}\\npy_dipole\\{filename}')

#%% Dipole visualise
dipfig = plt.figure(figsize = (5, 15), constrained_layout = True)
dipfig.suptitle("Dipole moment")
dipsubfigs = dipfig.subfigures(nrows = 4, ncols = 1)
dipaxes = [subfig.subplots(nrows = 1, ncols = 1) for subfig in dipsubfigs]
for indx, (category, categorydata) in enumerate(DipoleData.items()):
    dipsubfigs[2*indx].suptitle(f'{category} Contra')
    dipsubfigs[2*indx+1].suptitle(f'{category} Ipsi')
    timeindx = (find_nearest(TimeArray, TimeWindow[0]/1000)[0], find_nearest(TimeArray, TimeWindow[1]/1000)[0])
    dipaxes[2*indx].plot(TimeArray[timeindx[0]:timeindx[1]]*1000, categorydata[0,timeindx[0]:timeindx[1]].T, color = CategoryColor[category])
    dipaxes[2*indx+1].plot(TimeArray[timeindx[0]:timeindx[1]]*1000, categorydata[1,timeindx[0]:timeindx[1]].T, color = CategoryColor[category])
    dipaxes[2*indx].set_xlabel("Time (ms)")
    dipaxes[2*indx+1].set_xlabel("Time (ms)")
    dipaxes[2*indx].set_ylabel("Current dipole moment (nA*m)")
    dipaxes[2*indx+1].set_ylabel("Current dipole moment (nA*m)")
    dipaxes[2*indx].invert_yaxis()
    dipaxes[2*indx+1].invert_yaxis()

#%% cEEG load data
cEEGData = {}
for filename in os.listdir(f'{DataLocationEEG}\\mne'):
    if filename[-4:]==".fif" and filename[:4]=="cEEG":
        _, condition, hemifield, loc, _ = filename[:-4].split("_")
        cEEGData.setdefault(condition, {})
        cEEGData[condition].setdefault(hemifield, {})
        data = mne.read_epochs(f'{DataLocationEEG}\\mne\\{filename}', proj = False, verbose = False)
        cEEGData[condition][hemifield][loc] = data

#%% cEEG visualise
ceegfig = plt.figure(constrained_layout = True)
ceegfig.suptitle("cEEG Waveforms")
ceegsubfigs = ceegfig.subfigures(nrows = 3, ncols = 1)
ceegsubfigs[0].suptitle("Primed") # (blue: ipsi; red: contra; excluded indices: {ExclusionSessionIndx})
ceegsubfigs[1].suptitle("Unprimed") # (blue: ipsi; red: contra; excluded indices: {ExclusionSessionIndx})
ceegsubfigs[2].suptitle("Difference (purple: Primed; green: Unprimed)") # with indices {ExclusionSessionIndx} excluded 
ceegaxes = [subfig.subplots(nrows = 1, ncols = 1) for subfig in ceegsubfigs]
[ax.invert_yaxis() for ax in ceegaxes]
ceegfigz = plt.figure(constrained_layout = True)
ceegfigz.suptitle("cEEG Waveforms zoomed")
ceegsubfigsz = ceegfigz.subfigures(nrows = 3, ncols = 1)
ceegsubfigsz[0].suptitle("Primed") # (blue: ipsi; red: contra; excluded indices: {ExclusionSessionIndx})
ceegsubfigsz[1].suptitle("Unprimed") # (blue: ipsi; red: contra; excluded indices: {ExclusionSessionIndx})
ceegsubfigsz[2].suptitle("Difference (purple: Primed; green: Unprimed)") # with indices {ExclusionSessionIndx} excluded 
ceegaxesz = [subfig.subplots(nrows = 1, ncols = 1) for subfig in ceegsubfigsz]
[ax.invert_yaxis() for ax in ceegaxesz]
waveceegfig = {}

for onsindx, (category, categorydata) in enumerate(cEEGData.items()):
    for leadfieldname in LeadFieldSel:
        waveceegfig.setdefault(leadfieldname, {})
        ipsidata = categorydata[leadfieldname]["Ipsi"]
        contradata = categorydata[leadfieldname]["Contra"]
        diffdata = categorydata[leadfieldname]["Diff"]
        # diffdata.average().plot_joint(times = [0.15, 0.17, 0.19], title = f'cEEG {category} {leadfieldname} Difference', show = True, ts_args = dict(proj = False, sphere = [0, 0.025, 0, 0.09], highlight = None), topomap_args = dict(proj = False, sphere = [0, 0.025, 0, 0.09]))
        npipsi = ipsidata.get_data(picks = Electrode).squeeze()*1e6
        npcontra = contradata.get_data(picks = Electrode).squeeze()*1e6
        npdiff = diffdata.get_data(picks = Electrode).squeeze()*1e6
        nptime = ipsidata.times
        cliptimeindx = find_nearest(nptime, TimeClip/1000)[0]
        npipsi = npipsi[:,:cliptimeindx]
        npcontra = npcontra[:,:cliptimeindx]
        npdiff = npdiff[:,:cliptimeindx]
        nptime = nptime[:cliptimeindx]
        ceegaxes[onsindx].plot(nptime, np.nanmean(np.delete(npipsi, ExclusionSessionIndx, axis = 0), axis = 0), color = "blue")
        ceegaxes[onsindx].plot(nptime, np.nanmean(np.delete(npcontra, ExclusionSessionIndx, axis = 0), axis = 0), color = "red")
        ceegaxes[onsindx].set_xlabel("Time (s)")
        ceegaxes[onsindx].set_ylabel("Voltage (μV)")
        timeindx = (find_nearest(nptime, ZoomedTime[0]/1000)[0], find_nearest(nptime, ZoomedTime[1]/1000)[0])
        ceegaxesz[onsindx].plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npipsi, ExclusionSessionIndx, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = "blue")
        ceegaxesz[onsindx].plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npcontra, ExclusionSessionIndx, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = "red")
        ceegaxesz[onsindx].set_xlabel("Time (s)")
        ceegaxesz[onsindx].set_ylabel("Voltage (μV)")
        npdiffmean = np.nanmean(np.delete(npdiff, ExclusionSessionIndx, axis = 0), axis = 0)
        npdiffste = np.std(np.delete(npdiff, ExclusionSessionIndx, axis = 0), axis = 0) / np.sqrt(np.delete(npdiff, ExclusionSessionIndx, axis = 0).shape[0])
        ceegaxes[2].plot(nptime, np.nanmean(np.delete(npdiff, ExclusionSessionIndx, axis = 0), axis = 0), color = CategoryColor[category])
        ceegaxes[2].fill_between(nptime, np.subtract(npdiffmean, npdiffste*2), np.add(npdiffmean, npdiffste*2), alpha = 0.3, color = CategoryColor[category])
        ceegaxes[2].set_xlabel("Time (s)")
        ceegaxes[2].set_ylabel("Voltage (μV)")
        ceegaxesz[2].plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npdiff, ExclusionSessionIndx, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = CategoryColor[category])
        ceegaxesz[2].fill_between(nptime[timeindx[0]:timeindx[1]], np.subtract(npdiffmean, npdiffste*2)[timeindx[0]:timeindx[1]], np.add(npdiffmean, npdiffste*2)[timeindx[0]:timeindx[1]], alpha = 0.3, color = CategoryColor[category])
        ceegaxesz[2].set_xlabel("Time (s)")
        ceegaxesz[2].set_ylabel("Voltage (μV)")
        pltsize = (6,4)
        fig, axes = plt.subplots(figsize = (10, 10), nrows = pltsize[0], ncols = pltsize[1], constrained_layout = True)
        fig.suptitle(f'{category} cEEG waveforms per session (blue: ipsi; red: contra)')
        pltc = 0
        print()
        for ii in range(pltsize[0]):
            for jj in range(pltsize[1]):
                if pltc >= npipsi.shape[0]:
                    break
                axes[ii,jj].set_xlabel("Time (s)")
                axes[ii,jj].set_ylabel("Volt (μV)")
                axes[ii,jj].plot(nptime, npipsi[pltc], color = "blue")
                axes[ii,jj].plot(nptime, npcontra[pltc], color = "red")
                axes[ii,jj].set_title(f'index: {pltc}')
                axes[ii,jj].invert_yaxis()
                baselineindx = find_nearest(nptime, BaselineTime[1])[0]
                snr = calculate_snr(npipsi[pltc], baselineindx)
                if snr < SNRThresh:
                    print(f'{category} ipsi index {pltc} has low SNR value ({snr})')
                snr = calculate_snr(npcontra[pltc], baselineindx)
                if snr < SNRThresh:
                    print(f'{category} contra index {pltc} has low SNR value ({snr})')
                pltc += 1
        print()
        waveceegfig[leadfieldname][category] = fig

#%% cEEG N2pc onset latency; Jackknife statistics (R. Ulrich & J. Miller, 2001)
dfanova = {}
cEEG_stats = {}
peakceegfig = {}
for leadfieldname in LeadFieldSel:
    dfanova.setdefault(leadfieldname, pd.DataFrame())
    peakceegfig.setdefault(leadfieldname, {})
    for onsindx, (category, categorydata) in enumerate(cEEGData.items()):
        data = categorydata[leadfieldname]["Diff"]
        npdata = data.get_data(picks = Electrode).squeeze()
        nptime = data.times
        searchindices = (find_nearest(nptime, SearchInterval[0])[0], find_nearest(nptime, SearchInterval[1])[0])
        subjectids = np.arange(0, npdata.shape[0], 1, dtype = np.uint)
        npdata = np.delete(npdata, ExclusionSessionIndx, axis = 0)
        subjectids = np.delete(subjectids, ExclusionSessionIndx)
        dfanova[leadfieldname]["Omitted subject"] = subjectids
        means = np.empty(npdata.shape[0], dtype = np.float64)
        
        fig, axes = plt.subplots(figsize = (10, 10), nrows = pltsize[0], ncols = pltsize[1], constrained_layout = True)
        fig.suptitle(f'{category} cEEG peaks jackknife')
        ii = 0
        jj = 0
        for indx, subid in enumerate(subjectids):
            npmeanwave = np.nanmean(np.delete(npdata, indx, axis = 0), axis = 0)
            peakindx = np.nanargmin(npmeanwave[searchindices[0]:searchindices[1]]) + searchindices[0]
            minwave = npmeanwave[peakindx]
            maxwave = np.nanmean(npmeanwave[:searchindices[0]])
            proportions = (npmeanwave - maxwave) / (minwave - maxwave)
            axes[ii, jj].plot(nptime, proportions, color = CategoryColor[category])
            axes[ii, jj].axhline(PercentMax, color = "red")
            axes[ii, jj].scatter(nptime[peakindx], proportions[peakindx], color = "blue")
            axes[ii, jj].set_title(f'index: {subid}')
            pmthrindx = np.argwhere(proportions[searchindices[0]:peakindx+1] < PercentMax).T[0]
            
            if pmthrindx.size > 1:
                pmthr = (pmthrindx[1:] - pmthrindx[:-1]) == 1
                pmthrconc = np.concatenate([[False], pmthr, [False]])
                pmthredges = np.flatnonzero(pmthrconc[:-1] != pmthrconc[1:])
                pmthrlengths = pmthredges[1::2] - pmthredges[::2]
                means[indx] = nptime[searchindices[0]+pmthrindx[pmthredges[1::2][pmthrlengths >= MinTimePoints][0]]]
                axes[ii, jj].axvline(means[indx], color = "k")
            else:
                means[indx] = np.nan
            jj += 1
            if jj >= pltsize[1]:
                jj = 0
                ii += 1
        print(f'Nan-count cEEG jackknife {leadfieldname} {category}: {means[np.isnan(means)].size}')
        dfanova[leadfieldname][category] = means
        peakceegfig[leadfieldname][category] = fig
    
    ceegasfig = plt.figure(constrained_layout = True)
    ceegasfig.suptitle("Assumptions jackknife cEEG data")
    subfigs = ceegasfig.subfigures(nrows = 2, ncols = 1)
    subfigs[0].suptitle("Variance")
    subfigs[1].suptitle("Normal distributed")
    axes = [subfig.subplots(nrows = 1, ncols = ii+1) for ii, subfig in enumerate(subfigs)]
    dfanova[leadfieldname].boxplot(column = ["Primed", "Unprimed"], ax = axes[0])
    dfanova[leadfieldname].hist(column = ["Primed", "Unprimed"], ax = axes[1], edgecolor = "black")
    
    groupcount = dfanova[leadfieldname].shape[1] - 1 # -1 for subject column
    f_stat, _ = scipy.stats.f_oneway(dfanova[leadfieldname]["Primed"], dfanova[leadfieldname]["Unprimed"])
    print(f'\nF-stat cEEG {leadfieldname}: {f_stat}')
    cf_stat = f_stat/((dfanova[leadfieldname].shape[0] - 1)**2)
    print(f'corrected F-stat: {cf_stat}')
    groupcount = dfanova[leadfieldname].shape[1]-1
    dfn = groupcount - 1
    dfd = (dfanova[leadfieldname].shape[0] * groupcount) - groupcount
    cp = scipy.stats.f.sf(cf_stat, dfn, dfd)
    print(f'corrected p-value: {cp}\n')
    cEEG_stats[leadfieldname] = {
                "F stat": f_stat,
                "Corrected F stat": cf_stat,
                "Corrected p-value": cp,
                "dfn": dfn,
                "dfd": dfd
        }

#%% EEG load data
EEGData = {}
for filename in os.listdir(f'{DataLocationEEG}\\npy'):
    if filename[-4:]==".npy" and filename[:3]=="EEG":
        _, condition, hemifield = filename[:-4].split("_")
        EEGData.setdefault(condition, {})
        data = np.load(f'{DataLocationEEG}\\npy\\{filename}')
        EEGData[condition][hemifield] = data

#%% EEG visualise
eegfig = plt.figure(constrained_layout = True)
eegfig.suptitle("EEG Waveforms")
eegsubfigs = eegfig.subfigures(nrows = 3, ncols = 1)
eegsubfigs[0].suptitle("Primed") # (blue: ipsi; red: contra; excluded sessions: {ExclusionSessionIndxEEG})
eegsubfigs[1].suptitle("Unprimed") # (blue: ipsi; red: contra; excluded sessions: {ExclusionSessionIndxEEG})
eegsubfigs[2].suptitle("Difference (purple: Primed; green: Unprimed)") # with sessions {ExclusionSessionIndxEEG} excluded 
eegaxes = [subfig.subplots(nrows = 1, ncols = 1) for subfig in eegsubfigs]
[ax.invert_yaxis() for ax in eegaxes]
eegfigz = plt.figure(constrained_layout = True)
eegfigz.suptitle("EEG Waveforms zoomed")
eegsubfigsz = eegfigz.subfigures(nrows = 3, ncols = 1)
eegsubfigsz[0].suptitle("Primed") # (blue: ipsi; red: contra; excluded indices: {ExclusionSessionIndx})
eegsubfigsz[1].suptitle("Unprimed") # (blue: ipsi; red: contra; excluded indices: {ExclusionSessionIndx})
eegsubfigsz[2].suptitle("Difference (purple: Primed; green: Unprimed)") # with indices {ExclusionSessionIndx} excluded 
eegaxesz = [subfig.subplots(nrows = 1, ncols = 1) for subfig in eegsubfigsz]
[ax.invert_yaxis() for ax in eegaxesz]
waveeegfig = {}

for onsindx, (category, categorydata) in enumerate(EEGData.items()):
    npipsi = categorydata["Ipsi"].squeeze().T
    npcontra = categorydata["Contra"].squeeze().T
    cliptimeindx = find_nearest(nptime, TimeClip/1000)[0]
    npipsi = npipsi[:,:cliptimeindx]
    npcontra = npcontra[:,:cliptimeindx]
    nptime = nptime[:cliptimeindx]
    npdiff = npcontra - npipsi
    eegaxes[onsindx].plot(nptime, np.nanmean(np.delete(npipsi, ExclusionSessionIndxEEG, axis = 0), axis = 0), color = "blue")
    eegaxes[onsindx].plot(nptime, np.nanmean(np.delete(npcontra, ExclusionSessionIndxEEG, axis = 0), axis = 0), color = "red")
    eegaxes[onsindx].set_xlabel("Time (s)")
    eegaxes[onsindx].set_ylabel("Voltage (μV)")
    timeindx = (find_nearest(nptime, ZoomedTime[0]/1000)[0], find_nearest(nptime, ZoomedTime[1]/1000)[0])
    eegaxesz[onsindx].plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npipsi, ExclusionSessionIndx, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = "blue")
    eegaxesz[onsindx].plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npcontra, ExclusionSessionIndx, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = "red")
    eegaxesz[onsindx].set_xlabel("Time (s)")
    eegaxesz[onsindx].set_ylabel("Voltage (μV)")
    npdiffmean = np.nanmean(np.delete(npdiff, ExclusionSessionIndxEEG, axis = 0), axis = 0)
    npdiffste = np.std(np.delete(npdiff, ExclusionSessionIndxEEG, axis = 0), axis = 0) / np.sqrt(np.delete(npdiff, ExclusionSessionIndxEEG, axis = 0).shape[0])
    eegaxes[2].plot(nptime, np.nanmean(np.delete(npdiff, ExclusionSessionIndxEEG, axis = 0), axis = 0), color = CategoryColor[category])
    eegaxes[2].fill_between(nptime, np.subtract(npdiffmean, npdiffste*2), np.add(npdiffmean, npdiffste*2), alpha = 0.3, color = CategoryColor[category])
    eegaxes[2].set_xlabel("Time (s)")
    eegaxes[2].set_ylabel("Voltage (μV)")
    eegaxesz[2].plot(nptime[timeindx[0]:timeindx[1]], np.nanmean(np.delete(npdiff, ExclusionSessionIndx, axis = 0), axis = 0)[timeindx[0]:timeindx[1]], color = CategoryColor[category])
    eegaxesz[2].fill_between(nptime[timeindx[0]:timeindx[1]], np.subtract(npdiffmean, npdiffste*2)[timeindx[0]:timeindx[1]], np.add(npdiffmean, npdiffste*2)[timeindx[0]:timeindx[1]], alpha = 0.3, color = CategoryColor[category])
    eegaxesz[2].set_xlabel("Time (s)")
    eegaxesz[2].set_ylabel("Voltage (μV)")
    fig, axes = plt.subplots(figsize = (10, 10), nrows = pltsize[0], ncols = pltsize[1], constrained_layout = True)
    fig.suptitle(f'{category} EEG waveforms per session (blue: ipsi; red: contra)')
    pltc = 0
    print()
    for ii in range(pltsize[0]):
        for jj in range(pltsize[1]):
            if pltc >= npipsi.shape[0]:
                break
            axes[ii,jj].set_xlabel("Time (s)")
            axes[ii,jj].set_ylabel("Volt (μV)")
            axes[ii,jj].plot(nptime, npipsi[pltc], color = "blue")
            axes[ii,jj].plot(nptime, npcontra[pltc], color = "red")
            axes[ii,jj].set_title(f'index: {pltc}')
            axes[ii,jj].invert_yaxis()
            baselineindx = find_nearest(nptime, BaselineTime[1])[0]
            snr = calculate_snr(npipsi[pltc], baselineindx)
            if snr < SNRThresh:
                print(f'{category} ipsi index {pltc} has low SNR value ({snr})')
            snr = calculate_snr(npcontra[pltc], baselineindx)
            if snr < SNRThresh:
                print(f'{category} contra index {pltc} has low SNR value ({snr})')
            pltc += 1
    print()
    waveeegfig[category] = fig
    
#%% EEG N2pc onset latency; Jackknife statistics (R. Ulrich & J. Miller, 2001)
dfanovaeeg = pd.DataFrame()
peakeegfig = {}
for onsindx, (category, categorydata) in enumerate(EEGData.items()):
    clipindx = len(nptime)
    npdata = categorydata["Contra"][:,:clipindx,:].squeeze().T - categorydata["Ipsi"][:,:clipindx,:].squeeze().T
    npdata = npdata[:,:len(nptime)]
    searchindices = (find_nearest(nptime, SearchInterval[0])[0], find_nearest(nptime, SearchInterval[1])[0])
    subjectids = np.arange(0, npdata.shape[0], 1, dtype = np.uint)
    npdata = np.delete(npdata, ExclusionSessionIndxEEG, axis = 0)
    subjectids = np.delete(subjectids, ExclusionSessionIndxEEG)
    dfanovaeeg["Omitted subject"] = subjectids
    means = np.empty(npdata.shape[0], dtype = np.float64)
    
    fig, axes = plt.subplots(figsize = (10, 10), nrows = pltsize[0], ncols = pltsize[1], constrained_layout = True)
    fig.suptitle(f'{category} EEG peaks jackknife')
    ii = 0
    jj = 0
    for indx, subid in enumerate(subjectids):
        npmeanwave = np.nanmean(np.delete(npdata, indx, axis = 0), axis = 0)
        peakindx = np.nanargmin(npmeanwave[searchindices[0]:searchindices[1]]) + searchindices[0]
        minwave = npmeanwave[peakindx]
        maxwave = np.nanmean(npmeanwave[:searchindices[0]])
        proportions = (npmeanwave - maxwave) / (minwave - maxwave)
        axes[ii, jj].plot(nptime, proportions, color = CategoryColor[category])
        axes[ii, jj].axhline(PercentMax, color = "red")
        axes[ii, jj].scatter(nptime[peakindx], proportions[peakindx], color = "blue")
        axes[ii, jj].set_title(f'index: {subid}')
        pmthrindx = np.argwhere(proportions[searchindices[0]:peakindx+1] < PercentMax).T[0] # Check where the proportions are below the percent max
        
        if pmthrindx.size > 1:
            pmthr = (pmthrindx[1:] - pmthrindx[:-1]) == 1
            pmthrconc = np.concatenate([[False], pmthr, [False]])
            pmthredges = np.flatnonzero(pmthrconc[:-1] != pmthrconc[1:])
            pmthrlengths = pmthredges[1::2] - pmthredges[::2]
            means[indx] = nptime[searchindices[0]+pmthrindx[pmthredges[1::2][pmthrlengths >= MinTimePoints][0]]]
            axes[ii, jj].axvline(means[indx], color = "k")
        else:
            means[indx] = np.nan
        jj += 1
        if jj >= pltsize[1]:
            jj = 0
            ii += 1
    print(f'Nan-count EEG jackknife {category}: {means[np.isnan(means)].size}')
    peakeegfig[category] = fig
    dfanovaeeg[category] = means

eegasfig = plt.figure(constrained_layout = True)
eegasfig.suptitle("Assumptions jackknife EEG data")
subfigs = eegasfig.subfigures(nrows = 2, ncols = 1)
subfigs[0].suptitle("Variance")
subfigs[1].suptitle("Normal distributed")
axes = [subfig.subplots(nrows = 1, ncols = ii+1) for ii, subfig in enumerate(subfigs)]
dfanovaeeg.boxplot(column = ["Primed", "Unprimed"], ax = axes[0])
dfanovaeeg.hist(column = ["Primed", "Unprimed"], ax = axes[1], edgecolor = "black")

groupcount = dfanovaeeg.shape[1] - 1 # -1 for subject column
f_stat, _ = scipy.stats.f_oneway(dfanovaeeg["Primed"], dfanovaeeg["Unprimed"])
print(f'\nF-stat EEG: {f_stat}')
cf_stat = f_stat/((dfanovaeeg.shape[0] - 1)**2)
print(f'corrected F-stat: {cf_stat}')
groupcount = dfanovaeeg.shape[1]-1
dfn = groupcount - 1
dfd = (dfanovaeeg.shape[0] * groupcount) - groupcount
cp = scipy.stats.f.sf(cf_stat, dfn, dfd)
print(f'corrected p-value: {cp}\n')
EEG_stats = {
            "F stat": f_stat,
            "Corrected F stat": cf_stat,
            "Corrected p-value": cp,
            "dfn": dfn,
            "dfd": dfd
    }

#%% cEEG onsets
dfanova_normal = {}

for leadfieldname in LeadFieldSel:
    dfanova_normal.setdefault(leadfieldname, pd.DataFrame())
    for category, categorydata in cEEGData.items():
        data = categorydata[leadfieldname]["Diff"]
        npdata = data.get_data(picks = Electrode).squeeze()
        nptime = data.times
        searchindices = (find_nearest(nptime, SearchInterval[0])[0], find_nearest(nptime, SearchInterval[1])[0])
        npdata = np.delete(npdata, ExclusionSessionIndx, axis = 0)
        means = np.empty(npdata.shape[0], dtype = np.float64)
        for indx, data in enumerate(npdata):
            peakindx = np.nanargmin(data[searchindices[0]:searchindices[1]]) + searchindices[0]
            minwave = data[peakindx]
            proportions = data / minwave
            pmthrindx = np.argwhere(proportions[searchindices[0]:peakindx+1] < PercentMax).T[0]
            if pmthrindx.size > 1:
                pmthr = (pmthrindx[1:] - pmthrindx[:-1]) == 1
                pmthrconc = np.concatenate([[False], pmthr, [False]])
                pmthredges = np.flatnonzero(pmthrconc[:-1] != pmthrconc[1:])
                pmthrlengths = pmthredges[1::2] - pmthredges[::2]
                try:
                    means[indx] = nptime[searchindices[0]+pmthrindx[pmthredges[1::2][pmthrlengths >= MinTimePoints][0]]]
                except IndexError:
                    means[indx] = np.nan
            else:
                means[indx] = np.nan
        print(f'Nan-count cEEG {leadfieldname} {category}: {means[np.isnan(means)].size}')
        dfanova_normal[leadfieldname][category] = means
        
    ceegasfig_n = plt.figure(constrained_layout = True)
    ceegasfig_n.suptitle("Assumptions normal cEEG data")
    subfigs = ceegasfig_n.subfigures(nrows = 2, ncols = 1)
    subfigs[0].suptitle("Variance")
    subfigs[1].suptitle("Normal distributed")
    axes = [subfig.subplots(nrows = 1, ncols = ii+1) for ii, subfig in enumerate(subfigs)]
    dfanova_normal[leadfieldname].boxplot(column = ["Primed", "Unprimed"], ax = axes[0])
    dfanova_normal[leadfieldname].hist(column = ["Primed", "Unprimed"], ax = axes[1], edgecolor = "black")

#%% EEG onsets
dfanovaeeg_normal = pd.DataFrame()
for category, categorydata in EEGData.items():
    clipindx = len(nptime)
    npdata = categorydata["Contra"][:,:clipindx,:].squeeze().T - categorydata["Ipsi"][:,:clipindx,:].squeeze().T
    npdata = npdata[:,:len(nptime)]
    searchindices = (find_nearest(nptime, SearchInterval[0])[0], find_nearest(nptime, SearchInterval[1])[0])
    npdata = np.delete(npdata, ExclusionSessionIndx, axis = 0)
    means = np.empty(npdata.shape[0], dtype = np.float64)
    for indx, data in enumerate(npdata):
        peakindx = np.nanargmin(data[searchindices[0]:searchindices[1]]) + searchindices[0]
        minwave = data[peakindx]
        proportions = data / minwave
        pmthrindx = np.argwhere(proportions[searchindices[0]:peakindx+1] < PercentMax).T[0]
        if pmthrindx.size > 1:
            pmthr = (pmthrindx[1:] - pmthrindx[:-1]) == 1
            pmthrconc = np.concatenate([[False], pmthr, [False]])
            pmthredges = np.flatnonzero(pmthrconc[:-1] != pmthrconc[1:])
            pmthrlengths = pmthredges[1::2] - pmthredges[::2]
            try:
                means[indx] = nptime[searchindices[0]+pmthrindx[pmthredges[1::2][pmthrlengths >= MinTimePoints][0]]]
            except IndexError:
                means[indx] = np.nan
        else:
            means[indx] = np.nan
    print(f'Nan-count EEG {category}: {means[np.isnan(means)].size}')
    dfanovaeeg_normal[category] = means

eegasfig_n = plt.figure(constrained_layout = True)
eegasfig_n.suptitle("Assumptions normal EEG data")
subfigs = eegasfig_n.subfigures(nrows = 2, ncols = 1)
subfigs[0].suptitle("Variance")
subfigs[1].suptitle("Normal distributed")
axes = [subfig.subplots(nrows = 1, ncols = ii+1) for ii, subfig in enumerate(subfigs)]
dfanovaeeg_normal.boxplot(column = ["Primed", "Unprimed"], ax = axes[0])
dfanovaeeg_normal.hist(column = ["Primed", "Unprimed"], ax = axes[1], edgecolor = "black")

#%% Combined stats
dfanovaall = pd.concat([dfanova[leadfieldname].drop(15, axis = 0).reset_index(drop = True), dfanovaeeg], axis = 1)
dfanovaall = dfanovaall.set_axis(["Omitted subject cEEG", "Primed cEEG", "Unprimed cEEG",
                                  "Omitted subject EEG", "Primed EEG", "Unprimed EEG"], axis = 1)
allasfig = plt.figure(constrained_layout = True)
allasfig.suptitle("Assumptions jackknife all data")
subfigs = allasfig.subfigures(nrows = 2, ncols = 1)
subfigs[0].suptitle("Variance")
subfigs[1].suptitle("Normal distributed")
axes = [subfig.subplots(nrows = 1, ncols = ii*3+1) for ii, subfig in enumerate(subfigs)]
dfanovaall.boxplot(column = ["Primed cEEG", "Unprimed cEEG", "Primed EEG", "Unprimed EEG"], ax = axes[0])
dfanovaall.hist(column = ["Primed cEEG", "Unprimed cEEG", "Primed EEG", "Unprimed EEG"], ax = axes[1], edgecolor = "black")

groupcount = dfanovaall.shape[1] - 2 # -2 for subject columns
f_stat, _ = scipy.stats.f_oneway(dfanovaall["Primed cEEG"], dfanovaall["Unprimed cEEG"], dfanovaall["Primed EEG"], dfanovaall["Unprimed EEG"])
print(f'\nF-stat all: {f_stat}')
cf_stat = f_stat/((dfanovaall.shape[0] - 1)**2)
print(f'corrected F-stat: {cf_stat}')
dfn = groupcount - 1
dfd = (dfanovaall.shape[0] * groupcount) - groupcount
cp = scipy.stats.f.sf(cf_stat, dfn, dfd)
print(f'corrected p-value: {cp}\n')
All_stats = {
            "F stat": f_stat,
            "Corrected F stat": cf_stat,
            "Corrected p-value": cp,
            "dfn": dfn,
            "dfd": dfd
    }

means_all = np.array([np.mean(arr) for arr in dfanovaall[["Primed cEEG", "Unprimed cEEG", "Primed EEG", "Unprimed EEG"]].values.T])
q_stat_crit = scipy.stats.studentized_range.ppf((1-0.05)/2, groupcount, dfd)
mse = np.mean((dfanovaall[["Primed cEEG", "Unprimed cEEG", "Primed EEG", "Unprimed EEG"]].values - means_all)**2)
h_stat_crit = q_stat_crit * np.sqrt(mse/dfanovaall.shape[0])
ch_stat_crit = h_stat_crit * (dfanovaall.shape[0] - 1)
print(f'Primed significant: {np.abs(means_all[0] - means_all[2]) >= ch_stat_crit}')
All_stats_Primed = {
            "Critical q stat": q_stat_crit,
            "Critical H stat": h_stat_crit,
            "Corrected Critical H stat": ch_stat_crit,
            "Significant": bool(np.abs(means_all[0] - means_all[2]) >= ch_stat_crit),
            "dfd": dfd
    }
print(f'Unprimed significant: {np.abs(means_all[1] - means_all[3]) >= ch_stat_crit}')
All_stats_Unprimed = {
            "Critical q stat": q_stat_crit,
            "Critical H stat": h_stat_crit,
            "Corrected Critical H stat": ch_stat_crit,
            "Significant": bool(np.abs(means_all[1] - means_all[3]) >= ch_stat_crit),
            "dfd": dfd
    }


# t_stat, _ = scipy.stats.ttest_ind(dfanovaall["Primed cEEG"], dfanovaall["Primed EEG"])
# f_stat = t_stat**2
# print(f'\nF-stat Primed cEEG - EEG: {f_stat}')
# cf_stat = f_stat/((dfanovaall.shape[0] - 1)**2)
# print(f'corrected F-stat: {cf_stat}')
# dfn = groupcount - 1
# dfd = (dfanovaall.shape[0] * groupcount) - groupcount
# cp = min([scipy.stats.f.sf(cf_stat, dfn, dfd) *12, 1]) # Bonferroni correction
# print(f'corrected p-value: {cp}\n')
# All_stats_Primed = {
#             "F stat": f_stat,
#             "Corrected F stat": cf_stat,
#             "Corrected p-value": cp,
#             "dfn": dfn,
#             "dfd": dfd
#     }

# t_stat, _ = scipy.stats.ttest_ind(dfanovaall["Unprimed cEEG"], dfanovaall["Unprimed EEG"])
# f_stat = t_stat**2
# print(f'\nF-stat Unprimed cEEG - EEG: {f_stat}')
# cf_stat = f_stat/((dfanovaall.shape[0] - 1)**2)
# print(f'corrected F-stat: {cf_stat}')
# dfn = groupcount - 1
# dfd = (dfanovaall.shape[0] * groupcount) - groupcount
# cp = min([scipy.stats.f.sf(cf_stat, dfn, dfd) *12, 1]) # Bonferroni correction
# print(f'corrected p-value: {cp}\n')
# All_stats_Unprimed = {
#             "F stat": f_stat,
#             "Corrected F stat": cf_stat,
#             "Corrected p-value": cp,
#             "dfn": dfn,
#             "dfd": dfd
#     }

#%% Save figures
with PdfPages(f'{SaveLocation}\\plots.pdf') as pdf:
    pdf.savefig(lfpfig)
    pdf.savefig(dipfig)
    pdf.savefig(ceegfig)
    pdf.savefig(ceegfigz)
    pdf.savefig(eegfig)
    pdf.savefig(eegfigz)
    pdf.savefig(ceegasfig)
    pdf.savefig(ceegasfig_n)
    pdf.savefig(eegasfig)
    pdf.savefig(eegasfig_n)
    pdf.savefig(allasfig)
    for _, leadfieldplots in waveceegfig.items():
        for _, fig in leadfieldplots.items():
            pdf.savefig(fig)
    for _, fig in waveeegfig.items():
        pdf.savefig(fig)
    for _, leadfieldplots in peakceegfig.items():
        for _, fig in leadfieldplots.items():
            pdf.savefig(fig)
    for _, fig in peakeegfig.items():
        pdf.savefig(fig)

#%% Save variables
jsonsave = {
        "Plotting":
            {
                "Time window to plot": TimeWindow,
                "Category colors": CategoryColor,
                "Time point to clip data at": TimeClip,
                "Zoomed in plot time window": ZoomedTime,
                "Baseline time": BaselineTime
            },
        "cEEG":
            {
                "Leadfields": LeadFieldSel,
                "Electrode": Electrode
            },
        "Statistics":
            {
                "Excluded session indices cEEG": ExclusionSessionIndx,
                "Excluded session indices EEG": ExclusionSessionIndxEEG,
                "Minimum time points threshold for percent max": MinTimePoints,
                "Search interval": SearchInterval,
                "Percent max threshold": PercentMax,
                "Signal to Noise Ratio threshold": SNRThresh
            }
    }
with open(f'{SaveLocation}\\inputvariables.json', "w") as f:
    json.dump(jsonsave, f, indent = 4)

#%% Save results
for leadfieldname, lfdata in dfanova.items():
    lfdata.to_csv(f'{SaveLocation}\\cEEG_{leadfieldname}_results.csv')
dfanovaeeg.to_csv(f'{SaveLocation}\\EEG_results.csv')

#%% Save statistics
jsonsave = {
        "cEEG": cEEG_stats,
        "EEG": EEG_stats,
        "All": All_stats,
        "All-Primed": All_stats_Primed,
        "All-Unprimed": All_stats_Unprimed
    }
with open(f'{SaveLocation}\\statistics.json', "w") as f:
    json.dump(jsonsave, f, indent = 4)


plt.close("all")
plt.switch_backend('QtAgg')
