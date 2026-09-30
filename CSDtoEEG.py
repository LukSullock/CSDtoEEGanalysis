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
import os
import sys
import numpy as np
import pandas as pd
from scipy.io import savemat
import matplotlib.pyplot as plt
import mne
import shutil
import time
plt.switch_backend('pdf') # To prevent figures from opening

DataLocation = "F:\\saveddata\\260520-All"
SaveLocation = "F:\\SavedEEG\\260525-All"
DataFiles = {
    "Primed": {"Contra": "TargetPrimed.npy",
               "Ipsi": "DistractorPrimed.npy"},
    "Unprimed": {"Contra": "TargetUnprimed.npy",
                 "Ipsi": "DistractorUnprimed.npy"}
    }
ProbeFile = "ProbeDistances.csv"
TimeFile = "TimeArray.npy"
LeadFileLocation = "F:\\LeadFiles"
ChannelFile = "F:\\SavedEEG\\channel.npy"

CorticalColumnRadius = 1.5


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
os.makedirs(f'{SaveLocation}\\npy', exist_ok = True)
os.makedirs(f'{SaveLocation}\\npy_dipole', exist_ok = True)
os.makedirs(f'{SaveLocation}\\mat', exist_ok = True)
os.makedirs(f'{SaveLocation}\\mne', exist_ok = True)

def find_nearest(array, value):
    indx = (np.abs(array - value)).argmin()
    return indx, array[indx]

LeadFiles = {filename[:-7].split("_")[-1]: np.load(f'{LeadFileLocation}\\{filename}') for filename in os.listdir(LeadFileLocation)}
Data = {}
EEGData = {}
SessionInformation = {}
for category, files in DataFiles.items():
    Data.setdefault(category, {})
    EEGData.setdefault(category, {})
    SessionInformation.setdefault(category, {})
    for hemifield, filename in files.items():
        Data[category][hemifield] = np.load(f'{DataLocation}\\{filename}')
        EEGData[category][hemifield] = np.load(f'{DataLocation}\\EEG_{filename}')
        SessionInformation[category][hemifield] = pd.read_csv(f'{DataLocation}\\{filename[:-4]}.csv', index_col = 0)
ElectrodeCount = Data[category][hemifield].shape[0]
TimeArray = np.load(f'{DataLocation}\\{TimeFile}')/1000 # s
ProbeInformation = pd.read_csv(f'{DataLocation}\\{ProbeFile}', index_col = [0,1])
ProbeDict = {}
for (session, key), valuedict in ProbeInformation.T.to_dict().items():
    ProbeDict.setdefault(session, {"Indices": np.array(list(valuedict.keys())).astype(np.int_)})
    try:
        ProbeDict[session][key] = np.array(list(valuedict.values())).astype(np.float64)
    except ValueError:
        ProbeDict[session][key] = np.array(list(valuedict.values()))

UnitConversion = 1e-6 * (np.pi * CorticalColumnRadius**2) # From nA/mm3 to nA/m

DipoleAvg = {}
cEEGData = {}
cEEGDataAvg = {}
for category, categorydata in Data.items():
    cEEGData.setdefault(category, {})
    cEEGDataAvg.setdefault(category, {})
    CSDDipole = np.empty((2,categorydata[list(categorydata.keys())[0]].shape[1]))
    electrodepositions = ProbeDict[list(ProbeDict.keys())[0]]["position_from_L4_boundary_mm"]
    electrodedis = np.mean(np.diff(np.array([ProbeDict[list(ProbeDict.keys())[0]]["position_from_top_mm"]])))
    if len(categorydata["Contra"].shape) < 3:
        categorydata["Contra"] = np.expand_dims(categorydata["Contra"], axis = 2)
        categorydata["Ipsi"] = np.expand_dims(categorydata["Ipsi"], axis = 2)
    
    CSDDipoleSes = np.empty((2,*categorydata[list(categorydata.keys())[0]].shape[1:][::-1]))
    CSDDipoleSes[0,:,:] = np.array([UnitConversion * (electrodepositions * electrodedis).dot(array) for array in categorydata["Contra"].transpose(2, 0, 1)])
    CSDDipoleSes[1,:,:] = np.array([UnitConversion * (electrodepositions * electrodedis).dot(array) for array in categorydata["Ipsi"].transpose(2, 0, 1)])
    
    cliptime = np.median(SessionInformation[category]["Contra"]["reaction_time_ms"])/1000-0.01
    clipindx, cliptime = find_nearest(TimeArray, cliptime)
    meandata = np.nanmean(categorydata["Contra"], axis = 2)
    meandata[:,clipindx:] = np.nan
    CSDDipole[0,:] = UnitConversion * (electrodepositions * electrodedis).dot(meandata)
    cliptime = np.median(SessionInformation[category]["Ipsi"]["reaction_time_ms"])/1000-0.01
    clipindx, cliptime = find_nearest(TimeArray, cliptime)
    meandata = np.nanmean(categorydata["Ipsi"], axis = 2)
    meandata[:,clipindx:] = np.nan
    CSDDipole[1,:] = UnitConversion * (electrodepositions * electrodedis).dot(meandata)
    DipoleAvg[category] = CSDDipole
    np.save(f'{SaveLocation}\\npy_dipole\\dipole_{category}', CSDDipole)
    for leadfieldname, leadfield in LeadFiles.items():
        cEEGLeft = leadfield.dot(CSDDipole)
        cEEGRight = leadfield.dot(CSDDipole[[1,0],:])
        cEEGDataAvg[category][leadfieldname] = {"Contra": cEEGLeft, "Ipsi": cEEGRight, "Diff": cEEGLeft - cEEGRight}
        for key, data in cEEGDataAvg[category][leadfieldname].items():
            np.save(f'{SaveLocation}\\npy\\cEEG_{category}_Avg_{leadfieldname}_{key}.npy', data)
            savemat(f'{SaveLocation}\\mat\\cEEG_{category}_Avg_{leadfieldname}_{key}.mat', {"data": data})
        cEEGLeft = np.array([leadfield.dot(dipole) for dipole in CSDDipoleSes.transpose(1,0,2)])
        cEEGRight = np.array([leadfield.dot(dipole[[1,0],:]) for dipole in CSDDipoleSes.transpose(1,0,2)])
        cEEGData[category][leadfieldname] = {"Contra": cEEGLeft, "Ipsi": cEEGRight, "Diff": cEEGLeft - cEEGRight}
        for key, data in cEEGData[category][leadfieldname].items():
            np.save(f'{SaveLocation}\\npy\\cEEG_{category}_{leadfieldname}_{key}.npy', data)
            savemat(f'{SaveLocation}\\mat\\cEEG_{category}_{leadfieldname}_{key}.mat', {"data": data})

#%%
channels = np.load(ChannelFile, allow_pickle = True).item()
channel_types = [electrode["Type"].lower() for electrode in channels.values()]
channelmapping = {f'{int(key)-1}': electrode["Name"] for key, electrode in channels.items()}
timezeroindx = int(len(TimeArray)/2)

for category, categorydata in cEEGData.items():
    for leadfieldname, leadfielddata in categorydata.items():
        for locname, locdata in leadfielddata.items():
            n_channels = leadfielddata[locname].shape[1]
            nonnandata = leadfielddata[locname][:,:,~np.isnan(leadfielddata[locname]).any(axis=(0,1))]
            baselineindx, baselinetime = find_nearest(TimeArray, -0.1)
            nonnandata = nonnandata[:,:,baselineindx:]
            sampling_freq = SessionInformation[category]["Contra"]["Sampling frequency"][0]
            info = mne.create_info(n_channels, ch_types = channel_types, sfreq = sampling_freq)
            info.rename_channels(channelmapping)
            info.set_montage("standard_1020")
            data = mne.EpochsArray(nonnandata[:,:,:]*1e-6, info, tmin = baselinetime) # convert uV to V
            data.save(f'{SaveLocation}\\mne\\cEEG_{category}_{leadfieldname}_{locname}_epo.fif', overwrite = True, verbose = "CRITICAL")

#%%
for category, categorydata in EEGData.items():
    for hemifield, hemidata in categorydata.items():
        baselineindx, baselinetime = find_nearest(TimeArray, -0.1)
        nonnandata = hemidata[:,~np.isnan(hemidata).any(axis=(0,2)),:]
        data = nonnandata[:, baselineindx:, :]
        np.save(f'{SaveLocation}\\npy\\EEG_{category}_{hemifield}.npy', data, allow_pickle = False)
        savemat(f'{SaveLocation}\\mat\\EEG_{category}_{hemifield}.mat', {"data": data})

plt.switch_backend('QtAgg')

# Average trialcount -> number of sessions in plot


# Plg142V4
# Linear summation (last step before plotting)
# V4 alone best representation of the data measured


# CSD plots, dipole, waveform and waveform zoomed in
# Check statistics with dummy data

# Check EEG and CSD data comparison together if significant or samelike


# Report
#   look at cognitive neuroscience/neuroscience for opmaak guidelines
#   ~4000 words; 1-2x word limits of guidelines



# statistic boxplot means with the session onset

# Presentation
#   For non-EEG experts
#   Include MNE joint plots (prob 1 timepoint)


# Cognitive neuroscience westerberg 2022