# -*- coding: utf-8 -*-
"""
Created on Fri Mar  6 13:57:13 2026

@author: LukSu
"""
import os
import numpy as np
import pandas as pd
from scipy.signal import lfilter
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import json
import shutil
from pypdf import PdfWriter

def ImportDF(path, DFHeaderSelection):
    dfs = {}
    for dfname, fileselectionlist in DFHeaderSelection.items():
        dfs[dfname] = pd.DataFrame()
        for filename, headerselection in fileselectionlist.items():
            file_lastdotindx = filename.rfind(".")
            headersfilename = filename[:file_lastdotindx] + "_header.csv"
            headers_all = np.loadtxt(f'{path}\\{headersfilename}', delimiter = ",",
                                     dtype = str)
            header_indxs = []
            for header in headerselection:
                header_indxs.append(np.where(headers_all == header)[0][0])
            match filename.split(".")[-1]:
                case "csv":
                    tmp_df = pd.read_csv(f'{path}\\{filename}', header = None,
                                         usecols = header_indxs, names = headerselection)
                case "npy":
                    tmp_data = np.load(f'{path}\\{filename}')
                    tmp_df = pd.DataFrame(np.squeeze(tmp_data[:, header_indxs]),
                                          columns = headerselection)
            dfs[dfname] = pd.concat([dfs[dfname], tmp_df], axis = 1)
    return dfs
def ImportArray(path):
    array = np.load(path)
    # To fix an issue were some arrays have 17 channels, while they should have only 15
    shape = array.shape
    if shape[0] == 17:
        array = array[1:-1, :, :]
    return array

def ImportProbes(path):
    probeheaders = np.loadtxt(f'{path[:-4]}_header.csv', delimiter = ",", dtype = str)
    probes = pd.read_csv(path, header = None, names = probeheaders, index_col = 0).to_dict()
    return probes

def ValidationPrint(errorlist, monkeyid, date):
    print("\n\033[91mErrors have been found:\033[0m")
    print(f'\t\033[92mMacaca: \033[0m{monkeyid} \033[92mDate: \033[0m{date}')
    for dfname, valuedict in errorlist.items():
        print(f'\t\t\033[96mDataframe:\033[0m {dfname}')
        for key, value in valuedict.items():
            # Lists are, where possible, saved as pandas series to save indices
            #   This does lead to slightly worse readability in the next print
            #   statement when it comes to lists.
            print(f'\t\t\t\033[96mKey: \033[0m{key}\n\t\t\t\t\033[91mEncountered value: \033[0m{value[0]}')
            print(f'\t\t\t\t\033[92mExpected value:\033[0m {value[1]}')
    print()

def CheckDataInformation(dfs, monkeyid, date, infochecks):
    print("\t\033[92mChecking data\033[0m")
    invalidlist = {}
    for dfname, valuedict in infochecks.items():
        print(f'\t\t\033[96mDataframe name: \033[0m{dfname}')
        for key, value in valuedict.items():
            match value:
                case dict():
                    if (dfs[dfname][key].to_numpy() != value[monkeyid]).any():
                        invalidlist.setdefault(dfname, {})
                        invalidlist[dfname][key] = [dfs[dfname][key][0], value[monkeyid]]
                case list():
                    if (dfs[dfname][key].to_numpy() != date).any():
                        invalidlist.setdefault(dfname, {})
                        invalidlist[dfname][key] = [dfs[dfname][key][0], date]
                case _:
                    if (dfs[dfname][key].to_numpy() != value).any():
                        invalidlist.setdefault(dfname, {})
                        invalidlist[dfname][key] = [dfs[dfname][key][dfs[dfname][key].to_numpy() != value],
                                                                    value]
    if invalidlist:
        ValidationPrint(invalidlist, monkeyid, date)
    return invalidlist

def TrialSelection(dfs, dfname, array, eeg, dataselection):
    removelater = {}
    for key, value in dataselection.items():
        match value:
            case dict():
                indxrmv = np.zeros(dfs[dfname][key].to_numpy().shape)
                threshold = len(value.keys())
                for sign, val in value.items():
                    match sign:
                        case "<=":
                            indxrmv += np.array(dfs[dfname][key].to_numpy() <= val, dtype = np.int8)
                        case ">=":
                            indxrmv += np.array(dfs[dfname][key].to_numpy() >= val, dtype = np.int8)
                indxrmv[indxrmv < threshold] = 0
                indxrmv[indxrmv >= threshold] = 1
                indxrmv = indxrmv.astype(bool)
                indxrmv = np.logical_not(indxrmv)
                indices = dfs[dfname][key][indxrmv].index.to_numpy()
                removelater.setdefault(dfname, set())
                removelater[dfname].update(set(indices))
            case list():
                if np.array(pd.Index(pd.unique(np.array(value))).get_indexer(dfs[dfname][key].to_numpy()) < 0).any():
                    removelater.setdefault(dfname, set())
                    removelater[dfname].update(set((dfs[dfname][key][np.array(pd.Index(pd.unique(np.array(value))).get_indexer(dfs[dfname][key].to_numpy()) < 0)]).index.to_numpy()))
            case _:
                if (dfs[dfname][key].to_numpy() != value).any():
                    removelater.setdefault(dfname, set())
                    removelater[dfname].update(set((dfs[dfname][key][dfs[dfname][key].to_numpy() != value]).index.to_numpy()))
    for dfname, valueset in removelater.items():
        indicestoremove = np.sort(np.fromiter(valueset, dtype = np.int16))
        dfs[dfname] = dfs[dfname].drop(index = indicestoremove)
        dfs[dfname].reset_index(drop = True, inplace = True)
        array = np.delete(array, indicestoremove, axis = 2)
        eeg = np.delete(eeg, indicestoremove, axis = 2)
    return dfs, array, eeg

def SelectData(df, array, eeg, DataSelection):
    print("\t\033[92mSelecting data\033[0m")
    for dfname, selectioncriteria in DataSelection.items():
        match dfname:
            case "SessionInfo":
                for key, item in selectioncriteria.items():
                    if df["SessionInfo"][key][0] not in item:
                        continue
            case "Data":
                df, array, eeg = TrialSelection(df, dfname, array, eeg, selectioncriteria)
    return df, array, eeg

def PlotAverageToPdf(pdfname, exportfolder, data, time, xlimitlower = None, xlimitupper = None, sesmon = "Session"):
    print(f'\t\t\033[96mExporting to: \033[0m{exportfolder}')
    with PdfPages(f'{exportfolder}\\{pdfname}') as pdf:
        fig, axes = plt.subplots(nrows = data.shape[0], figsize = (len(time)/200, data.shape[0]))
        for channelindx, ax in enumerate(axes):
            ax.plot(time, np.nanmean(data[channelindx,:,:], axis = 1))
        for ax in axes[1:]:
            ax.sharex(axes[0])
        if xlimitupper:
            axes[0].set_xlim(right = xlimitupper)
        if xlimitlower:
            axes[0].set_xlim(left = xlimitlower)
        fig.suptitle(f'{sesmon} average: {pdfname[:-4]}')
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

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

def find_nearest(array, value):
    indx = (np.abs(array - value)).argmin()
    return indx, array[indx]

def PlotCSDtoPdf(pdfname, exportfolder, data, time, xlimitlower = None, xlimitupper = None, sesmon = "Session"):
    print(f'\t\t\033[96mExporting to: \033[0m{exportfolder}')
    resolution = 10
    csd = np.nanmean(data, axis = 2)
    fs_csd = SmoothCSD_2D(np.vstack((csd[0,:], csd, csd[-1,:])), resolution)
    with PdfPages(f'{exportfolder}\\{pdfname}') as pdf:
        fig, ax = plt.subplots()
        if xlimitupper:
            ax.set_xlim(right = xlimitupper)
        if xlimitlower:
            ax.set_xlim(left = xlimitlower)
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
        ax.figure.colorbar(im, ax=ax)
        ax.set_title(f'CSD {sesmon} average: {pdfname[:-4]}')
        ax.set_xlabel("Time from event (ms)")
        ax.set_ylabel("Lower<-Cx (Depth)->Upper")
        ax.axes.get_yaxis().set_ticks(np.arange(1, data.shape[0]+1, 1))
        
        fig.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

def DataSeparation(dfs, arrays, eegs, tmp_df, tmp_array, tmp_eeg, monkeyid, date,
                   DataSeparationConditions, MinDataPoints, DegConversion,
                   SessionAverageSelection, sfreq, catcliptimes, cliptimes):
    print("\t\033[92mSeparating Data\033[0m")
    dfs_separated = {}
    array_separated = {}
    eeg_separated = {}
    cliptimes_separated = {}
    excludedsessions = {}
    cattotalcount = len(DataSeparationConditions.keys())
    catcounter = 0
    leastcattrials = [-1, None]
    for separationcat, criteria in DataSeparationConditions.items():
        catcounter += 1
        print(f'\t\t\033[96mCategory: \033[0m{separationcat}\t({catcounter}/{cattotalcount})')
        indxrmv = np.zeros(tmp_df["Data"]["trial_number_count"].to_numpy().shape)
        threshold = 0
        for dfname, dfcriteria in criteria.items():
            match dfname:
                case "Data":
                    for key, keycriteria in dfcriteria.items():
                        match key:
                            case "array_target_position_deg":
                                degpositions = set(tmp_df[dfname][key].to_numpy())
                                maxdeg = max(degpositions)
                                degmapping = {deg:  ((deg - DegConversion)%360)%360 for deg in degpositions}
                                deglookup = np.arange(maxdeg+1)
                                for deg, adjdeg in degmapping.items():
                                    deglookup[deg] = adjdeg
                                converteddeg = deglookup[tmp_df[dfname][key].to_numpy()]
                                convdegpositions = np.fromiter(set(converteddeg),dtype=np.int16)
                                for sign, value in keycriteria[monkeyid].items():
                                    match sign:
                                        case "<":
                                            deginclusion = np.array(convdegpositions[convdegpositions < keycriteria[monkeyid]["<"]])
                                        case ">":
                                            deginclusion = np.array(convdegpositions[convdegpositions > keycriteria[monkeyid][">"]])
                                        case "match":
                                            matchdegree = (tmp_df["SessionInfo"][value].to_numpy() - DegConversion)%360
                                            deginclusion = np.array(convdegpositions[convdegpositions == matchdegree])
                                        case "mismatch":
                                            matchdegree = (tmp_df["SessionInfo"][value].to_numpy() - DegConversion)%360
                                            deginclusion = np.array(convdegpositions[convdegpositions != matchdegree])
                                    indxrmv += np.array(pd.Index(pd.unique(deginclusion)).get_indexer(converteddeg) >= 0)
                                    threshold += 1
                            case _:
                                for sign, value in keycriteria.items():
                                    threshold += 1
                                    match sign:
                                        case "<=":
                                            indxrmv += np.array(tmp_df[dfname][key].to_numpy() <= value, dtype = np.int8)
                                        case ">=":
                                            indxrmv += np.array(tmp_df[dfname][key].to_numpy() >= value, dtype = np.int8)
                                        case "<":
                                            indxrmv += np.array(tmp_df[dfname][key].to_numpy() < value, dtype = np.int8)
                                        case ">":
                                            indxrmv += np.array(tmp_df[dfname][key].to_numpy() > value, dtype = np.int8)
        indxrmv[indxrmv < threshold] = 0
        indxrmv[indxrmv > threshold] = 1
        indxrmv = indxrmv.astype(bool)
        dfs_separated[separationcat] = tmp_df["Data"].loc[indxrmv]
        dfs_separated[separationcat].reset_index(drop = True, inplace = True)
        array_separated[separationcat] = tmp_array[:,:,indxrmv]
        eeg_separated[separationcat] = tmp_eeg[:,:,indxrmv]
        cliptimes_separated[separationcat] = cliptimes[indxrmv]
        if int(sum(indxrmv)) < leastcattrials[0] or leastcattrials[0] == -1:
            leastcattrials = [int(sum(indxrmv)), separationcat]
    if leastcattrials[0] >= MinDataPoints:
        for separationcat, df in dfs_separated.items():
            dfdropcolumns = list(dfs_separated[separationcat].keys())
            for key in SessionAverageSelection:
                dfdropcolumns.remove(key)
            df_dropped = dfs_separated[separationcat].drop(columns = dfdropcolumns)
            trialcount = df_dropped.shape[0]
            df_dropped_average = pd.DataFrame(df_dropped.mean(numeric_only = True)).T
            df_dropped_average.insert(0, "Session", f'{monkeyid}{date}')
            df_dropped_average.insert(1, "Receptive field position", tmp_df["SessionInfo"]["receptive_field_position_deg"][0])
            df_dropped_average["Trialcount"] = trialcount
            df_dropped_average["Sampling frequency"] = sfreq
            if separationcat in dfs.keys():
                dfs[separationcat] = pd.concat([dfs[separationcat], df_dropped_average])
                dfs[separationcat].reset_index(drop = True, inplace = True)
                arrays[separationcat] = np.dstack((arrays[separationcat], np.nanmean(array_separated[separationcat], axis = 2)))
                eegs[separationcat] = np.dstack((eegs[separationcat], np.nanmean(eeg_separated[separationcat], axis = 2)))
                catcliptimes[separationcat] = np.hstack((catcliptimes[separationcat], cliptimes_separated[separationcat]))
            else:
                dfs[separationcat] = df_dropped_average
                arrays[separationcat] = np.nanmean(array_separated[separationcat], axis = 2)
                eegs[separationcat] = np.nanmean(eeg_separated[separationcat], axis = 2)
                catcliptimes[separationcat] = cliptimes_separated[separationcat]
    else:
        excludedsessions[f'{monkeyid}{date}'] = leastcattrials
    return dfs, arrays, eegs, catcliptimes

def ExtractData(ImportFolder, ExportFolder, DFHeaderSelection, DataFile, TimeFile,
                InfoChecks, DataSelection, BaselineTime, MinDataPoints, DegConversion,
                DataSeparationConditions, SessionAverageSelection, ProbeFile, EEGFile):
    plt.switch_backend('pdf') # To prevent figures from opening
    # Initialise variables
    dfs = {}
    arrays = {}
    eegs = {}
    catcliptimes = {}
    monkey_array_average = {}
    monkey_array_avgsize = {}
    monkey_array_average_baseline = {}
    probes = {}
    plotssavepdfs = {}
    clippedtimepoints = {}
    if os.path.exists(ExportFolder):
        confirm = input(f'\n\n\033[91mFolder \033[96m{ExportFolder} \033[91malready exists\033[0m, overwrite? [y/n] ')
        if confirm.upper() != 'Y':
            return 1, dfs, arrays
        else:
            print("\n\n")
            shutil.rmtree(ExportFolder)
    os.makedirs(ExportFolder, exist_ok = True)
    os.makedirs(f'{ExportFolder}\\plots', exist_ok = True)
    os.makedirs(f'{ExportFolder}\\tmp\\plots', exist_ok = True)
    exportplotfolder_tmp = f'{ExportFolder}\\tmp\\plots'
    exportplotfolder = f'{ExportFolder}\\plots'
    # Get all folders in directory.
    #   If there are no folders in directory, assume the directory is the folder
    #   to be imported
    foldertotalcount = sum([1 for name in os.listdir(ImportFolder) if not os.path.isfile(f'{ImportFolder}\\{name}')])
    if foldertotalcount:
        folderlist = [name for name in os.listdir(ImportFolder) if not os.path.isfile(f'{ImportFolder}\\{name}')]
    else:
        folderlist = [ImportFolder.split("\\")[-1]]
        ImportFolder = "\\".join(ImportFolder.split("\\")[:-1])
    # Loop over folders to import one by one
    for foldercount, foldername in enumerate(folderlist):
        print(f'\033[95mImporting \033[0m{foldername}\t({foldercount+1}/{len(folderlist)})')
        # Import folder
        tmp_df = ImportDF(f'{ImportFolder}\\{foldername}', DFHeaderSelection)
        tmp_array = ImportArray(f'{ImportFolder}\\{foldername}\\{DataFile}')
        tmp_time = np.load(f'{ImportFolder}\\{foldername}\\{TimeFile}')[0]
        tmp_eeg = np.load(f'{ImportFolder}\\{foldername}\\{EEGFile}')
        monkeyid = tmp_df["SessionInfo"]["monkey_identifier"][0]
        date = tmp_df["SessionInfo"]["date"][0]
        sfreq = tmp_df["SessionInfo"]["data_sampling_rate_hz"][0]
        probes[f'{monkeyid}{date}'] = ImportProbes(f'{ImportFolder}\\{foldername}\\{ProbeFile}')
        # Check data
        CheckDataInformation(tmp_df, monkeyid, date, InfoChecks)
        # Data selection
        tmp_df, tmp_array, tmp_eeg = SelectData(tmp_df, tmp_array, tmp_eeg, DataSelection)
        # Clip data
        for indx, reactiontime in enumerate(tmp_df["Data"]["reaction_time_ms"]):
            reactiontimeindx, timearraytime = find_nearest(tmp_time, reactiontime-10) # Everything after the reactiontime - 10ms is to be set to nan
            tmp_array[:,reactiontimeindx:,indx] = np.nan
            tmp_eeg[:,reactiontimeindx:,indx] = np.nan
            clippedtimepoints.setdefault(f'{monkeyid}{date}', 0)
            if clippedtimepoints[f'{monkeyid}{date}']:
                clippedtimepoints[f'{monkeyid}{date}']["Absolute time"][indx] = reactiontime-10
                clippedtimepoints[f'{monkeyid}{date}']["Time point in time array"][indx] = timearraytime
            else:
                clippedtimepoints[f'{monkeyid}{date}'] = {"Absolute time": {indx: reactiontime-10},
                                                          "Time point in time array": {indx: timearraytime}}
        clippedmedian = np.median(np.array(list(clippedtimepoints[f'{monkeyid}{date}']["Time point in time array"].values())))
        # Add data to monkey average
        monkey_array_avgsize.setdefault(monkeyid, 0)
        if monkey_array_avgsize[monkeyid]:
            monkey_array_average[monkeyid] = np.average([monkey_array_average[monkeyid], np.nanmean(tmp_array, axis = 2, keepdims = True)],
                                                        weights = [monkey_array_avgsize[monkeyid], 1], axis = 0)
        else:
            monkey_array_average[monkeyid] = np.nanmean(tmp_array, axis = 2, keepdims = True)
        monkey_array_avgsize[monkeyid] += 1
        # Generate raw data plots
        plotssavepdfs.setdefault(monkeyid, [])
        plotssavepdfs[monkeyid].append(f'{monkeyid}{date}.pdf')
        print("\t\033[92mGenerating session average raw data plots\033[0m")
        timezeroindx = np.where(tmp_time == 0)[0][0]
        baselinestartindx, baselinetimestart = find_nearest(tmp_time, -BaselineTime)
        PlotAverageToPdf(plotssavepdfs[monkeyid][-1], exportplotfolder_tmp, tmp_array, tmp_time, xlimitupper = clippedmedian, xlimitlower = baselinetimestart)
        plotssavepdfs[monkeyid].append(f'{monkeyid}{date}_CSD.pdf')
        PlotCSDtoPdf(plotssavepdfs[monkeyid][-1], exportplotfolder_tmp, tmp_array, tmp_time, xlimitupper = clippedmedian, xlimitlower = baselinetimestart)
        # Baseline correction
        print("\t\033[92mCorrecting for baseline\033[0m")
        baselineavg = np.nanmean(tmp_array[:, baselinestartindx:timezeroindx, :], axis = 1, keepdims = True)
        tmp_array_baseline = tmp_array - baselineavg
        if monkey_array_avgsize[monkeyid] - 1:
            monkey_array_average_baseline[monkeyid] = np.average([monkey_array_average_baseline[monkeyid], np.nanmean(tmp_array_baseline, axis = 2, keepdims = True)],
                                                                 weights = [monkey_array_avgsize[monkeyid], 1], axis = 0)
        else:
            monkey_array_average_baseline[monkeyid] = np.nanmean(tmp_array_baseline, axis = 2, keepdims = True)
        baselineavgeeg = np.nanmean(tmp_eeg[:, baselinestartindx:timezeroindx, :], axis = 1, keepdims = True)
        tmp_eeg_baseline = tmp_eeg - baselineavgeeg
        # Generate baseline corrected plot
        plotssavepdfs[monkeyid].append(f'{monkeyid}{date}_baseline.pdf')
        print("\t\033[92mGenerating session average baseline corrected plots\033[0m")
        PlotAverageToPdf(plotssavepdfs[monkeyid][-1], exportplotfolder_tmp, tmp_array_baseline, tmp_time, xlimitupper = clippedmedian, xlimitlower = baselinetimestart)
        plotssavepdfs[monkeyid].append(f'{monkeyid}{date}_baseline_CSD.pdf')
        PlotCSDtoPdf(plotssavepdfs[monkeyid][-1], exportplotfolder_tmp, tmp_array_baseline, tmp_time, xlimitupper = clippedmedian, xlimitlower = baselinetimestart)
        # Separate data into given categories
        dfs, arrays, eegs, catcliptimes = DataSeparation(dfs, arrays, eegs, tmp_df, tmp_array_baseline, tmp_eeg_baseline, monkeyid, date, DataSeparationConditions, MinDataPoints, DegConversion, SessionAverageSelection,
                                                  sfreq, catcliptimes, np.array(list(clippedtimepoints[f'{monkeyid}{date}']["Time point in time array"].values())))
    print()
    for cat, catclips in catcliptimes.items():
        catcliptimes[cat] = np.median(catclips)
    clippedmedians = {}
    for session, clippedtimes in clippedtimepoints.items():
        monkeyid = session[0]
        clippedmedians.setdefault(monkeyid, np.uint8(0))
        if clippedmedians[monkeyid].any():
            clippedmedians[monkeyid] = np.hstack((clippedmedians[monkeyid], np.array(list(clippedtimes["Time point in time array"].values()))))
        else:
            clippedmedians[monkeyid] = np.array(list(clippedtimes["Time point in time array"].values()))
    for monkeyid, cliptimes in clippedmedians.items():
        clippedmedians[monkeyid] = np.median(np.array(list(clippedmedians[monkeyid])))
    # Save session average plots to pdf
    for monkeyid in plotssavepdfs.keys():
        print(f'\033[95mGenerating monkey average plots:\033[0m Monkey {monkeyid}')
        exportplotfolder_tmp = f'{ExportFolder}\\tmp\\plots'
        plotssavepdfs.setdefault(monkeyid, [])
        plotssavepdfs[monkeyid].append(f'{monkeyid}_average_raw.pdf')
        print("\t\033[92mRaw data average plot\033[0m")
        PlotAverageToPdf(plotssavepdfs[monkeyid][-1], exportplotfolder_tmp, monkey_array_average[monkeyid], tmp_time, baselinetimestart, clippedmedians[monkeyid], "Monkey")
        plotssavepdfs[monkeyid].append(f'{monkeyid}_average_raw_CSD.pdf')
        PlotCSDtoPdf(plotssavepdfs[monkeyid][-1], exportplotfolder_tmp, monkey_array_average[monkeyid], tmp_time, baselinetimestart, clippedmedians[monkeyid], "Monkey")
        plotssavepdfs[monkeyid].append(f'{monkeyid}_average_baseline.pdf')
        print("\t\033[92mBaseline corrected average plot\033[0m")
        PlotAverageToPdf(plotssavepdfs[monkeyid][-1], exportplotfolder_tmp, monkey_array_average_baseline[monkeyid], tmp_time, baselinetimestart, clippedmedians[monkeyid], "Monkey")
        plotssavepdfs[monkeyid].append(f'{monkeyid}_average_baseline_CSD.pdf')
        PlotCSDtoPdf(plotssavepdfs[monkeyid][-1], exportplotfolder_tmp, monkey_array_average_baseline[monkeyid], tmp_time, baselinetimestart, clippedmedians[monkeyid], "Monkey")
    # Save condition average plots to pdf
    for arrayname, array in arrays.items():
        print(f'\033[95mGenerating condition average plots:\033[0m {arrayname}')
        if len(array.shape) < 3:
            array = np.expand_dims(array, axis = 2)
        exportplotfolder_tmp = f'{ExportFolder}\\tmp\\plots'
        plotssavepdfs.setdefault(arrayname, [])
        plotssavepdfs[arrayname].append(f'{arrayname}_average.pdf')
        PlotAverageToPdf(plotssavepdfs[arrayname][-1], exportplotfolder_tmp, array, tmp_time, baselinetimestart, catcliptimes[arrayname], sesmon=arrayname)
        plotssavepdfs[arrayname].append(f'{arrayname}_average_CSD.pdf')
        PlotCSDtoPdf(plotssavepdfs[arrayname][-1], exportplotfolder_tmp, array, tmp_time, baselinetimestart, catcliptimes[arrayname], sesmon=arrayname)
    # Merge pdfs
    for monkeyid, pdflist in plotssavepdfs.items():
        print(f'\033[95mMerging plot pdfs:\033[0m Monkey {monkeyid}')
        exportplotfile = f'{exportplotfolder}\\{monkeyid}.pdf'
        writer = PdfWriter()
        for pdffile in pdflist:
            writer.append(f'{exportplotfolder_tmp}\\{pdffile}')
        with open(exportplotfile, 'wb') as f:
            writer.write(f)
    # Save separated data
    print("\033[95mSaving data\033[0m")
    jsonsave = {"Header selection": DFHeaderSelection,
                "Data filename": DataFile,
                "Time filename": TimeFile,
                "Used checks": InfoChecks,
                "Data selection": DataSelection,
                "Baseline (ms)": {"Duration": BaselineTime,
                                         "Start": tmp_time[baselinestartindx],
                                         "Stop": tmp_time[timezeroindx]},
                "Minimum datapoints per session per category": MinDataPoints,
                "Degree conversion": DegConversion,
                "Data separation categories": DataSeparationConditions}
    print("\t\033[92mSaving dataframes\033[0m")
    # Save dataframes
    for dfname, dfdata in dfs.items():
        dfdata.to_csv(f'{ExportFolder}\\{dfname}.csv')
    probes_flattened = {(session, header): values for session, innerdict in probes.items() for header, values in innerdict.items()}
    probes_multiindex = pd.MultiIndex.from_tuples(probes_flattened.keys())
    probes_df = pd.DataFrame(list(probes_flattened.values()), index = probes_multiindex)
    probes_df.to_csv(f'{ExportFolder}\\ProbeDistances.csv')
    clippedtimepoints = pd.DataFrame.from_dict({(outer, inner): values for outer, inner_dict in clippedtimepoints.items() for inner, values in inner_dict.items()}, orient = "index")
    clippedtimepoints.index = pd.MultiIndex.from_tuples(clippedtimepoints.index, names = ["Session", "Time type"])
    clippedtimepoints.to_csv(f'{ExportFolder}\\ClipTimePoints.csv')
    print("\t\033[92mSaving arrays\033[0m")
    for arrayname, array in arrays.items():
        np.save(f'{ExportFolder}\\{arrayname}', array, allow_pickle = False)
    print("\t\033[92mSaving eegs\033[0m")
    for eegname, eeg in eegs.items():
        np.save(f'{ExportFolder}\\EEG_{eegname}', eeg, allow_pickle = False)
    np.save(f'{ExportFolder}\\TimeArray.npy', tmp_time, allow_pickle = False)
    print("\t\033[92mSaving other variables to json\033[0m")
    with open(f'{ExportFolder}\\inputvariables.json', "w") as f:
        json.dump(jsonsave, f, indent = 4)
    print("\033[95mClearing temporary files\033[0m")
    shutil.rmtree(f'{ExportFolder}\\tmp')
    plt.switch_backend('QtAgg')
    return 0, dfs, arrays
