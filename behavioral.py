# -*- coding: utf-8 -*-
"""
Created on Wed Aug 19 22:02:58 2026

@author: LukSu
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

plt.switch_backend('pdf') # To prevent figures from opening

DataLocationLFP = "F:\\saveddata\\260520-All"
SaveFile = "F:\\behavioral.pdf"

DataFiles = {
    "Primed": {"Contra": "TargetPrimed.npy",
               "Ipsi": "DistractorPrimed.npy"},
    "Unprimed": {"Contra": "TargetUnprimed.npy",
                 "Ipsi": "DistractorUnprimed.npy"}
    }
SessionInformation = {}
for category, files in DataFiles.items():
    SessionInformation.setdefault(category, {})
    for hemifield, filename in files.items():
        SessionInformation[category][hemifield] = pd.read_csv(f'{DataLocationLFP}\\{filename[:-4]}.csv', index_col = 0)

data = pd.DataFrame()
for ii, (cond, conddata) in enumerate(SessionInformation.items()):
    for jj, (hemifield, hemidata) in enumerate(conddata.items()):
        data[f'{cond}_{hemifield}'] = hemidata["reaction_time_ms"]

f, ax = plt.subplots(1,1, constrained_layout = True)
data.boxplot(ax = ax)

for cond in data:
    print(f'mean {cond}: {np.mean(data[cond])}')
    print(f'se {cond}: {np.std(data[cond])/(np.sqrt(data[cond].size))}')

with PdfPages(SaveFile) as pdf:
    pdf.savefig(f)

plt.close("all")
plt.switch_backend('QtAgg')