import h5py
import pandas as pd
import numpy as np


filename = 'P:/REALTEK-HeadNeck-Project/Masteroppgaver_2024/Torjus/HNC dataset/outcome_ous.h5'

# see the file structure
with h5py.File(filename, 'r') as f:
    for key in f.keys():
        print(key)
        for ds in f[key].keys():
            print('---', ds, f[key][ds])


# read the ous survival data
df = pd.read_csv('P:/REALTEK-HeadNeck-Project/Masteroppgaver_2024/Torjus/HNC dataset/response_ous.csv', delimiter=';')
df[df.patient_id==4]

for i in range(5):
    with h5py.File(filename, 'r') as f:
        pids = f[f'fold_{i}']['patient_idx'][:]
    OS = []
    DFS = []
    LRC = []
    for pid in pids:
        selected_item = df[df.patient_id==pid]
        OS.append([selected_item.event_OS.values[0], selected_item.OS.values[0]])
        DFS.append([selected_item.event_DFS.values[0], selected_item.DFS.values[0]])
        LRC.append([selected_item.event_LRC.values[0], selected_item.LRC.values[0]])
    with h5py.File(filename, 'a') as f:
        f[f'fold_{i}'].create_dataset('OS_surv', data=np.array(OS), dtype='f4')
        f[f'fold_{i}'].create_dataset('DFS_surv', data=np.array(DFS), dtype='f4')
        f[f'fold_{i}'].create_dataset('LRC_surv', data=np.array(LRC), dtype='f4')





filename = 'P:/REALTEK-HeadNeck-Project/Masteroppgaver_2024/Torjus/HNC dataset/outcome_maastro.h5'

# see the file structure
with h5py.File(filename, 'r') as f:
    for key in f.keys():
        print(key)
        for ds in f[key].keys():
            print('---', ds, f[key][ds])

# check pid
with h5py.File(filename, 'r') as f:
    for key in f.keys():
        print(f[key]['patient_idx'][:])

# read the ous survival data
df = pd.read_csv('P:/REALTEK-HeadNeck-Project/Masteroppgaver_2024/Torjus/HNC dataset/maastro_response_full.csv')
df[df.patient_id==4]

for i in range(4):
    with h5py.File(filename, 'r') as f:
        pids = f[f'fold_{i}']['patient_idx'][:]
    OS = []
    DFS = []
    LRC = []
    for pid in pids:
        selected_item = df[df.patient_id==pid]
        OS.append([selected_item.OS_event.values[0], selected_item.OS.values[0]])
        DFS.append([selected_item.DFS_event.values[0], selected_item.DFS.values[0]])
        LRC.append([selected_item.LRC_event.values[0], selected_item.LRC.values[0]])
    with h5py.File(filename, 'a') as f:
        f[f'fold_{i}'].create_dataset('OS_surv', data=np.array(OS), dtype='f4')
        f[f'fold_{i}'].create_dataset('DFS_surv', data=np.array(DFS), dtype='f4')
        f[f'fold_{i}'].create_dataset('LRC_surv', data=np.array(LRC), dtype='f4')
