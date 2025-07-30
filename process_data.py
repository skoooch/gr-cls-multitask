from os import remove
import numpy as np
from scipy.signal import butter, sosfilt, filtfilt
import scipy.io

"""
This script calculates the RSMs for the EEG data.

"""
# Define the band-pass filter function
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyquist = 0.5 * fs  # Nyquist frequency
    low = lowcut / nyquist
    high = highcut / nyquist
    # Create a Butterworth band-pass filter
    b, a = butter(order, [low, high], btype='band')
    return b, a

def apply_bandpass_filter(data, lowcut, highcut, fs, order=4):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    # Apply the filter using filtfilt (zero-phase filtering)
    y = filtfilt(b, a, data, axis=1)  # Filter along the time dimension
    return y

def get_data(task, avr=False):
    lowcut = 0.2  # Low cutoff frequency (Hz)
    highcut = 115  # High cutoff frequency (Hz)
    fs = 512  # Sampling frequency (Hz), adapt this to your actual EEG data sampling rate
    order = 4  # Filter order (typically between 3-5)
    categories = ['figurine', 'pen', 'chair', 'lamp', 'plant']
    all_data = {}
    participant = ["8","9"]
    for category in categories:
        all_data[category] = []
        for i in range(1, 6):
            data = np.loadtxt('data/%s_%s_%s%s.csv' % (category, participant[1], task, i), delimiter=',') 
            num_rows = data.shape[0]
            data = apply_bandpass_filter(data, lowcut, highcut, fs, order=order)
            data = data.reshape(num_rows, 307, 64)
            # average across trials
            averaged_trials = np.mean(data, axis=0)[None, :, :]
            # Append data
            all_data[category].append(averaged_trials)
    return all_data

def get_data_matlab(task,avr=True, left = False):
    categories = ['figurine', 'pen', 'chair', 'lamp', 'plant']
    dataset = []
    if task == "grasp":
        removed_participants = [2,11,7,3,10,12,13] # add more participants as needed
        # removed_participants = [2,10,12,13]
    else:
        removed_participants = [1,10,12,13, 16]
        #removed_participants = [1,2,11,7,3,10,12,13]
    all_data = {}
    if not avr:
        all_data = []
        
    for i in range(1,17):
        if i not in removed_participants:
            mat = scipy.io.loadmat(f'matlab_files/{task}_erps/kirtan_exp_{i}_{task}.mat')
            dataset.append(mat)
            if not avr: all_data.append({})
    if avr:
        for category in categories:
            all_data[category] = []
            for i in range(1, 6):
                object_to_average_over_exp = []
                for p, file in enumerate(dataset):
                    data = file[category][f"ob{i}"][0][0]
                    data = data.transpose(0, 2, 1)
                    num_trials, num_timepoints, num_channels = data.shape
                    if(np.where(data) == 0):
                        print("here")
                    # Determine the number of new trials after averaging every 4
                    # Initialize the array to hold the averaged data
                    averaged_trials = np.mean(data, axis=0)[None, :, :]
                    # Append data
                    object_to_average_over_exp.append(averaged_trials)
                concat_begin = object_to_average_over_exp[0]
                for j in range(1, len(object_to_average_over_exp)):
                    concat_begin = np.concatenate((concat_begin, object_to_average_over_exp[j]), axis=0)
                summed = concat_begin.sum(axis=0)/len(object_to_average_over_exp)
                # 20:33 + 56: is all the back
                
                if not left: summed = np.concatenate((summed[:, 20:33], summed[:, 56:]), axis=1)
                else: summed = summed[:, 20:33]
                assert(summed.shape[0] == 307)
                all_data[category].append(summed)
        return all_data
    else:
        for p, file in enumerate(dataset):
            for category in categories:
                all_data[p][category] = []
                for i in range(1, 6):
                    data = file[category][f"ob{i}"][0][0]
                    data = data.transpose(0, 2, 1)
                    num_trials, num_timepoints, num_channels = data.shape
                    if(np.where(data) == 0):
                        print("here")

                    # Determine the number of new trials after averaging every 4
                    # Initialize the array to hold the averaged data
                    summed = np.mean(data, axis=0)[:, :]
                                        
                    if not left: summed = np.concatenate((summed[:, 20:33], summed[:, 56:]), axis=1)
                    else: summed = summed[:, 20:33]
                    assert(summed.shape[0] == 307)
                    all_data[p][category].append(summed)
        return all_data
            
# # this function was just to test whether the new data aligned with the new stuff (it did not...)
# def test_data():
#     categories = ['figurine', 'pen', 'chair', 'lamp', 'plant']
#     all_data = {}
#     dataset = []
#     removed_participants = [13] # add more participants as needed
#     mat = scipy.io.loadmat(f'matlab_files/classification_erps/exp_8_class.mat')
#     dataset.append(mat)
#     for category in categories:
#         all_data[category] = []
#         for i in range(1, 6):
#             object_to_average_over_exp = []
#             for file in dataset:
#                 data = file[category][f"ob{i}"][0][0]
#                 data = data.transpose(0, 2, 1)
#                 # Determine the number of new trials after averaging every 4
#                 # Initialize the array to hold the averaged data
#                 averaged_trials = np.mean(data, axis=0)[None, :, :]
#                 # Append data
#                 object_to_average_over_exp.append(averaged_trials)
#             concat_begin = object_to_average_over_exp[0]
#             for j in range(1, len(object_to_average_over_exp)):
#                 concat_begin = np.concatenate((concat_begin, object_to_average_over_exp[j]), axis=0)
#             summed = concat_begin.sum(axis=0)
#             all_data[category].append(summed)
#     all_data_other = {}
#     participant = ["8","9"]
#     for category in categories:
#         all_data_other[category] = []
#         for i in range(1, 6):
#             data = np.loadtxt('data/%s_%s_%s%s.csv' % (category, participant[1], "cls", i), delimiter=',') 
#             num_rows = data.shape[0]
#             data = data.reshape(num_rows, 307, 64)
#             # average across trials
#             averaged_trials = np.mean(data, axis=0)[None, :, :]
#             # Append data
#             all_data_other[category].append(averaged_trials)
#     print(np.array_equal(all_data_other["figurine"][0][0, :,:],all_data["figurine"][0]) )
#     print(all_data_other["figurine"][0][0][:, 0].sum())
#     print(all_data["figurine"][0][:, 0].sum())
