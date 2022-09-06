import argparse

from scipy import signal
from sklearn.cluster import KMeans

import numpy as np
import matplotlib.pyplot as plt



# READ FILE FUNCITONS
def read_annotation(good_trace_file):
    f = open(good_trace_file, 'r')
    label = set()
    for line in f:
        label.add(int(line.strip()))
    f.close()
    return label


def read_pattern(pattern_csv_file):
    f = open(pattern_csv_file, 'r')
    matrix_data = []
    for line in f:
        list_data = [np.double(k) for k in line.strip().split(',')]
        matrix_data.append(list_data)
    f.close()
    return matrix_data


# DATA PROCESS UTILITY FUNCTIONS
def median_filter(data_array, bin_len=10):
    list_median_array = []
    for idx in range(len(data_array) - bin_len):
        list_median_array.append(np.mean(data_array[idx:idx+bin_len]))
    return np.array(list_median_array)


def filter_topN(data_array, topN=0.05):
    sorted_array = sorted(data_array)
    len_topN = int(len(data_array)*topN)
    cut_off = np.mean(sorted_array[-len_topN:])
    np_array = np.array(data_array)
    return np_array[np.where(np_array < cut_off)]


def check_categories(list_avg_count):
    for idx in range(len(list_avg_count)-1):
        if list_avg_count[idx+1][0] - list_avg_count[idx][0] < 30:
            return False
    for avg, count in list_avg_count:
        if count < 100:
            return False
    return True


# CATEGORRY VARIFICATION FUNCTIONS
def check_peaks_1(list_data, debug=False):
    """
    Check if it is a legit 1 stage data pattern
    Find the peaks (place > avg + 4*std) and return the number of peaks
    """
    np_array = np.array(list_data)
    filtered_array = sorted(np_array)[:int(len(np_array)*0.9)]
    avg = np.mean(np_array)
    std = np.std(np_array)
    count_peak = 0

    idx = 20
    flag_peak = True
    while idx < len(np_array):
        if flag_peak:
            if np_array[idx] > avg + 4*std: # peak
                pass
            else:
                flag_peak = False
                idx += 10
        else:
            if np_array[idx] > avg + 4*std:
                flag_peak = True
                count_peak += 1
        idx += 1
    if debug:
        print(":::::: Checking Stage 1 Peaks :::::::")
        print(":::::: Peaks:", count_peak)
        print(":::::: Avg:", avg)
        print(":::::: Std", std)
    return count_peak


def check_peaks_2(median_data, list_data, debug=False):
    """
    Check if it is a legit 2 stage data pattern
    1. Find the middle point that separate high stage and low stage
    2. Check if there are extremely high peaks in the low stage
    3. Count peak numbers in the high stage
    """
    np_array = np.array(median_data)
    
    reshape_array = np_array.reshape(-1,1)
    km2 = KMeans(2)
    km2.fit(reshape_array)
    list_label_2 = km2.labels_

    unique, counts = np.unique(list_label_2, return_counts=True)
    dict_count = dict(zip(unique, counts))
    list_avg = [np.mean(reshape_array[np.where(list_label_2 == 0)]), np.mean(reshape_array[np.where(list_label_2 == 1)])]
    order = np.argsort(list_avg)
    list_avg_count_2 = []
    for idy in order:
        list_avg_count_2.append((int(list_avg[idy]), dict_count[idy]))
    
    count_high = list_avg_count_2[1][1]
    count_low  = list_avg_count_2[0][1]
    score = count_low
    array_score = np.zeros(len(list_label_2))
    for idx, label in enumerate(list_label_2):
        array_score[idx] = score
        if label == order[0]: ## low
            score -= 1
        else:
            score += 1
    max_idx = np.argmax(array_score)
    if debug:
        print("::::Check Stage Separation in Check_Peaks_2:")
        print("::::Max Score Idx:", max_idx)
        print("::::Score:", array_score[max_idx], array_score[max_idx]/len(list_label_2))
    
    if array_score[max_idx]/len(list_label_2) < 0.97: # drop in the middle cases
        if debug:
            print("::::Drop in the middle")
        return False
    
    np_array = np.array(list_data)
    # check if there are peak in the low part
    avg = np.mean(np_array[max_idx+50:])
    std = np.std(np_array[max_idx+50:])
    # low stage constrain
    cut_off = avg + 5*std
    avg = np.mean(np_array[:max_idx])
    std = np.std(np_array[:max_idx])
    # incorporate high stage constrain
    cut_off = max(cut_off, avg+std)
    for idx in range(max_idx+50, len(np_array)):
        if np_array[idx] > cut_off:
            if debug:
                print("::::Peaks after die out.")
                print("::::Avg, Std, Cutoff", avg, std, cut_off)
            return False

    # check how many peaks in the high part
    np_array = np_array[:max_idx]
    filtered_array = sorted(np_array)[:int(len(np_array)*0.9)]
    avg = np.mean(np_array)
    std = np.std(np_array)
    count_peak = 0

    idx = 0
    flag_peak = False
    cut_off = min(avg + 4*std, 2*avg)
    while idx < len(np_array):
        if flag_peak:
            if np_array[idx] > cut_off: # peak
                pass
            else:
                flag_peak = False
                idx += 10
        else:
            if np_array[idx] > cut_off:
                flag_peak = True
                count_peak += 1
        idx += 1
    return count_peak


def sample_for_baseline(matrix_array):
    target_array = matrix_array[:,:20]
    mean = np.mean(target_array)
    std  = np.std( target_array)

    target_list = np.reshape(target_array, (1,-1))[0]
    percentile_5 = sorted(target_list)[int(len(target_list)/20)]
    return max(mean-2*std, percentile_5)


def coherent_3_stages(list_avg_count_3, list_label_3, order, debug=False):
    """
    This function check if the 3rd class (The class with highest average value) are peaks.
    It check if the 3rd class points are clustered together in temporal dimension,
    if 0.9 of the dots are clustered together, it is a real stage, otherwise they are likely
    scattered like the peaks.
    """
    high_label = order[2]

    array_score = np.zeros(len(list_label_3))
    array_begin = np.zeros(len(list_label_3))
    score = 0       # score of the dynamic programming cell
    begin_idx = 0   # traceback information (begin (left bound) of the cluster)
    # A 1D Dynamic Programming algorithm to find the longest cluster
    for idx, label in enumerate(list_label_3):
        if label == high_label:
            score += 1
        else:
            score -= 1
        
        if score <= 0: # initial as 0 if the score becomes negative
            score = 0
            begin_idx = idx
        array_score[idx] = score
        array_begin[idx] = begin_idx
    max_idx = np.argmax(array_score) # the end (right bound) of the cluster is the same is the max score idx
    max_begin = array_begin[max_idx]
    
    count_high    = list_avg_count_3[2][1]
    cluster_count = ((max_idx-max_begin) + array_score[max_idx])/2 # number of high labels in the range
    if debug:
        print("Check for coherent of the 3 stages")
        print("Slice:", max_begin, max_idx, cluster_count/count_high)
    if cluster_count/count_high > 0.9:
        return True
    else:
        if debug:
            print("\t\t\t3rd stage classification error!")
        return False


def iteration_check(matrix_data, debug=False):
    """
    The main checking program, check if the patterns in the matrix are legit.
    1. Use K-means method to fit 2 categories and 3 categories with respect to only intensity dimensions of the pattern
    2. Determine the stages of the pattern is 1, 2, or 3 ups
    3. Check if they are legit 1 stages or 2 stages (e.g. how many peaks they have)
    """
    baseline = sample_for_baseline(np.array(matrix_data))
    
    set_3_up = set()
    set_2_stages = set()
    set_1_stages = set()
    set_1_positive = set()
    set_2_positive = set()
    for idx, list_data in enumerate(np.array(matrix_data)):
        filter_array = filter_topN(list_data)
        list_median_array = median_filter(filter_array, 50)
        reshape_data = np.array(list_median_array).reshape(-1,1)
        # K-means fitting 2 stages
        km2 = KMeans(2)
        km2.fit(reshape_data)
        list_label_2 = km2.labels_
        # K-means fitting 3 stages
        km3 = KMeans(3)
        km3.fit(reshape_data)
        list_label_3 = km3.labels_
        print("=============== Idx:", idx+1, "===============")
        
        unique, counts = np.unique(list_label_3, return_counts=True)
        dict_count = dict(zip(unique, counts))
        list_avg = [np.mean(reshape_data[np.where(list_label_3 == 0)]), np.mean(reshape_data[np.where(list_label_3 == 1)]), \
                np.mean(reshape_data[np.where(list_label_3 == 2)])]
        order_3 = np.argsort(list_avg)
        list_avg_count_3 = []
        for idy in order_3:
            list_avg_count_3.append((int(list_avg[idy]), dict_count[idy]))
            if debug:
                print(idy, int(list_avg[idy]), dict_count[idy])
    
        if debug:
            print("----------------")
        unique, counts = np.unique(list_label_2, return_counts=True)
        dict_count = dict(zip(unique, counts))
        list_avg = [np.mean(reshape_data[np.where(list_label_2 == 0)]), np.mean(reshape_data[np.where(list_label_2 == 1)])]
        order = np.argsort(list_avg)
        list_avg_count_2 = []
        for idy in order:
            list_avg_count_2.append((int(list_avg[idy]), dict_count[idy]))
            if debug:
                print(int(list_avg[idy]), dict_count[idy])
        
        if check_categories(list_avg_count_3) and \
                coherent_3_stages(list_avg_count_3, list_label_3, order_3):
            set_3_up.add(idx+1)
        else:
            if check_categories(list_avg_count_2):
                set_2_stages.add(idx+1)
                peak_num = check_peaks_2(median_filter(list_data,50), list_data, debug)
                if peak_num > 0:
                    set_2_positive.add(idx+1)
            else:
                set_1_stages.add(idx+1)
                if np.mean(list_data) < baseline: # too low
                    continue
                peak_num = check_peaks_1(median_filter(list_data,10), debug)
                if peak_num > 1:
                    set_1_positive.add(idx+1)

    return set_1_positive, set_2_positive




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-pt', '--pattern_csv_file', help='donor pattern csv file')
    parser.add_argument('-br', '--bleach_report_file', help='bleach pattern report, optional')
    parser.add_argument('-nbr', '--no_bleach_report_file', help='no bleach pattern report, optional')
    args = parser.parse_args()

    pattern_csv_file      = args.pattern_csv_file
    bleach_report_file    = args.bleach_report_file
    no_bleach_report_file = args.no_bleach_report_file

    matrix_data = read_pattern(pattern_csv_file)
    set_1_positive, set_2_positive = iteration_check(matrix_data)

    # report results
    print("==================== Results =========================")
    if no_bleach_report_file:
        set_1_annotation = read_annotation(no_bleach_report_file)
        print("No Bleach Overlap:", len(set_1_positive.intersection(set_1_annotation)))
        print("False Negative (Missed):", sorted(set_1_annotation - set_1_positive))
        print("False Positive (Addit.):", sorted(set_1_positive - set_1_annotation))
    else:
        print("No Bleach Detect:", sorted(set_1_positive))
    if bleach_report_file:
        set_2_annotation = read_annotation(bleach_report_file)
        print("Bleach Overlap:", len(set_2_positive.intersection(set_2_annotation)))
        print("False Negative (Missed):", sorted(set_2_annotation - set_2_positive))
        print("False Positive (Addit.):", sorted(set_2_positive - set_2_annotation))
    else:
        print("Bleach Detect:", sorted(set_2_positive))
