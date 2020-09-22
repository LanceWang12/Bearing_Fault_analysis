import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def transform(arr):
    # --------------------------
    # | Parameter:
    # | arr: Please input an 1darray, list, or series.
    # --------------------------
    
    # --------------------------
    # | Return:
    # | out: mafnitude
    # --------------------------
    
    arr = np.asarray(arr, dtype = 'float64')
    
    # 左右對稱 -> 取一半的資料
    FFT = np.fft.fft(arr, norm='ortho')[:len(arr)//2]
    
    # get magnitude
    mag = np.abs(FFT)
    
    return mag

def outlier_detect_statistic(data, lower = 10, upper = 90):
    mask = []
    different = []

    if len(data.shape) != 2:
        datas = data.reshape(-1,128)
    else:
        datas = data
    
    for tmp in datas:
        different.append(abs(np.max(tmp)-np.min(tmp)))
    different = np.array(different)

    L, U = np.percentile(different, [lower, upper])
    mask = (different >= L) & (different <= U)

    return mask

def outlier_detect_Isolation_Forest(data, percentage = 0.1):
    IF = IsolationForest(n_estimators=30, max_samples='auto', contamination=percentage, max_features=1.0, bootstrap=True, n_jobs=None, random_state=12, verbose=0, warm_start=False)
    IF.fit(data)
    outlier = IF.predict(data)
    mask = np.where(outlier == 1)[0]

    return mask

def read_file(file_lst, name_lst, n = 10, wnd = 256, outlier_detect_mode = 1, lower = 5, upper = 95, percentage = 0.1):
    # | file_lst: [檔名1, 檔名2, ...]
    # | name_lst: key 的名字
    # | n: 每個類別有幾筆 4096 的資料
    # | wnd: window size
    
    data = dict()
    for i, name in enumerate(file_lst):
        # 取得資料
        data[name_lst[i]] = np.load(name)
    
    mag_arr = []
    time_arr = []
    for name in name_lst:
        for i in range(n):
            for j in range(0, data[name].shape[1], wnd):
                time = data[name][i][j: j + wnd]
                time_arr.append(time)
                
    time_arr = np.array(time_arr)    
    
    # generate label: [0, 0, 0, ..., 5, 5, 5, ..., 6, 6, 6]
    label = np.zeros((4096//wnd * len(file_lst) * n, ))
    one_part = label.shape[0]//len(file_lst)
    for i in range(len(file_lst), ):
        label[i*one_part: i*one_part+one_part] = i
    
    if (outlier_detect_mode == 1) or (outlier_detect_mode == 3):
        mask = outlier_detect_statistic(time_arr, lower = lower, upper = upper)
        time_arr = time_arr[mask]
        label = label[mask]

    mag_arr = np.zeros((time_arr.shape[0], time_arr.shape[1]//2))
    for i in range(time_arr.shape[0]):
        mag_arr[i] = transform(time_arr[i])
    
    if (outlier_detect_mode == 2) or (outlier_detect_mode == 3):
        mask = outlier_detect_Isolation_Forest(mag_arr, percentage = percentage)
        time_arr = time_arr[mask]
        mag_arr = mag_arr[mask]
        label = label[mask]

    # 檢查形狀
    print('X.shape = {}'.format(mag_arr.shape))
    print('Y.shape = {}'.format(label.shape))
    
    return mag_arr, time_arr, label

