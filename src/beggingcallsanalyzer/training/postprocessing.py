import numpy as np
import pandas as pd
import math


def post_process(y_pred: np.ndarray, win_length, hop_length, merge_window, cut_length) -> np.ndarray:
    pred_len = len(y_pred)

    # remove very short observations - most likely artifacts
    i = 0
    while i < len(y_pred):
        if y_pred[i] == 1:
            right_idx = i + 1
            while right_idx < len(y_pred) and y_pred[right_idx] == 1:
                right_idx = right_idx + 1
            difference = right_idx - i
            if difference * hop_length <= cut_length:
                for j in range(i, right_idx):
                    y_pred[j] = 0
            i = right_idx
        else:
            i = i + 1


    # merge observations that are very close to each other
    check_range = math.floor(merge_window/hop_length)
    
    for i in range(0, pred_len):
        if y_pred[i] == 1:
            for j in range(1, check_range + 1):
                left_idx = max(i - j, 0)
                if y_pred[left_idx] == 1:
                    for k in range(left_idx, i):
                        y_pred[k] = 1

    return y_pred

def to_audacity_labels(y_pred, duration, window, step, result_label="feeding"):
    df_export = pd.DataFrame()
    df_export['y'] = y_pred
    time = np.arange(0, duration, step)
    time_interval = pd.arrays.IntervalArray.from_arrays(time, time + window)
    df_export['time'] = time_interval[:len(df_export)]
    df_export = df_export[df_export['y'] == 1].reset_index(drop=True)
    interval_index = pd.IntervalIndex(df_export['time'], closed='both')

    df_export['start'] = interval_index.left
    df_export['end'] = interval_index.right
    df_export['group'] = (df_export["start"].round(1) > df_export['end'].shift().round(1)).cumsum()
    result = df_export.groupby("group").agg({'start': 'min', 'end': 'max'})
    result['class'] = result_label
    return result