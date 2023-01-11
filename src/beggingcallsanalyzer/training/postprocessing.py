import numpy as np
import pandas as pd


def post_process(y_pred: np.ndarray) -> np.ndarray:
    y = y_pred.copy()
    # include singular frames that are surrounded by detections
    for i in range(1, len(y_pred)-1):
        if y_pred[i-1] == 1 and y_pred[i+1] == 1:
            y[i] = 1

    # remove singular observations with no neighbors - most likely artifacts
    for i in range(1, len(y_pred)-1):
        if y_pred[i-1] == 0 and y_pred[i+1] == 0:
            y[i] = 0

    return y

def to_audacity_labels(y_pred, duration, window, step, result_label="feeding"):
    df_export = pd.DataFrame()
    df_export['y'] = y_pred
    time = np.arange(0, duration - step, step)
    df_export['time'] = pd.arrays.IntervalArray.from_arrays(time, time + window)
    df_export = df_export[df_export['y'] == 1].reset_index(drop=True)
    interval_index = pd.IntervalIndex(df_export['time'], closed='both')

    df_export['start'] = interval_index.left
    df_export['end'] = interval_index.right
    df_export['group'] = (df_export["start"].round(1) > df_export['end'].shift().round(1)).cumsum()
    result = df_export.groupby("group").agg({'start': 'min', 'end': 'max'})
    result['class'] = result_label
    return result