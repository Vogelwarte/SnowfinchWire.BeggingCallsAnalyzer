import pandas as pd
import os


def create_summary_csv(results: dict, output_path, extension):
    df = pd.concat([v['predictions'] for v in results.values()], keys=results.keys(), names=['filename', 'idx']).reset_index()
    df['filename'] = df['filename'].astype(str)
    pattern = rf'(.*)[\\/](?P<brood_id>[^\\/]+)[\\/](?:[^\\/]+)[\\/](?P<datetime>.*)\.{extension}'
    data = df['filename'].str.extract(pattern)
    df = df.join(data)
    df = pd.get_dummies(df, columns=['class'])
    df['duration'] = df["end"] - df['start']
    df['datetime'] = pd.to_datetime(df['datetime'], format='%Y%m%d_%H%M%S', errors='coerce')
    df = df.dropna(axis='index', subset=['datetime'])
    df = df.drop(columns=['index', 'start', "end", 'filename'], errors='ignore').groupby(['brood_id', 'datetime']).agg({
        "duration": "mean",
        "class_feeding": "sum"
    }).reset_index().rename(columns={"class_feeding": "feeding_count", "duration": "mean_feeding_duration"})
    df = df.drop(columns=['filename', 'idx', 0], errors='ignore')
    return df
