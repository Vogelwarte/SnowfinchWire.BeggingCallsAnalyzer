import pandas as pd
from more_itertools import chunked
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


def plot_feeding_count_daily(df, output_dir):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.groupby([df['datetime'].dt.date, 'brood_id']).mean().drop(columns=['datetime']).reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(['brood_id', 'datetime'])
    df = df.unstack(level=[0]).resample('1d').first().stack(level=[1], dropna=False).swaplevel(1, 0).sort_index()
    lst = [(b, df) for b, df in df.groupby(level=0)]
    lst_sorted = sorted(lst, key=lambda d: d[1].first_valid_index()[1])

    for batch in chunked(lst_sorted, 5):
        fig, ax = plt.subplots(figsize=(15, 5))
        broods = []
        for brood, new_df in batch:
            broods.append(brood)
            ax.plot(new_df.index.get_level_values(1), new_df['feeding_count'], label=brood, marker='o')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.legend()
        plt.xticks(rotation=30)
        plt.ylabel("Mean number of feedings")
        plt.xlabel("Date")
        plt.grid(True)
        plt.savefig(Path(output_dir) / f"{'-'.join(broods)}.png")


def plot_feeding_count_hourly(df, output_dir):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(['brood_id', 'datetime'])
    df = df.unstack(level=[0]).resample('1h').first().stack(level=[1], dropna=False).swaplevel(1, 0).sort_index().reset_index()
    result = df.groupby([df['datetime'].dt.hour, 'brood_id']).mean().swaplevel(1, 0).sort_index()
    for batch in chunked(result.groupby(level=0), 5):
        fig, ax = plt.subplots(figsize=(15, 5))
        broods = []
        for brood, new_df in batch:
            broods.append(brood)
            ax.plot(pd.to_datetime(new_df.index.get_level_values(1), format='%H').strftime('%H:%M'), new_df['feeding_count'], label=brood, marker='o')
        plt.ylim(bottom=0)
        plt.xlim([0, 23])
        plt.legend()
        plt.xticks(rotation=30)
        plt.ylabel("Mean number of feedings")
        plt.xlabel("Time of day")
        plt.grid(True)
        plt.savefig(Path(output_dir) / f"{'-'.join(broods)}.png")


def plot_feeding_duration_daily(df, output_dir):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df['total_feeding_time'] = df['duration'] * df['feeding_count']
    groups = df.groupby([df['datetime'].dt.date, 'brood_id'])
    result = groups['total_feeding_time'].sum() / groups['feeding_count'].sum()
    df = result.to_frame('mean_feeding_duration').reset_index()
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(['brood_id', 'datetime'])
    df = df.unstack(level=[0]).resample('1d').first().stack(level=[1], dropna=False).swaplevel(1, 0).sort_index()
    lst = [(b, df) for b, df in df.groupby(level=0)]
    lst_sorted = sorted(lst, key=lambda d: d[1].first_valid_index()[1])

    for batch in chunked(lst_sorted, 5):
        fig, ax = plt.subplots(figsize=(15, 5))
        broods = []
        for brood, new_df in batch:
            broods.append(brood)
            ax.plot(new_df.index.get_level_values(1), new_df['mean_feeding_duration'], label=brood, marker='o')
        ax.set_ylim(bottom=0)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.legend()
        plt.xticks(rotation=30)
        plt.ylabel("Mean feeding duration [s]")
        plt.xlabel("Date")
        plt.grid(True)
        plt.savefig(Path(output_dir) / f"{'-'.join(broods)}.png")


def plot_feeding_duration_hourly(df, output_dir):
    df['datetime'] = pd.to_datetime(df['datetime'])
    df = df.set_index(['brood_id', 'datetime'])
    df['total_feeding_time'] = df['duration'] * df['feeding_count']
    df = df.unstack(level=[0]).resample('1h').first().stack(level=[1], dropna=False).swaplevel(1, 0).sort_index().reset_index()
    groups = df.groupby([df['datetime'].dt.hour, 'brood_id'])
    result = groups['total_feeding_time'].sum() / groups['feeding_count'].sum()
    df = result.to_frame('mean_feeding_duration').swaplevel(0, 1)

    for batch in chunked(df.groupby(level=0), 5):
        fig, ax = plt.subplots(figsize=(15, 5))
        broods = []
        for brood, new_df in batch:
            broods.append(brood)
            ax.plot(pd.to_datetime(new_df.index.get_level_values(1), format='%H').strftime('%H:%M'), new_df['mean_feeding_duration'], label=brood, marker='o')
        plt.ylim(bottom=0)
        plt.xlim([0, 23])
        plt.legend()
        plt.xticks(rotation=30)
        plt.ylabel("Mean feeding duration [s]")
        plt.xlabel("Time of day")
        plt.grid(True)
        plt.savefig(Path(output_dir) / f"{'-'.join(broods)}.png")
