import argparse
import subprocess
from pathlib import Path
from datetime import datetime

import pandas as pd

if __name__ == '__main__':
	arg_parser = argparse.ArgumentParser()
	arg_parser.add_argument('data_path', type = str)
	arg_parser.add_argument('-m', '--model-path', type = str)
	arg_parser.add_argument('-r', '--rec-data', type = str)
	args = arg_parser.parse_args()

	start_time = datetime.now()

	rec_df = pd.read_csv(args.rec_data)
	brood_ids = list(rec_df['brood_id'].unique())
	data_path = Path(args.data_path)
	brood_dirs = [path for path in data_path.rglob('*') if path.stem in brood_ids]

	print(f'Starting processes for broods: {brood_dirs}')

	processes = []
	for brood_dir in brood_dirs:
		cmd = [
			'./venv/bin/python', '-m', 'beggingcallsanalyzer', 'predict',
			args.model_path, brood_dir.as_posix(), '--extension', 'WAV'
		]
		processes.append(subprocess.Popen(cmd))

	print(f'Started {len(processes)}')

	for process in processes:
		exit_code = process.wait()
		if exit_code:
			print(f'Process {process} exited with code {exit_code}')

	task_duration = datetime.now() - start_time
	print(f'Task completed in {task_duration}')
