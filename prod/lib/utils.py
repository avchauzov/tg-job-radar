import logging
import os
from pathlib import Path

from dotenv import load_dotenv

from prod.utils.functions_common import setup_logging


file_name = os.path.splitext(os.path.basename(__file__))[0]
setup_logging(file_name)


def get_correct_path(file_name):
	load_dotenv()
	
	base_dir = os.getenv('CONFIG_BASE_DIR', None)
	local_dir = os.getenv('CONFIG_BASE_LOCAL_DIR', None)
	
	logging.info(f'Starting to resolve file path for: {file_name}')
	logging.debug(f'Base directory from env: {base_dir}')
	logging.debug(f'Local directory from env: {local_dir}')
	
	if base_dir:
		base_path = Path(base_dir).resolve()
		logging.info(f'Resolved base directory: {base_path}')
	
	else:
		base_path = None
		logging.warning('Base directory is not set in the environment.')
	
	if local_dir:
		local_path = Path(local_dir).resolve()
		logging.info(f'Resolved local directory: {local_path}')
	
	else:
		local_path = None
		logging.warning('Local directory is not set in the environment.')
	
	if base_path and base_path.exists():
		file_path = base_path / file_name
		logging.info(f'Using base directory for file path: {file_path}')
	
	elif local_path and local_path.exists():
		file_path = local_path / file_name
		logging.info(f'Using local directory for file path: {file_path}')
	
	else:
		script_dir = Path(__file__).resolve().parent
		file_path = script_dir / file_name
		file_path = file_path.resolve()
		logging.info(f'Using script directory for file path: {file_path}')
	
	if file_path.exists():
		logging.info(f'File found at path: {file_path}')
		return file_path
	
	logging.error(f'File not found at {file_path}')
	raise FileNotFoundError(f'File not found at {file_path}')
