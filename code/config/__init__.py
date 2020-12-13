import yaml
from pathlib import Path

current_directory = Path(__file__).parents[0].resolve()

config_path = current_directory / Path('config.yml')
logger_path = current_directory / Path('logger.yml')
json_request_schema_path = current_directory / Path('json_request_schema.yml')

config = yaml.safe_load(open(str(config_path), 'r'))
logger_config = yaml.safe_load(open(str(logger_path), 'r'))
json_request_schema = yaml.safe_load(open(str(json_request_schema_path), 'r'))