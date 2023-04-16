import sys
import yaml
from yaml.loader import SafeLoader

config_file_path = sys.argv[1]
raw_config_dict = {}
with open(config_file_path, 'r') as config_file:
    raw_config_dict = yaml.load(config_file, SafeLoader)
sys.path.append(raw_config_dict['root_directory'])

from src.TestHelpers.TestConfiguration import TestConfiguration
from src.TestHelpers.TestRunner import run

test_config = TestConfiguration.from_file(config_file_path)

run(test_config)