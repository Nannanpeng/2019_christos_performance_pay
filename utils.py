# Utility Functions
# Mainly for configuration management

import yaml


def load_model_from_yaml(file_path):
    with open(file_path, 'r') as stream:
        try:
            print(yaml.safe_load(stream))
        except yaml.YAMLError as exc:
            print(exc)
