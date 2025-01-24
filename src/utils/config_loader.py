import json

def load_config_file(config_file):
    with open(config_file, "r") as f:
        data = json.load(f)
        print("JSON is valid:", data)
    return data
