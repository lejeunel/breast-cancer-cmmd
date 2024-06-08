import yaml
from pathlib import Path


def save_to_yaml(path, cfg_dict):

    cfg_dict = {k: str(v) if isinstance(v, Path) else v for k, v in cfg_dict.items()}
    yaml_data = yaml.dump(cfg_dict, default_flow_style=False)
    with open(path, "w") as f:
        f.write(yaml_data)
