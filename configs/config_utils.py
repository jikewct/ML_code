from copy import deepcopy

import ml_collections
import ml_collections.config_dict

# new config


def n(config, *keys):
    current_config = config
    for key in keys:
        if not hasattr(current_config, key):
            current_config[key] = ml_collections.ConfigDict()
        current_config = current_config[key]
    return current_config


from .default_configs import DEFAULT_CONFIGS

# print(DEFAULT_CONFIGS)


# copy default config if not has key else return config[key]
def c(config, *keys):
    current_config = config
    current_defualt_config = DEFAULT_CONFIGS
    for key in keys:
        if not hasattr(current_defualt_config, key):
            raise KeyError(f"not found default config, key:{key}")
        current_defualt_config = current_defualt_config[key]
        # if key == "condition":
        #     #print("===\n", current_config)
        if not hasattr(current_config, key):
            if not isinstance(current_defualt_config, ml_collections.ConfigDict):
                current_config[key] = current_defualt_config
                break
            else:
                current_config[key] = ml_collections.ConfigDict()
                current_config = current_config[key]
                for item_key, item_val in current_defualt_config.items():
                    if not isinstance(item_val, ml_collections.ConfigDict):
                        current_config[item_key] = item_val
        else:
            current_config = current_config[key]
    return current_config
