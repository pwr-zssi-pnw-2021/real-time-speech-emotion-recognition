import numpy as np
import yaml


def get_params() -> dict:
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    return params


RND_STATE = np.random.RandomState(get_params()['train']['seed'])
