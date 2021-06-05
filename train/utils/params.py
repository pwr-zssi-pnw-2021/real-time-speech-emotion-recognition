import yaml


def get_params() -> dict:
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    return params
