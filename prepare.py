from pathlib import Path

import yaml


def save_params(params: dict) -> None:
    with open('params.yaml', 'w') as f:
        yaml.safe_dump(params, f)


def get_extraction_list(params: dict) -> list[dict]:
    datasets = params['data']['datasets']
    features = params['data']['features']

    index_dir = Path(params['data']['index_dir'])
    features_dir = Path(params['data']['data_dir'])

    extraction = []
    for ds_name, ds_prop in datasets.items():
        for f_name in features:
            index_file = index_dir / f'{ds_name}_{f_name}'
            f_dir = features_dir / f'{ds_name}_{f_name}'

            extraction_item = {
                'dataset': ds_name,
                'name': f_name,
                'out_path': str(f_dir),
                'in_path': ds_prop['wav'],
                'index': str(index_file),
            }
            extraction.append(extraction_item)

    return extraction


if __name__ == '__main__':
    with open('params-base.yaml') as f:
        params = yaml.safe_load(f)

    extraction = get_extraction_list(params)
    params['data']['extraction'] = extraction

    save_params(params)
