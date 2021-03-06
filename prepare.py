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
            index_file = index_dir / f'{ds_name}_{f_name}.index'
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


def get_training_list(params: dict) -> list[dict]:
    models = params['train']['models']
    features = params['data']['features']

    results_dir = Path(params['train']['results_dir'])

    window_lookup = {
        'mfcc': 50,
        'lpcc': 120,
        'sc': 50,
    }  # 1 percentile + 1 for attention heads

    training = []
    for m in models:
        for f in features:
            out_file = results_dir / f'{m}_{f}.pkl'
            training_item = {
                'model': m,
                'features': f,
                'window_size': window_lookup[f],
                'out_file': str(out_file),
            }
            training.append(training_item)

    return training


if __name__ == '__main__':
    with open('params-base.yaml') as f:
        params = yaml.safe_load(f)

    extraction = get_extraction_list(params)
    params['data']['extraction'] = extraction

    training = get_training_list(params)
    params['train']['training'] = training

    save_params(params)
