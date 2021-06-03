import argparse

from utils import (
    CLASS_EXTRACTOR_LOOKUP,
    DATASETS,
    FEATURE_EXTRACTOR_LOOKUP,
    FEATURES,
    extract,
    generate_extractor,
    mp_wrapper,
)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--features', required=True, choices=FEATURES)
    parser.add_argument('-d', '--dataset', required=True, choices=DATASETS)
    parser.add_argument('-i', '--in_path', required=True)
    parser.add_argument('-o', '--out_path', required=True)
    parser.add_argument('-x', '--index', required=True)

    args = parser.parse_args()

    dataset = args.dataset
    features = args.features
    data_path = args.in_path
    out_path = args.out_path
    index_path = args.index

    class_extractor = CLASS_EXTRACTOR_LOOKUP[dataset]
    feature_extractor = FEATURE_EXTRACTOR_LOOKUP[features]

    extractor = generate_extractor(class_extractor, feature_extractor)
    mp_extractor = mp_wrapper(extractor)

    extract(data_path, out_path, index_path, mp_extractor)
