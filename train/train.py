import argparse

from utils.utils import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-f', '--features', required=True)

    args = parser.parse_args()
    model = args.model
    features = args.features

    train_model(model, features)
