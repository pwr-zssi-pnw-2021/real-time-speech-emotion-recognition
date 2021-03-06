import argparse

from utils.utils import train_model

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', required=True)
    parser.add_argument('-f', '--features', required=True)
    parser.add_argument('-w', '--window', type=int, required=True)
    parser.add_argument('-o', '--output', required=True)

    args = parser.parse_args()
    model = args.model
    features = args.features
    window_size = args.window
    out_file = args.output

    train_model(model, features, window_size, out_file)
