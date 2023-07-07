import pickle
import os


def load(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def save(file, data):
    directory = os.path.dirname(file)
    os.makedirs(directory, exist_ok=True)

    with open(file, "wb") as f:
        pickle.dump(data, f)
        print('saved')
