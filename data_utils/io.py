import pickle


def load(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def save(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)
        print('saved')
