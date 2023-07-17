# import pickle
# from pathlib import Path
#
#
# def load(file):
#     with open(file, "rb") as f:
#         return pickle.load(f)
#
#
# def save(file, data):
#     p = Path(file)
#     p.parent.mkdir(parents=True, exist_ok=True)
#     print(f"Saving file to {p.resolve()}")
#
#     with open(file, "wb") as f:
#         pickle.dump(data, f)
#         print('saved')
