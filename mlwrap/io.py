from pickle import dump, dumps, load, loads


def save_pkl(obj, path):
    dump(obj, open(path, "wb"))


def save_pkl_bytes(obj) -> bytes:
    return dumps(obj)


def load_pkl(path_or_bytes):
    if isinstance(path_or_bytes, bytes):
        return loads(path_or_bytes)
    return load(open(path_or_bytes, "rb"))
