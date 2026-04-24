import os

ROOT_PATH = os.getcwd()
TMP_PATH = os.path.join(ROOT_PATH, "tmp")
DATASET_PATH = os.path.join(ROOT_PATH, "dataset")
WEIGHTS_PATH = os.path.join(ROOT_PATH, "weights")
EXPORTS_PATH = os.path.join(ROOT_PATH, "exports")
PRETRAINED_PATH = os.path.join(ROOT_PATH, "pretrained")

def get_tmp_path(*args):
    return os.path.join(TMP_PATH, *map(str, args))

def get_dataset_path(*args):
    return os.path.join(DATASET_PATH, *map(str, args))

def get_weight_file_path(*args):
    return os.path.join(WEIGHTS_PATH, *map(str, args))

def get_export_file_path(*args):
    return os.path.join(EXPORTS_PATH, *map(str, args))

def get_pretrained_file_path(*args):
    return os.path.join(PRETRAINED_PATH, *map(str, args))
