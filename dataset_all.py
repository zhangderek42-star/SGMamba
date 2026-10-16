import os
from pathlib import Path


DATASET_ROOT = Path(__file__).resolve().parent / "dataset"

CAVE_train_path = str(DATASET_ROOT / "cave" / "train") + os.sep
CAVE_eval_path = str(DATASET_ROOT / "cave" / "eval") + os.sep
CAVE_test_path = str(DATASET_ROOT / "cave" / "test") + os.sep
CAVE_val_path = CAVE_eval_path

Harvard_train_path = str(DATASET_ROOT / "harvard" / "train") + os.sep
Harvard_eval_path = str(DATASET_ROOT / "harvard" / "eval") + os.sep
Harvard_test_path = str(DATASET_ROOT / "harvard" / "test") + os.sep


def get_dataset_path(name="CAVE", mode="train"):
    """Return a project-local split path; legacy 'val' is an alias for 'eval'."""
    if name == "CAVE":
        paths = {
            "train": CAVE_train_path,
            "eval": CAVE_eval_path,
            "test": CAVE_test_path,
        }
    elif name == "Harvard":
        paths = {
            "train": Harvard_train_path,
            "eval": Harvard_eval_path,
            "test": Harvard_test_path,
        }
    else:
        raise NotImplementedError(f"Dataset {name} is not implemented.")

    if mode == "val":
        mode = "eval"
    if mode not in paths:
        raise ValueError(f"Mode {mode} is not recognized.")
    return paths[mode]
