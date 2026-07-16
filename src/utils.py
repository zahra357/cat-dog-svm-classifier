from pathlib import Path


def ensure_dir(path):
    """
    Create a directory if it does not already exist.
    """
    Path(path).mkdir(parents=True, exist_ok=True)


def create_output_dirs(*dirs):
    """
    Create multiple output directories.
    """
    for directory in dirs:
        ensure_dir(directory)