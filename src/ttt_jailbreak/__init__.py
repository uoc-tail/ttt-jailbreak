from pathlib import Path

import git
from omegaconf import OmegaConf


def decode_path(path: str | Path) -> Path:
    """Convert input to Path object and create directory if it doesn't exist.

    Args:
        path: Input path as string or Path object
    Returns:
        Path object with created directory
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


if not OmegaConf.has_resolver("path"):
    OmegaConf.register_new_resolver("path", lambda val: decode_path(val))

try:
    PROJECT_ROOT = Path(
        git.Repo(Path.cwd(), search_parent_directories=True).working_dir
    )
except git.exc.InvalidGitRepositoryError:
    PROJECT_ROOT = Path.cwd()
