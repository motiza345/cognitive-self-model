from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional


def get_git_commit_hash(
    repository_dir: str | Path = ".",
) -> Optional[str]:
    """
    Return the current Git commit hash.

    Returns None if the code runs outside a valid Git repository.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_dir),
                "rev-parse",
                "HEAD",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip()

    except Exception:
        return None


def get_git_branch_name(
    repository_dir: str | Path = ".",
) -> Optional[str]:
    """
    Return the current Git branch name.

    Returns None in detached-HEAD mode or outside a Git repository.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_dir),
                "branch",
                "--show-current",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        branch_name = result.stdout.strip()

        return branch_name if branch_name else None

    except Exception:
        return None


def is_git_working_tree_clean(
    repository_dir: str | Path = ".",
) -> Optional[bool]:
    """
    Return True if the repository has no uncommitted changes.

    Returns None if Git status cannot be determined.
    """
    try:
        result = subprocess.run(
            [
                "git",
                "-C",
                str(repository_dir),
                "status",
                "--porcelain",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        return result.stdout.strip() == ""

    except Exception:
        return None
