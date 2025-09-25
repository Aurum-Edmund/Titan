"""Compatibility wrapper for legacy imports."""
from __future__ import annotations

from src.titan.train_context import main, train, train_from_yaml

__all__ = ["main", "train", "train_from_yaml"]

if __name__ == "__main__":
    main()
