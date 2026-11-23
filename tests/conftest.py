"""Ensure PyTorch initializes before PyArrow on Windows during pytest collection.

Importing ``pyarrow`` before ``torch`` in the same Windows Python process can trigger
``OSError: [WinError 1114]`` when loading ``c10.dll``.
"""

from __future__ import annotations

import torch  # noqa: F401
