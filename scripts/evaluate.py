#!/usr/bin/env python3
"""Thin wrapper so the CLI runs without ``pip install -e .``.

For a console_scripts install, the entry point is ``roofseg-evaluate``.
"""

from __future__ import annotations

import os
import sys

if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from roofseg.cli.evaluate import main

    raise SystemExit(main())
