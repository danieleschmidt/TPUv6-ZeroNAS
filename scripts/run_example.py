#!/usr/bin/env python3
"""Script to run TPUv6-ZeroNAS example search."""

import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from examples.basic_search import main

if __name__ == '__main__':
    main()