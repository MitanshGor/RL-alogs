#!/usr/bin/env python3
"""
Root entrypoint to generate all 6 Multi-Armed Bandit comparison plots.
Delegates to Multiarmbandit/plot_comparisons.py.
"""
import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDIT_DIR = os.path.join(SCRIPT_DIR, "Multiarmbandit")
sys.path.insert(0, BANDIT_DIR)

from plot_comparisons import main

if __name__ == "__main__":
    main()
