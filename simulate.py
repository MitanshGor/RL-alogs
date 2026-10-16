#!/usr/bin/env python3
"""
Root entrypoint for running the Multi-Armed Bandit terminal simulation.
Delegates directly to Multiarmbandit/simulate.py.
"""
import os
import sys

# Ensure Multiarmbandit module directory is on sys.path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BANDIT_DIR = os.path.join(SCRIPT_DIR, "Multiarmbandit")
sys.path.insert(0, BANDIT_DIR)

from simulate import main

if __name__ == "__main__":
    main()
