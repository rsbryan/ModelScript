#!/usr/bin/env python3
"""
ModelScript REPL Launcher
Launch the interactive ModelScript environment
"""
import sys
import os

# Add modelscript directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'modelscript'))

from repl import main

if __name__ == "__main__":
    main()