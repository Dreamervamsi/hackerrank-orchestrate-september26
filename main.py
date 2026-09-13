#!/usr/bin/env python3
"""
Buy or Wait? Financial Decision Agent
Top-level entry point to execute the full production batch pipeline.
"""
import sys

# Add code folder to python path so local imports work seamlessly
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from code.main import main

if __name__ == "__main__":
    main()
