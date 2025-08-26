#!/usr/bin/env python3
"""
Simple script to compile datasets from folders containing robot data.

Usage:
    python compile_dataset.py <folder_name>

Example:
    python compile_dataset.py dataset_20250821_115559
"""

import sys
import os
from dataset_compiler import compile_dataset

def main():
    if len(sys.argv) != 2:
        print("Usage: python compile_dataset.py <folder_name>")
        print("Example: python compile_dataset.py dataset_20250821_115559")
        sys.exit(1)
    
    folder_name = sys.argv[1]
    
    if not os.path.exists(folder_name):
        print(f"Error: Folder '{folder_name}' does not exist")
        sys.exit(1)
    
    print(f"Compiling dataset from folder: {folder_name}")
    compile_dataset(folder_name)

if __name__ == "__main__":
    main()