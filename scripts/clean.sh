#!/bin/bash

# clean environemnt from objects and cache files
find . -name "*.pyc" -type f -exec rm -r {} +
find . -name "__pycache__" -type d -exec rm -rf {} +
find . -name ".ipynb*" -type d -exec rm -rf {} +