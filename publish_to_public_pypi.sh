#!/usr/bin/env bash
set -e
echo "Building and uploading BioMorphX v1.0.0 to public PyPI (https://pypi.org)..."
python3 -m pip install --upgrade build twine
python3 -m build
python3 -m twine check dist/*
python3 -m twine upload dist/*
echo "Published! Anyone can now run: pip install biomorphx"
