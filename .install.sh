#!/bin/bash 
python setup.py bdist_wheel
pip install dist/*.whl
pytest pytest/