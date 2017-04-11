#!/usr/bin/env bash
python3 sample.py > /dev/null &
nosetests --with-coverage
