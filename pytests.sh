#!/usr/bin/env bash
python3 -c "import nltk; nltk.download('all')"
python3 sample.py > /dev/null &
nosetests --with-coverage
