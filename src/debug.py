#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

###################################
# sample runs for checking
###################################

# run single instance, old embedding m_0
os.system("python3 train.py \
          --name m_0")

# run gridding instance, old embedding m_0
os.system("python3 train.py \
          --mode grid --name m_0")

# run combination instance, old embedding m_0
os.system("python3 train.py \
          --mode combination --name m_0")

###################################
# sample variables for debugging
###################################

parses_train_filepath = "./data/en.train/parses.json"
parses_dev_filepath = "./data/en.dev/parses.json"
parses_test_filepath = "./data/en.test/parses.json"
relations_train_filepath = "./data/en.train/relations.json"
relations_dev_filepath = "./data/en.dev/relations.json"
relations_test_filepath = "./data/en.test/relations.json"
googlevecs_filepath = "./data/GoogleNews-vectors-negative300.bin"