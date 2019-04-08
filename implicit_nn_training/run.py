#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# run single instance, old embedding m_0
os.system("python3 train.py \
          single m_0 \
          ./data/en-01-12-16-train/en.train/parses.json \
          ./data/en-01-12-16-dev/en.dev/parses.json \
          ./data/en-01-12-16-train/en.train/relations.json \
          ./data/en-01-12-16-dev/en.dev/relations.json \
          ./data/GoogleNews-vectors-negative300.bin")

# run gridding instance, old embedding m_0
os.system("python3 train.py \
          grid m_0 \
          ./data/en-01-12-16-train/en.train/parses.json \
          ./data/en-01-12-16-dev/en.dev/parses.json \
          ./data/en-01-12-16-train/en.train/relations.json \
          ./data/en-01-12-16-dev/en.dev/relations.json \
          ./data/GoogleNews-vectors-negative300.bin")

# run combination instance, old embedding m_0
os.system("python3 train.py \
          combination m_0 \
          ./data/en-01-12-16-train/en.train/parses.json \
          ./data/en-01-12-16-dev/en.dev/parses.json \
          ./data/en-01-12-16-train/en.train/relations.json \
          ./data/en-01-12-16-dev/en.dev/relations.json \
          ./data/GoogleNews-vectors-negative300.bin")