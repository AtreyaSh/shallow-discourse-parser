#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

# run single instance, old embedding m_0
os.system("python3 train.py \
          --name m_0")

# run gridding instance, old embedding m_0
#os.system("python3 train.py \
#          --mode grid --name m_0")

# run combination instance, old embedding m_0
#os.system("python3 train.py \
#          --mode combination --name m_0")