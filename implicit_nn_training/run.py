import os

os.system("python2 test_single.py \
          ./data/en-01-12-16-train/en.train/parses.json \
          ./data/en-01-12-16-dev/en.dev/parses.json \
          ./data/en-01-12-16-train/en.train/relations.json \
          ./data/en-01-12-16-dev/en.dev/relations.json ./data/GoogleNews-vectors-negative300.bin")