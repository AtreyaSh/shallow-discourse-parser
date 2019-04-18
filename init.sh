#!/bin/bash
set -e

# move pre-commit hook into local .git folder for activation
cp ./hooks/pre-commit-pipreqs.sample ./.git/hooks/pre-commit

# download google vector file and move to data directory
# source for gvec download script:
# https://gist.github.com/yanaiela/cfef50380de8a5bfc8c272bb0c91d6e1
cd hooks && ./word2vec-download300model.sh ../tmp && cd ../tmp && \
gunzip -k GoogleNews-vectors-negative300.bin.gz && \
mv GoogleNews-vectors-negative300.bin ../implicit_nn_training/data/
