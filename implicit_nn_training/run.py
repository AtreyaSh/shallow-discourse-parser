import os

# run single instance
os.system("python3 test_single.py \
          ./data/en-01-12-16-train/en.train/parses.json \
          ./data/en-01-12-16-dev/en.dev/parses.json \
          ./data/en-01-12-16-train/en.train/relations.json \
          ./data/en-01-12-16-dev/en.dev/relations.json ./data/GoogleNews-vectors-negative300.bin")

# run best model
# os.system("python3 re-classify-implicit_entrel_senses_nn.py \
          #input_to_be_reclassified_for_implicit_senses.json \
          #./data/en-01-12-16-dev/en.dev/parses.json")