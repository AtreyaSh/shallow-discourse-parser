#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os

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
# code for debugging
###################################

parses_train_filepath = "./data/en.train/parses.json"
parses_dev_filepath = "./data/en.dev/parses.json"
parses_test_filepath = "./data/en.test/parses.json"
relations_train_filepath = "./data/en.train/relations.json"
relations_dev_filepath = "./data/en.dev/relations.json"
relations_test_filepath = "./data/en.test/relations.json"
googlevecs_filepath = "./data/GoogleNews-vectors-negative300.bin"

# 0. load all parses/relations information into memory
# 1. loop through all directories and find embedding model, load it into memory
# 2. assign relevant convert_relation function, create input and output pickles and save to file
# 3. exp.predict on the input/outputs for train/dev and create classification reports
# 4. save classification reports as another results file, or possibly edit old file in place

def recalc():
    # load necessary files
    f = open("pickles/relations_dev.pickle", "rb")
    relations_dev = pickle.load(f)
    f.close()
    f = open("pickles/relations_train.pickle", "rb")
    relations_train = pickle.load(f)
    f.close()
    f = open("pickles/relations_test.pickle", "rb")
    relations_test = pickle.load(f)
    f.close()
    files = glob.glob("./pickles/2019*")
    for direct in files:
        # identify labels and embedding
        label = glob.glob(direct+"/label*")
        embed = glob.glob(direct+"/m*")
        # read embedding and labels
        f = open(label[0], "rb")
        label_subst = pickle.load(f)
        f.close()
        f = open(embed[0], "rb")
        m = pickle.load(f)
        f.close()
        # convert relations
        mName = re.sub(".pickle", "", os.path.basename(embed[0]))
        function = [item for item in dir(trainW) if mName in item]
        if len(function) == 0:
            function = "convert_relations"
        else:
            function = function[0]
        convert_relations = getattr(trainW, function)
        (input_train, output_train) = convert_relations(relations_train, label_subst, m)
        (input_dev, output_dev) = convert_relations(relations_dev, label_subst, m)
        (input_test, output_test) = convert_relations(relations_test, label_subst, m)
        # write out inputs and outputs
        # identify nn
        NN = glob.glob(files[0]+"/n*")
        for nn in NN:
            f = open(nn, "rb")
            exp = pickle.load(nn)
            f.close()
            reportTrain = classification_report(output_train, exp.network.predict(input_train), digits = 7, labels = np.unique(exp.network.predict(input_train)))
            reportDev = classification_report(output_dev, exp.network.predict(input_dev), digits = 7, labels = np.unique(exp.network.predict(input_dev)))