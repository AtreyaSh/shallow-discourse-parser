#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import training.train_embedding as trainW
from training.train_embedding import convert_relations, read_file, RelReader
from training.metrics import Metrics
import logging
import gensim
from training.train_NN import train_theanet, train_keras, create_activation, create_model, create_optimizer, create_weight_regularizer
import sys
import csv
import os
import json
import pickle
import datetime
import argparse
def start_vectors_b(parses_train_filepath, parses_dev_filepath, parses_test_filepath, parses_blind_filepath, relations_train_filepath,
                  relations_dev_filepath, relations_test_filepath, relations_blind_filepath, googlevecs_filepath, direct, name):
    """ train vectors """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Initalize semantic model (with None data)
    m = gensim.models.word2vec.Word2Vec(None, size=300, window=8, min_count=3, workers=4, negative=20, sg=1, hs = 1)
    
    print("Reading from source...")
    parses = json.load(open(parses_train_filepath))
    parses.update(json.load(open(parses_dev_filepath)))
    parses.update(json.load(open(parses_test_filepath)))
    parses.update(json.load(open(parses_blind_filepath)))
    (relations_train, all_relations_train) = read_file(relations_train_filepath, parses)
    (relations_dev, all_relations_dev) = read_file(relations_dev_filepath, parses)
    (relations_test, all_relations_test) = read_file(relations_test_filepath, parses)
    (relations_blind, all_relations_blind) = read_file(relations_blind_filepath, parses)
        
    relations = relations_train + relations_dev 
    all_relations = all_relations_train + all_relations_dev + all_relations_blind
    # Substitution dictionary for class labels to integers
    label_subst = dict([(y,x) for x,y in enumerate(set([r[0][0] for r in relations]))])
    print(("Label subst", label_subst))
    print("Build vocabulary...")
    m.build_vocab(RelReader(all_relations))
    print("Reading pre-trained word vectors...")
    m.intersect_word2vec_format(googlevecs_filepath, binary=True)
    print("Training segment vectors...")
    for iter in range(1, 4):
        ## Training of word vectors
        m.alpha = 0.01/(2**iter)
        m.min_alpha = 0.01/(2**(iter+1))
        print("Vector training iter", iter, m.alpha, m.min_alpha)
        # m.train(ParseReader(parses), total_examples = m.corpus_count, epochs=m.epochs)
        m.train(RelReader(all_relations), total_examples = m.corpus_count, epochs=m.epochs)
    # dump pickles to save basic data
    (input_train, output_train) = convert_relations(relations_train, label_subst, m)
    (input_dev, output_dev) = convert_relations(relations_dev, label_subst, m)
    (input_test, output_test) = convert_relations(relations_test, label_subst, m)
    (input_blind, output_blind) = convert_relations(relations_blind, label_subst, m)
    return input_train, output_train, input_dev, output_dev, input_test, output_test,input_blind, output_blind, label_subst

if __name__ == "__main__":
    testpath = "data/en.test/"
    trainpath ="data/en.train/"
    devpath = "data/en.dev/"
    blindpath = "data/en.blind_test/"
    embpath = "data/GoogleNews-vectors-negative300.bin"
    embeddings = start_vectors_b("%sparses.json" % trainpath, "%sparses.json" % devpath,
                                          "%sparses.json" % testpath, "%sparses.json" % blindpath,
                                          "%srelations.json" % trainpath, "%srelations.json" % devpath, 
                                          "%srelations.json" % testpath, "%srelations.json" % blindpath,
                                          embpath, "final_test", "blind")
    method, learning_rate, momentum, decay, regularization = 'adam', 0.0001, 0.6, 0.0001, 0.0001
    hidden, min_improvement, validate_every, patience, weight_lx, hidden_lx = (1000, 'prelu'), 0.001, 5, 5, "l2", "l2"
    dropout, epochs, depth = False, 50, 3
    ''' train neural network, calculate confusion matrix, save neural network'''
    input_train, output_train, input_dev, output_dev, input_test, output_test, input_blind, output_blind, label_subst = embeddings
    
    train = (input_train, np_utils.to_categorical(output_train, num_classes=None))
    num_classes = train[1].shape[1]
    dev = (input_dev, np_utils.to_categorical(output_dev, num_classes=num_classes))
    test = (input_test, np_utils.to_categorical(output_test, num_classes=num_classes))
    blind = (input_blind, np_utils.to_categorical(output_blind, num_classes=num_classes))
    metrics = Metrics()
    metrics.register_datasets(["train", "dev", "test", "blind"])
    
    for nexp in range(5):
        w_reg = create_weight_regularizer(decay, weight_lx)
        b_reg = create_weight_regularizer(regularization, hidden_lx)
        model = create_model(depth=depth, hidden_nodes=hidden[0], activation_hidden=hidden[1], 
                            activation_output='softmax',output_shape=num_classes, input_shape=train[0].shape[1],
                            drop=drop, w_reg = w_reg, b_reg = b_reg)
        opt = create_optimizer(method, learning_rate, momentum, decay)
    
        model.compile(loss='categorical_crossentropy', 
                    optimizer=opt, 
                    metrics=['accuracy'])
        _ = model.fit(train[0], train[1], 
                        epochs = epochs, 
                        batch_size = 500,
                        validation_data = (dev[0], dev[1]),
                        verbose = 1)

        # Done training, now compute acc, prec etc which only makes sense after training.
        metrics.update_metrics(model, train, "train")
        metrics.update_metrics(model, dev, "dev")
        metrics.update_metrics(model, test, "test")
        metrics.update_metrics(model, blind, "blind")
        
    temp = metrics.get_averages_ordered_by(["train", "dev", "test", "blind"])
    accs, recs, precs, f1s
    print("%s,%s,%s,%s" % (accs[3],recs[3],precs[3],f1s[3]))    
    ## interface with the training in case mode is single
