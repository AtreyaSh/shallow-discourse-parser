#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theanets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle

def train_theanet(method, learning_rate, momentum, decay, regularization, hidden,
                  min_improvement, validate_every, patience, weight_lx, hidden_lx,
                  input_train, output_train, input_dev, output_dev, label_subst, direct, name = 0):
    ''' train neural network, calculate confusion matrix, save neural network'''
    ## Training options:
    ## 1.) use 90% training set to train, use remaining 10% to validate, use test set to test
    #split_point = int(len(input_train)*0.9)
    #train_data = (input_train[:split_point], output_train[:split_point])
    #valid_data = (input_train[split_point:], output_train[split_point:])
    #test_data = (input_dev, output_dev)
    ## 2.) use 100% training set to train, use test set to validate and to test
    train_data = (input_train, output_train)
    test_data = (input_dev, output_dev)
    valid_data = (input_dev, output_dev)
    accs = []
    train_accs = []
    valid_accs = []
    exp = theanets.main.Experiment(theanets.feedforward.Classifier, layers=(len(input_train[0]), hidden, len(label_subst)), loss='XE')
    if weight_lx == "l1":
        if hidden_lx == "l1":
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l1=decay,
                                            hidden_l1=regularization,
                                            min_improvement=min_improvement,
                                            validate_every=validate_every,
                                            patience=patience)
        else:
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l1=decay,
                                            hidden_l2=regularization,
                                            min_improvement=min_improvement,
                                            validate_every=validate_every,
                                            patience=patience)
    else:
        if hidden_lx == "l1":   
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l2=decay,
                                            hidden_l1=regularization,
                                            min_improvement=min_improvement,
                                            validate_every=validate_every,
                                            patience=patience)
        else:
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l2=decay,
                                            hidden_l2=regularization,
                                            min_improvement=min_improvement,
                                            validate_every=validate_every,
                                            patience=patience)
    confmx = confusion_matrix(exp.network.predict(test_data[0]), test_data[1])
    acc = float(sum(np.diag(confmx)))/sum(sum(confmx))
    report = classification_report(exp.network.predict(test_data[0]),test_data[1])
    accs.append(acc)
    train_acc = accuracy_score(exp.network.predict(train_data[0]),train_data[1])
    train_accs.append(train_acc)
    valid_acc = accuracy_score(exp.network.predict(valid_data[0]),valid_data[1])
    valid_accs.append(valid_acc)
    file = open("pickles/"+str(direct)+"/neuralnetwork_"+str(name)+".pickle", "wb")
    pickle.dump(exp.network, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return np.average(accs), np.average(valid_accs), np.average(train_accs), report