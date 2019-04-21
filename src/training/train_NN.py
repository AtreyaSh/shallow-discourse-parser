#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theanets
from sklearn.metrics import classification_report
import pickle
from .metrics import Metrics

####################################
# theanets baseline training
####################################

def train_theanet(method, learning_rate, momentum, decay, regularization, hidden,
                  min_improvement, validate_every, patience, weight_lx, hidden_lx,
                  embeddings, direct, name = 0):
    ''' train neural network, calculate confusion matrix, save neural network'''
    input_train, output_train, input_dev, output_dev, input_test, output_test, label_subst = embeddings
    train_data = (input_train, output_train)
    valid_data = (input_dev, output_dev)
    test_data = (input_test, output_test)
    metrics = Metrics()
    metrics.register_datasets(["train", "dev", "test"])
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
    
    metrics.update_metrics(train_data[1], exp.network.predict(train_data[0]), "train")
    metrics.update_metrics(valid_data[1], exp.network.predict(valid_data[0]), "dev")
    metrics.update_metrics(test_data[1], exp.network.predict(test_data[0]), "test")
    reportTrain = classification_report(train_data[1], exp.network.predict(train_data[0]), digits = 7)
    reportDev = classification_report(valid_data[1], exp.network.predict(valid_data[0]), digits = 7)
    reportTest = classification_report(test_data[1], exp.network.predict(test_data[0]), digits = 7)
    file = open("pickles/"+str(direct)+"/neuralnetwork_"+str(name)+".pickle", "wb")
    pickle.dump(exp.network, file, protocol=pickle.HIGHEST_PROTOCOL)
    accs, recs, precs, f1s = metrics.get_averages_ordered_by(["train", "dev", "test"])
    file.close()
    return accs, reportTrain, reportDev, reportTest, recs, precs, f1s