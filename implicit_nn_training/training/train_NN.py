#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theanets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from .metrics import Metrics

    
def train_theanet(method, learning_rate, momentum, decay, regularization, hidden,
                  min_improvement, validate_every, patience, weight_lx, hidden_lx,
                  embeddings, direct, name = 0):
    ''' train neural network, calculate confusion matrix, save neural network'''
    input_train, output_train, input_dev, output_dev, input_test, output_test, label_subst = embeddings
    train_data = (input_train, output_train)
    test_data = (input_dev, output_dev)
    valid_data = (input_dev, output_dev)
    metrics = Metrics()
    metrics.register_datasets(["train", "dev", "test"])

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
    
    metrics.update_metrics(train_data[1], exp.network.predict(train_data[0]), "train")
    metrics.update_metrics(valid_data[1], exp.network.predict(valid_data[0]), "dev")
    metrics.update_metrics(test_data[1], exp.network.predict(test_data[0]), "test")
    # confmx = confusion_matrix(test_data[1], exp.network.predict(test_data[0]))
    # acc = float(sum(np.diag(confmx)))/sum(sum(confmx))
    report = classification_report(test_data[1], exp.network.predict(test_data[0]), digits = 7, labels = np.unique(exp.network.predict(test_data[0])))
    # accs.append(acc)
    # train_acc = accuracy_score(train_data[1], exp.network.predict(train_data[0]))
    # train_accs.append(train_acc)
    # valid_acc = accuracy_score(valid_data[1], exp.network.predict(valid_data[0]))
    # valid_accs.append(valid_acc)
    file = open("pickles/"+str(direct)+"/neuralnetwork_"+str(name)+".pickle", "wb")
    pickle.dump(exp.network, file, protocol=pickle.HIGHEST_PROTOCOL)
    accs, recs, precs, f1s = metrics.get_averages_ordered_by(["train", "dev", "test"])
    
    file.close()
    return accs, report, recs, precs, f1s
