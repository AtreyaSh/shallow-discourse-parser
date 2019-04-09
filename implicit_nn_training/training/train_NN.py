#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input, SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2 # for keras 2
from keras.utils import np_utils
from keras import optimizers
# TODO: Implement own f1


def create_optimizer(method, learning_rate, momentum, decay):
    """Creates the optimizing function based on training arguments"""
    if method == "nag":
        return optimizers.SGD(lr = learning_rate, momentum=momentum, decay = decay, nesterov = True)
    elif method == "sgd":
        return optimizers.SGD(lr = learning_rate, momentum=momentum, decay = decay)
    else:
        return optimizers.adam(lr = learning_rate)

def train_theanet(method, learning_rate, momentum, decay, regularization, hidden,
                  min_improvement, validate_every, patience, weight_lx, hidden_lx,
                  embeddings, direct, name = 0):
    ''' train neural network, calculate confusion matrix, save neural network'''

    input_train, output_train, input_dev, output_dev, input_test, output_test, label_subst = embeddings
    ## Training options:
    ## 1.) use 90% training set to train, use remaining 10% to validate, use test set to test
    #split_point = int(len(input_train)*0.9)
    #train_data = (input_train[:split_point], output_train[:split_point])
    #valid_data = (input_train[split_point:], output_train[split_point:])
    #test_data = (input_dev, output_dev)
    ## 2.) use 100% training set to train, use test set to validate and to test
    accs = []
    train_accs = []
    dev_accs = []
    output_train = np_utils.to_categorical(output_train, num_classes=None)
    num_classes = output_train.shape[1]
    output_dev = np_utils.to_categorical(output_dev, num_classes=num_classes)
    output_test = np_utils.to_categorical(output_test, num_classes=num_classes) 
    
    for nexp in range(5):
        best_eval, best_eval_test = 0, 0
        inlayer = Input((input_dev.shape[1],))
        # TODO: Activation function
        # TODO: learning rate
        # TODO: method
        # TODO: regularization
        # TODO: hidden
        # TODO: min improvement
        # TODO: patience
        # TODO: weight lx
        # TODO: hiddenlx
        output = Dense(output_train.shape[1], activation = 'relu')(inlayer)
        opt = create_optimizer(method, learning_rate, momentum, decay)
        
        # Early stopping monitors the development of loss and aborts the training if it starts
        # to increase again (prevents overfitting)
        es = EarlyStopping(monitor='acc',
                        patience=patience,
                        mode='min',
                        min_delta = min_improvement,
                        verbose=1)

        model = Model(inputs = inlayer, outputs = output)
        model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        
        for epoch in range(25):
            _ = model.fit(input_train, output_train, 
                        epochs = epoch +1, 
                        batch_size = 80,
                        verbose = 1,
                        callbacks = [es],
                        initial_epoch = epoch)
            if epoch % validate_every == 0:
                print("\tdev\ttest")
                dev_scores = model.evaluate(input_dev, output_dev, batch_size = 80, verbose = 0)
                print("acc.\t%.2f%%" % (dev_scores[1]*100), end = "", flush = True)
                test_scores = model.evaluate(input_test, output_test, verbose=0, batch_size= 80)
                print ("\t%.2f%%" % (test_scores[1]*100), end="", flush=True)
                # TODO: Fix this shit

                if dev_scores[1] > best_eval:
                    print("\t*")
                    best_val = dev_scores[1]
        accs.append(test_scores[0])
        dev_accs.append(dev_scores[0])

    # confmx = confusion_matrix(exp.network.predict(test_data[0]), test_data[1])
    # acc = float(sum(np.diag(confmx)))/sum(sum(confmx))
    # report = classification_report(exp.network.predict(test_data[0]),test_data[1])
    # accs.append(acc)
    # train_acc = accuracy_score(exp.network.predict(train_data[0]),train_data[1])
    # train_accs.append(train_acc)
    # valid_acc = accuracy_score(exp.network.predict(valid_data[0]),valid_data[1])
    # valid_accs.append(valid_acc)
    # file = open("pickles/"+str(direct)+"/neuralnetwork_"+str(name)+".pickle", "wb")
    # pickle.dump(exp.network, file, protocol=pickle.HIGHEST_PROTOCOL)
    # file.close()
    return np.average(accs),np.average(dev_accs),0.00,"N/A"
    # return np.average(accs), np.average(valid_accs), np.average(train_accs), report