#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import pickle
from keras.models import Model
from keras.layers import Dense, LSTM, Dropout, Input, SpatialDropout1D
from keras.callbacks import EarlyStopping
from keras.regularizers import l2, l1 # for keras 2
from keras.utils import np_utils, plot_model
from keras import optimizers
import theanets
from .metrics import Metrics
import warnings
warnings.filterwarnings('ignore')

def train_theanet(method, learning_rate, momentum, decay, regularization, hidden,
                  min_improvement, validate_every, patience, weight_lx, hidden_lx,
                  embeddings, direct, name = 0, depth = 2, drop = True, epochs = 25):
    ''' train neural network, calculate confusion matrix, save neural network'''
    input_train, output_train, input_dev, output_dev, input_test, output_test, label_subst = embeddings
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
    confmx = confusion_matrix(test_data[1], exp.network.predict(test_data[0]))
    acc = float(sum(np.diag(confmx)))/sum(sum(confmx))
    report = classification_report(test_data[1], exp.network.predict(test_data[0]), digits = 7, labels = np.unique(exp.network.predict(test_data[0])))
    accs.append(acc)
    train_acc = accuracy_score(train_data[1], exp.network.predict(train_data[0]))
    train_accs.append(train_acc)
    valid_acc = accuracy_score(valid_data[1], exp.network.predict(valid_data[0]))
    valid_accs.append(valid_acc)
    file = open("pickles/"+str(direct)+"/neuralnetwork_"+str(name)+".pickle", "wb")
    pickle.dump(exp.network, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return (np.average(accs), np.average(valid_accs), np.average(train_accs)), report, (0,0,0), (0,0,0), (0,0,0)


def create_weight_regularizer(decay, regularizer = "l2"):
    if regularizer == "l2":
        return l2(decay)
    elif regularizer == "l1":
        return l1(decay)

def create_model(depth, hidden_nodes, activation_hidden, activation_output, output_shape,
                input_shape, drop, w_reg, b_reg):
    """Creates the model based on inputs. Nothing special just cleanup"""
    inlayer = Input((input_shape,))
    if depth == 1:
        output = Dense(output_shape, activation = activation_output)(inlayer)
        model = Model(inputs = inlayer, outputs = output)
        return model
    elif depth == 2:
        hidden = Dense(hidden_nodes, activation = activation_hidden, kernel_regularizer=w_reg, bias_regularizer = b_reg)(inlayer)
        if drop:
            drop = Dropout(0.5)(hidden)
            output = Dense(output_shape, activation = activation_output)(drop)
        else:
            output = Dense(output_shape, activation = activation_output)(hidden)
        model = Model(inputs = inlayer, outputs = output)
        return model
    elif depth == 3:
        hidden = Dense(hidden_nodes, activation = activation_hidden, kernel_regularizer=w_reg, bias_regularizer = b_reg)(inlayer)
        hidden2 = Dense(hidden_nodes, activation = activation_hidden)(hidden)
        if drop:
            drop = Dropout(0.5)(hidden2)
            output = Dense(output_shape, activation = activation_output)(drop)
        else:
            output = Dense(output_shape, activation = activation_output)(hidden2)
        model = Model(inputs = inlayer, outputs = output)
        return model

    
def create_optimizer(method, learning_rate, momentum, decay):
    """Creates the optimizing function based on training arguments"""
    if method == "nag":
        return optimizers.SGD(lr = learning_rate, momentum=momentum, decay = decay, nesterov = True)
    elif method == "sgd":
        return optimizers.SGD(lr = learning_rate, momentum=momentum, decay = decay)
    elif method == "adam":
        return optimizers.adam(lr = learning_rate)
    else:
        return optimizers.adam(lr = learning_rate)

def train_keras(method, learning_rate, momentum, decay, regularization, hidden,
                  min_improvement, validate_every, patience, weight_lx, hidden_lx,
                  embeddings, direct, name = 0, depth = 2, drop = True, epochs = 25):
    ''' train neural network, calculate confusion matrix, save neural network'''
    input_train, output_train, input_dev, output_dev, input_test, output_test, label_subst = embeddings
    # print(output_train.shape)
    # output_train = np_utils.to_categorical(output_train)
    train = (input_train, np_utils.to_categorical(output_train, num_classes=None))
    num_classes = train[1].shape[1]
    dev = (input_dev, np_utils.to_categorical(output_dev, num_classes=num_classes))
    test = (input_test, np_utils.to_categorical(output_test, num_classes=num_classes))
    metrics = Metrics()
    metrics.register_datasets(["train", "dev", "test"])
    
    for nexp in range(3):
# def create_model(depth, hidden_nodes, activation_hidden, activation_output, output_shape,
#                 input_shape, drop = True):
        w_reg = create_weight_regularizer(decay, weight_lx)
        b_reg = create_weight_regularizer(regularization, hidden_lx)
        model = create_model(depth=depth, hidden_nodes=hidden[0], activation_hidden=hidden[1], 
                            activation_output='softmax',output_shape=num_classes, input_shape=train[0].shape[1],
                            drop=drop, w_reg = w_reg, b_reg = b_reg)
        opt = create_optimizer(method, learning_rate, momentum, decay)
        
        # Early stopping monitors the development of loss and aborts the training if it starts
        # to increase again (prevents overfitting)
        es = EarlyStopping(monitor='acc',
                        patience=patience,
                        mode='min',
                        min_delta = min_improvement,
                        verbose=0)

        model.compile(loss='categorical_crossentropy', 
                    optimizer=opt, 
                    metrics=['accuracy'])
        for epoch in range(epochs):
            _ = model.fit(train[0], train[1], 
                        epochs = epoch +1, 
                        batch_size = 80,
                        validation_data = (dev[0], dev[1]),
                        verbose = 0,
                        callbacks = [es],
                        initial_epoch = epoch)
            if epoch % validate_every == 0: # TODO: this is wrong right?
                print("\ttrain\tdev\ttest")
                # scores are loss, acc, f1, recall, precision
                train_scores = model.evaluate(train[0], train[1], verbose = 0, batch_size=80)
                print("acc.\t%.2f%%" % (train_scores[1]*100), end="", flush=True)
                dev_scores = model.evaluate(dev[0], dev[1], batch_size = 80, verbose = 0)
                print("\t%.2f%%" % (dev_scores[1]*100), end="", flush=True)
                test_scores = model.evaluate(test[0], test[1], verbose=0, batch_size= 80)
                print("\t%.2f%%" % (test_scores[1]*100), flush=True)

        # Done training, now compute acc, prec etc which only makes sense after training.

        metrics.update_metrics(model, train, "train")
        metrics.update_metrics(model, dev, "dev")
        metrics.update_metrics(model, test, "test")
        
        del model # Reset

    temp = metrics.get_averages_ordered_by(["train", "dev", "test"])
    accs, recs, precs, f1s = temp
    return accs, "N/A", recs, precs, f1s
    