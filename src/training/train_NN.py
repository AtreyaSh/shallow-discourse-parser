#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theanets
from sklearn.metrics import classification_report
import pickle
from keras.utils import np_utils
from keras.models import Model
from keras.layers import Dense, Dropout, Input
from keras.layers.advanced_activations import PReLU
from keras.layers import Activation
from keras.regularizers import l2, l1 # for keras 2
from keras.utils import np_utils
import numpy as np
from keras import optimizers
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
    reportTrain = classification_report(train_data[1], exp.network.predict(train_data[0]), digits = 7, output_dict=True)
    reportDev = classification_report(valid_data[1], exp.network.predict(valid_data[0]), digits = 7, output_dict=True)
    reportTest = classification_report(test_data[1], exp.network.predict(test_data[0]), digits = 7, output_dict=True)
    file = open("pickles/"+str(direct)+"/neuralnetwork_"+str(name)+".pickle", "wb")
    pickle.dump(exp.network, file, protocol=pickle.HIGHEST_PROTOCOL)
    accs, recs, precs, f1s = metrics.get_averages_ordered_by(["train", "dev", "test"])
    file.close()
    return accs, reportTrain, reportDev, reportTest, recs, precs, f1s


def create_activation(activation):
    if activation == "prelu":
        return PReLU()
    elif activation == "relu":
        return Activation("relu")
    elif activation == "tanh":
        return Activation("tanh")
    elif activation == "softmax":
        return Activation("softmax")
    else:
        return Activation("tanh")

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
        hidden = Dense(hidden_nodes, kernel_regularizer=w_reg, bias_regularizer = b_reg)(inlayer)
        act = create_activation(activation_hidden)(hidden)
        if drop:
            drop = Dropout(0.5)(act)
            output = Dense(output_shape, activation = activation_output)(drop)
        else:
            output = Dense(output_shape, activation = activation_output)(act)
        model = Model(inputs = inlayer, outputs = output)
        return model
    elif depth == 3:
        hidden = Dense(hidden_nodes, kernel_regularizer=w_reg, bias_regularizer = b_reg)(inlayer)
        act1 = create_activation(activation_hidden)(hidden)
        hidden2 = Dense(int(round(hidden_nodes*1.25)))(act1)
        act2 = create_activation(activation_hidden)(hidden2)
        if drop:
            drop = Dropout(0.5)(act2)
            output = Dense(output_shape, activation = activation_output)(drop)
        else:
            output = Dense(output_shape, activation = activation_output)(act2)
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
                  embeddings, direct, name = 0, depth = 3, drop = False, epochs = 50):
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
    
    for nexp in range(10):
# def create_model(depth, hidden_nodes, activation_hidden, activation_output, output_shape,
#                 input_shape, drop = True):
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
        metrics.update_metrics(y_true = np.argmax(train[1], axis=1),
                                y_pred = np.argmax(model.predict(train[0]), axis=1),
                                name = "train")
        metrics.update_metrics(y_true = np.argmax(dev[1], axis=1),
                                y_pred = np.argmax(model.predict(dev[0]), axis=1),
                                name = "dev")
        metrics.update_metrics(y_true = np.argmax(test[1], axis=1),
                                y_pred = np.argmax(model.predict(test[0]), axis=1),
                                name = "test")
        
        del model # Reset

    temp = metrics.get_averages_ordered_by(["test", "dev", "train"])
    accs, recs, precs, f1s = temp
    return accs, "N/A", "N/A", "N/A", recs, precs, f1s
    
