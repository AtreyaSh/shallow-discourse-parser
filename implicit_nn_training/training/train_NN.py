#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import theanets
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import pickle
from .metrics import Metrics

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
    
    for nexp in range(10):
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
        # es = EarlyStopping(monitor='acc',
        #                 patience=patience,
        #                 mode='min',
        #                 min_delta = min_improvement,
        #                 verbose=0)

        model.compile(loss='categorical_crossentropy', 
                    optimizer=opt, 
                    metrics=['accuracy'])
        _ = model.fit(train[0], train[1], 
                        epochs = epochs, 
                        batch_size = 500,
                        validation_data = (dev[0], dev[1]),
                        verbose = 1)
        # for epoch in tqdm(range(epochs)):
        #     _ = model.fit(train[0], train[1], 
        #                 epochs = epoch +1, 
        #                 batch_size = 80,
        #                 validation_data = (dev[0], dev[1]),
        #                 verbose = 0,
        #                 callbacks = [es],
        #                 initial_epoch = epoch)

        # Done training, now compute acc, prec etc which only makes sense after training.

        metrics.update_metrics(model, train, "train")
        metrics.update_metrics(model, dev, "dev")
        metrics.update_metrics(model, test, "test")
        
        del model # Reset

    temp = metrics.get_averages_ordered_by(["train", "dev", "test"])
    accs, recs, precs, f1s = temp
    return accs, "N/A", recs, precs, f1s
    
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
                                            patience=1)
        else:
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l1=decay,
                                            hidden_l2=regularization,
                                            min_improvement=min_improvement,
                                            validate_every=validate_every,
                                            patience=1)
    else:
        if hidden_lx == "l1":
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l2=decay,
                                            hidden_l1=regularization,
                                            min_improvement=min_improvement,
                                            validate_every=validate_every,
                                            patience=1)
        else:
            exp.train(train_data, valid_data, optimize=method,
                                            learning_rate=learning_rate,
                                            momentum=momentum,
                                            weight_l2=decay,
                                            hidden_l2=regularization,
                                            min_improvement=min_improvement,
                                            validate_every=validate_every,
                                            patience=1)
    
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
