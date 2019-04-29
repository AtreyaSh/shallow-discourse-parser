#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import training.train_embedding as trainW
from training.train_embedding import convert_relations
from training.train_NN import train_theanet
import csv
import os
import pickle
import re
import glob
import datetime
import argparse
import pandas as pd

####################################
# combination hyperparameter search
####################################

def combination(trainpath, devpath, testpath, vecpath, network1, network2, iterations, training):
    current_time = getCurrentTime()
    current_run_name = "%s_%s" % (current_time, "comparison")
    os.makedirs("pickles/"+current_run_name)
    csvfile = open('pickles/'+ current_run_name + '/Results.csv', 'w')
    fieldnames = ['Counter','Word-Model','Neural-Model','Train Acc', 'Dev Acc', 'Test Acc', "Train Recall","Dev Recall", "Test Recall","Train Precision", "Dev Precision","Test Precision" , "Train F1", "Dev F1","Test F1", "MinImprov", "Method", "LernR", "Momentum", "Decay", "Regular.", "Hidden", 'ReportTrain','ReportDev','ReportTest']
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()
    for network in [network1, network2]:
        # get network name and model name
        countN = re.sub(r"[^\d+]", "", os.path.basename(network))
        wFile = glob.glob(os.path.dirname(network)+"/m*")[0]
        countW = re.sub(r"[^\d+]", "", os.path.basename(wFile))
        # find model parameters
        df = pd.read_csv(os.path.dirname(network)+"/Results.csv")
        if "Counter" in df.keys():
            df = df.loc[df['Counter'] == int(countN)]
        # derive model parameters from imports and csv file
        method = df["Method"].tolist()[0]
        learning_rate = df['LernR'].tolist()[0]
        momentum = df['Momentum'].tolist()[0]
        min_improvement = df['MinImprov'].tolist()[0]
        hidden1, hidden2 = [re.sub(r"[() ]", "", o) for o in df['Hidden'].tolist()[0].split(",")]
        hidden = (int(hidden1), str(hidden2),)
        weight_lx, decay = df['Decay'].tolist()[0].split("=")
        hidden_lx, regularization = df['Regular.'].tolist()[0].split("=")
        validate_every = 5 
        patience = 5
        counter = 0
        ID = str(countW)+"_"+str(countN)
        for _ in range(iterations):
            counterName = ID+"_"+str(counter)
            embeddings = trainW.start_vectors("%sparses.json" % trainpath, "%sparses.json" % devpath,
                                              "%sparses.json" % testpath, "%srelations.json" % trainpath,
                                              "%srelations.json" % devpath, "%srelations.json" % testpath,
                                              vecpath, current_run_name, "m_"+countW)
            if training == "theanets":
                accs, reportTrain, reportDev, reportTest, recs, precs, f1s = train_theanet(method, float(learning_rate), 
                                                                                       float(momentum), float(decay), 
                                                                                       float(regularization), hidden, 
                                                                                       float(min_improvement),
                                                                                       validate_every,
                                                                                       patience, weight_lx, 
                                                                                       hidden_lx, embeddings, 
                                                                                       direct=current_run_name, 
                                                                                       name=counterName)
            elif training == "keras":
                accs, reportTrain, reportDev, reportTest, recs, precs, f1s = train_keras(method, float(learning_rate), 
                                                                                       float(momentum), float(decay), 
                                                                                       float(regularization), hidden, 
                                                                                       float(min_improvement),
                                                                                       validate_every,
                                                                                       patience, weight_lx, 
                                                                                       hidden_lx, embeddings, 
                                                                                       direct=current_run_name, 
                                                                                       name=counterName)
            else:
                print("Unknown training framework %s." % args.training)
            writer.writerow({'Counter':str(counter),'Word-Model':"m_"+countW,'Neural-Model':countN,
                            'Train Acc': round(accs[0],5), 'Dev Acc': round(accs[1],5) , "Test Acc": round(accs[2],5), 
                            'Train Recall': round(recs[0],5), 'Dev Recall': round(recs[1],5) , "Test Recall": round(recs[2],5), 
                            'Train Precision': round(precs[0],5), 'Dev Precision': round(precs[1],5) , "Test Precision": round(precs[2],5), 
                            'Train F1': round(f1s[0],5), 'Dev F1': round(f1s[1],5) , "Test F1": round(f1s[2],5), 
                            "MinImprov": min_improvement, "Method": method, "LernR": learning_rate,
                            "Momentum":momentum, "Decay":"{0}={1}".format(weight_lx, decay), "Regular.": "{0}={1}".format(hidden_lx, regularization),
                            "Hidden": "({0}, {1})".format(hidden[0],hidden[1]), 'ReportTrain': str(reportTrain), 'ReportDev': str(reportDev), 'ReportTest': str(reportTest)})
            counter += 1
            csvfile.flush()
    csvfile.close()
    return 0

####################################
# aux functions/debugging
####################################

def restart():
    """saves the time reconverting the relations each and every time."""
    pseudo_tuple = []
    for load_file in ["input_train", "output_train", "input_dev", "output_dev", "input_test" , "output_test", "label_subst"]:
        with open("pickles/%s.pickle" % load_file, "rb") as f:
            loaded = pickle.load(f)
            pseudo_tuple.append(loaded)
    return tuple(pseudo_tuple)

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

def create_training_parameters(parameter_lists):
    """function that creates all possible parameter combinations"""
    pass

def import_pickle(path):
    f = open(path, "rb")
    m = pickle.load(f)
    f.close()
    f = open(os.path.dirname(path)+"/label_subst.pickle", "rb")
    label_subst = pickle.load(f)
    f.close()
    f = open("pickles/relations_dev.pickle", "rb")
    relations_dev = pickle.load(f)
    f.close()
    f = open("pickles/relations_train.pickle", "rb")
    relations_train = pickle.load(f)
    f.close()
    (input_train, output_train) = convert_relations(relations_train, label_subst, m)
    (input_dev, output_dev) = convert_relations(relations_dev, label_subst, m)
    return input_train, output_train, input_dev, output_dev, label_subst

####################################
# main command call
####################################

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, default="data/en.train/",
                        help="Path to train data folder, defaults to 'data/en.train/'")
    parser.add_argument("--dev", type=str, default="data/en.dev/",
                        help="Path to development data folder, defaults to 'data/en.dev/'")
    parser.add_argument("--test", type=str, default="data/en.test/",
                        help="Path to test data folder, defaults to 'data/en.test/'")
    parser.add_argument("--emb", type=str, default="data/GoogleNews-vectors-negative300.bin",
                        help="Path to pretrained google embeddings, defaults to 'data/GoogleNews-vectors-negative300.bin'")
    parser.add_argument("--iterations", type=int, default=20,
                        help="number of iterations for each network, defaults to 20")
    parser.add_argument("--training", type=str, default="theanets",
                        help="Which NN training framework to use (theanets/keras), defaults to 'theanets'")
    requiredNamed = parser.add_argument_group('required named arguments')
    requiredNamed.add_argument('-n1', '--network1', help='path to network 1', required=True)
    requiredNamed.add_argument('-n2', '--network2', help='path to network 2', required=True)
    args = parser.parse_args()
    # make combination run
    combination(args.train, args.dev, args.test, args.emb, args.network1, args.network2, args.iterations, args.training)
