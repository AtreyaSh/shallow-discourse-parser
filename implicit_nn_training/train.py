import training.train_embedding as trainW
from training.train_embedding import convert_relations
from training.train_NN import train_theanet
import sys
import csv
import os
import pickle
import datetime
import argparse

# TODO: gensim runs on cpu -> optimize this for cluster
# TODO: fix warnings of runs -> tupe seq indexing of multidim arrays, recall/f-score ill-defined for samples
# TODO: fix warning in gensim where supplied example does not match expected, perhaps related to m.corpus_count

def combination(trainpath, devpath, testpath, args):
    # example for parameter (learning_rate, min_improvement, method are fix in this code)
    parameter = [(0.1, 95, "prelu", "l2", 0.0001, "l1", 0.1), (0.3, 100, "prelu", "l2", 0.0001, "l2", 0.1 ),
                 (0.35, 95, "rect:max", "l1", 0.0001, "l1", 0.1), (0.35, 95, "prelu", "l2", 0.0001, "l1", 0.1),
                 (0.35, 100, "prelu", "l2", 0.0001, "l1", 0.1), (0.4, 80, "prelu", "l2", 0.0001, "l1", 0.1)]
    current_time = getCurrentTime()
    current_run_name = "%s_%s" % (current_time, args.name)
    os.makedirs("pickles/"+current_run_name)
    csvfile = open('pickles/'+ current_run_name + '/Results.csv', 'w')
    fieldnames = ['VectorTraining','NN Training', 'Test Acc', 'Valid Acc', 'Train Acc', "MinImprov", "Method", "LernR", "Momentum", "Decay", "Regular.", "Hidden", "Report"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()
    counter_vec = 0
    counter_nn = 0
    for iter1 in range(1,4):
        #train vectors 3x
        if args.debug:
            embeddings = restart()
        else:
            embeddings = trainW.start_vectors("%sparses.json" % trainpath, "%sparses.json" % devpath,
                                              "%sparses.json" % testpath, "%srelations.json" % trainpath,
                                              "%srelations.json" % devpath, "%srelations.json" % testpath,
                                              args.emb, current_run_name, args.name)
        for iter2 in range(len(parameter)*5):
            #for each trained vectors train each NN parameter combination 5x
            if iter2%5 == 0:
                triple = parameter[iter2//5]
            (acc, valid_acc, train_acc, report) = train_theanet('nag', 0.0001, triple[0],
                                                                             triple[4], triple[6],(triple[1],triple[2]), 0.001, 5,5, 
                                                                             triple[3], triple[5], embeddings, current_run_name, str(counter_vec)+"_"+str(counter_nn))
            writer.writerow({'VectorTraining': counter_vec ,'NN Training': counter_nn,  'Test Acc': round(acc*100,2), 'Valid Acc': round(valid_acc*100,2) , 
                   "Train Acc": round(train_acc*100,2), "MinImprov": 0.001, "Method": "nag", "LernR": 0.0001,"Momentum":triple[0], 
                   "Decay":"{0}={1}".format(triple[3], triple[4]), "Regular.": "{0}={1}".format(triple[5],triple[6]), "Hidden": 
                   "({0}, {1})".format(triple[1],triple[2]),"Report":report})
            counter_nn+=1
            csvfile.flush()
        counter_vec+=1
    csvfile.close()

def grid(trainpath, devpath, testpath, args):
    current_time = getCurrentTime()
    current_run_name = "%s_%s" % (current_time, args.name)
    os.makedirs("pickles/"+current_run_name)
    csvfile = open('pickles/'+ current_run_name + '/' + 'Results.csv', 'w')
    fieldnames = ['Counter','Test Acc', 'Valid Acc', 'Train Acc', "MinImprov", "Method", "LernR", "Momentum", "Decay", "Regular.", "Hidden", "Report"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()
    if args.debug:
        embeddings = restart()
    else:
        embeddings = trainW.start_vectors("%sparses.json" % trainpath, "%sparses.json" % devpath,
                                          "%sparses.json" % testpath, "%srelations.json" % trainpath,
                                          "%srelations.json" % devpath, "%srelations.json" % testpath,
                                          args.emb, current_run_name, args.name)
    #different parameter options, e.g.:
    method = ['nag']
    min_improvements = [0.001]
    learning_rates = [0.0001]
    w_h = [('l2', 'l1'), ('l1', 'l2'), ('l2','l2'), ("l1", "l1")]
    momentum_alts = [0.1, 0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    hidden_alts = [60, 65, 70, 75, 80, 85, 90, 95, 100]
    act_funcs = ['rect:max','prelu','lgrelu']
    d_r = [(0.0001, 0.0001), (0.0001, 0.1)]
    ## more parameter options, e.g.:
    #method = ['nag', 'sgd', 'rprop','rmsprop', 'adadelta', 'hf', 'sample','layerwise']
    #min_improvements = [0.001, 0.005, 0.1, 0.2]
    #w_h = [('l2', 'l1'), ('l1', 'l2'), ('l2','l2'), ("l1", "l1")]
    #learning_rates = [0.0001, 0.0005, 0.001, 0.005]
    #momentum_alts = [0.1,0.2, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    #hidden_alts = [20, 30, 40, 50, 60, 65, 70, 75, 80, 85, 90, 95, 100]
    #act_funcs = ['linear','logistic','tanh+norm:z',  
    #             'softplus', #'softmax', --> is to bad
    #             'relu','rect:min', 'rect:max',
    #             'norm:mean','norm:max', 'norm:std',
    #             'norm:z', 'prelu','lgrelu']

    #decay = [0.0001, 0.1, 0.2, 0.5]
    #regularization = [0.0001, 0.1, 0.2, 0.5]
    #d_r = []
    #for i in decay:
    #    for j in regularization:
    #        d_r.append((i,j))
    counter = 0
    for h in method:
        for i in min_improvements:
            for j in learning_rates:
                for k in momentum_alts:
                    for l in hidden_alts:
                        for m in act_funcs:
                            for n in w_h:
                                for o in d_r:
                                    (acc, valid_acc, train_acc, report) = train_theanet(h, j, k, o[0], o[1], (l, m), i, 5,5, n[0], 
                                                                                n[1], embeddings, current_run_name, counter)
                                    writer.writerow({'Counter': counter, 'Test Acc': round(acc*100,2), 'Valid Acc': round(valid_acc*100,2) , 
                                                     "Train Acc": round(train_acc*100,2),
                                                     "MinImprov": i, "Method": h, "LernR": j,
                                                     "Momentum":k, "Decay":"{0}={1}".format(n[0], o[0]), "Regular.": "{0}={1}".format(n[1], o[1]),
                                                     "Hidden": "({0}, {1})".format(l,m), "Report": report})
                                    counter += 1
                                    csvfile.flush()
    csvfile.close()

def single(trainpath, devpath, testpath, args):
    ''' train the neural network with a given parameter setting'''
    current_time = getCurrentTime()
    current_run_name = "%s_%s" % (current_time, args.name)
    os.makedirs("pickles/"+current_run_name)
    csvfile = open('pickles/'+ current_run_name + '/' + 'Results.csv', 'w')
    fieldnames = ['Test Acc', 'Valid Acc', 'Train Acc', "MinImprov", "Method", "LernR", "Momentum", "Decay", "Regular.", "Hidden", "Report"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    csvfile.flush()
    if args.debug:
        embeddings = restart()
    else:
        # make sure we dont break paths with appending
        embeddings = trainW.start_vectors("%sparses.json" % trainpath, "%sparses.json" % devpath,
                                          "%sparses.json" % testpath, "%srelations.json" % trainpath,
                                          "%srelations.json" % devpath, "%srelations.json" % testpath,
                                          args.emb, current_run_name, args.name)
    # train neural network
    method, learning_rate, momentum, decay, regularization, hidden, min_improvement, validate_every, patience, weight_lx, hidden_lx = 'nag', 0.0001, 0.6, 0.0001, 0.0001, (60, 'lgrelu'), 0.001, 5, 5, "l1", "l2"
    (acc, valid_acc, train_acc, report) = train_theanet(method, learning_rate, momentum, decay, regularization, hidden, min_improvement, validate_every, patience, weight_lx, hidden_lx, embeddings, current_run_name)
    writer.writerow({'Test Acc': round(acc*100,2), 'Valid Acc': round(valid_acc*100,2), 
                                                     "Train Acc": round(train_acc*100,2),
                                                     "MinImprov": min_improvement, "Method": method, "LernR": learning_rate,
                                                     "Momentum":momentum, "Decay":"{0}={1}".format(weight_lx, decay), "Regular.": "{0}={1}".format(hidden_lx, regularization),
                                                     "Hidden": "({0}, {1})".format(hidden[0],hidden[1]), 'Report': report})
    csvfile.flush()
    csvfile.close()

def restart():
    """saves the time reconverting the relations each and every time."""
    pseudo_tuple = []
    for load_file in ["input_train", "output_train", "input_dev", "output_dev", "input_test" , "output_test", "label_subst"]:
        with open("pickles/%s.pickle" % load_file, "rb") as f:
            loaded = pickle.load(f)
            pseudo_tuple.append(loaded)
    return tuple(pseudo_tuple)

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

def getCurrentTime():
    return datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", type=str, default="data/en.test/",
                        help="Path to test data folder")
    parser.add_argument("--train", type=str, default="data/en.train/",
                        help="Path to train data folder")
    parser.add_argument("--dev", type=str, default="data/en.dev/",
                        help="Path to development data folder")
    parser.add_argument("--emb", type=str, default="data/GoogleNews-vectors-negative300.bin",
                        help="Path to pretrained embeddings")
    parser.add_argument("--mode", type=str, default="single",
                        help="what to test")
    parser.add_argument("--name", type=str)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    if args.debug:
        print("WARNING DEBUG MODE")
    #ensure we later don't break paths
    args.test = args.test if args.test.endswith("/") else "%s/" % args.test
    args.dev = args.dev if args.dev.endswith("/") else "%s/" % args.dev
    args.train = args.train if args.train.endswith("/") else "%s/" % args.train
    if args.mode == "single":
        single(args.train, args.dev, args.test, args)
    elif args.mode == "grid":
        grid(args.train, args.dev, args.test, args)
    elif args.mode == "combination":
        combination(args.train, args.dev, args.test, args)
    else:
        print("Unknown args")
        sys.exit()
