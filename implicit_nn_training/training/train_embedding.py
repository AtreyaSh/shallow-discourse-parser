#!/usr/bin/env python3
# -*- coding: utf-8 -*-

####################################
# import dependencies
####################################

import json
import gensim
import logging
import pickle
import os
import numpy as np
import collections
from nltk.corpus import stopwords

####################################
# train word embeddings
####################################

# TODO: play with aggregations, try to play with context with exponential decay
# TODO: m_0 -> baseline, m_1 -> negative sampling, m_2 -> stop-words, m_3 -> contexts with exponential decay
# m_4 -> change structure of word embeddings where they return tokens of implicit arguments
# m_N* -> combinations

def start_vectors(parses_train_filepath, parses_dev_filepath, parses_test_filepath, relations_train_filepath,
                  relations_dev_filepath, relations_test_filepath, googlevecs_filepath, direct, name):
    """ train vectors """
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    # Initalize semantic model (with None data)
    if name == "m_1":
        m = gensim.models.word2vec.Word2Vec(None, size=300, window=8, min_count=3, workers=4, negative=10, sg=1)
    else:
        m = gensim.models.word2vec.Word2Vec(None, size=300, window=8, min_count=3, workers=4, negative=0, sg=1)
    print("Reading data...")
    # Load parse file
    check = [os.path.exists("pickles/relations_train.pickle"),
            os.path.exists("pickles/relations_dev.pickle"),
            os.path.exists("pickles/relations_test.pickle"),
            os.path.exists("pickles/all_relations_train.pickle"),
            os.path.exists("pickles/all_relations_dev.pickle"),
            os.path.exists("pickles/parses.pickle")]
    if all(check):
        print("Reading from cache...")
        relations_train, relations_dev, relations_test, all_relations_train, all_relations_dev, parses = readDump()
    else:
        print("Reading from source...")
        parses = json.load(open(parses_train_filepath))
        parses.update(json.load(open(parses_dev_filepath)))
        parsesTest = json.load(open(parses_test_filepath))
        (relations_train, all_relations_train) = read_file(relations_train_filepath, parses)
        (relations_dev, all_relations_dev) = read_file(relations_dev_filepath, parses)
        (relations_test, all_relations_test) = read_file(relations_test_filepath, parsesTest)
    relations = relations_train + relations_dev
    all_relations = all_relations_train + all_relations_dev
    # Substitution dictionary for class labels to integers
    label_subst = dict([(y,x) for x,y in enumerate(set([r[0][0] for r in relations]))])
    print(("Label subst", label_subst))
    print("Build vocabulary...")
    m.build_vocab(RelReader(all_relations))
    print("Reading pre-trained word vectors...")
    m.intersect_word2vec_format(googlevecs_filepath, binary=True)
    print("Training segment vectors...")
    for iter in range(1, 20):
        ## Training of word vectors
        m.alpha = 0.01/(2**iter)
        m.min_alpha = 0.01/(2**(iter+1))
        print("Vector training iter", iter, m.alpha, m.min_alpha)
        # m.train(ParseReader(parses), total_examples = m.corpus_count, epochs=m.epochs)
        m.train(RelReader(all_relations), total_examples = m.corpus_count, epochs=m.epochs)
    # dump pickles to save basic data
    dump(direct, name, m, label_subst, relations_train, relations_dev, relations_test,
         all_relations_train, all_relations_dev, parses)
    if name == "m_2":
        (input_train, output_train) = convert_relations_modified(relations_train, label_subst, m)
        (input_dev, output_dev) = convert_relations_modified(relations_dev, label_subst, m)
        (input_test, output_test) = convert_relations_modified(relations_test, label_subst, m)    
    else:
        (input_train, output_train) = convert_relations(relations_train, label_subst, m)
        (input_dev, output_dev) = convert_relations(relations_dev, label_subst, m)
        (input_test, output_test) = convert_relations(relations_test, label_subst, m)
    return input_train, output_train, input_dev, output_dev, input_test, output_test ,label_subst

def readDump():
    f = open("pickles/relations_train.pickle", "rb")
    relations_train = pickle.load(f)
    f.close()
    f = open("pickles/relations_dev.pickle", "rb")
    relations_dev = pickle.load(f)
    f.close()
    f = open("pickles/relations_test.pickle", "rb")
    relations_test = pickle.load(f)
    f.close()
    f = open("pickles/all_relations_train.pickle", "rb")
    all_relations_train = pickle.load(f)
    f.close()
    f = open("pickles/all_relations_dev.pickle", "rb")
    all_relations_dev = pickle.load(f)
    f.close()
    f = open("pickles/parses.pickle", "rb")
    parses = pickle.load(f)
    f.close()
    return relations_train, relations_dev, relations_test, all_relations_train, all_relations_dev, parses

def dump(direct, name, m, label_subst, relations_train, relations_dev, relations_test,
         all_relations_train, all_relations_dev, parses):
    if not os.path.exists("pickles"):
        os.makedirs("pickles")
    file = open("pickles/"+str(direct)+"/"+str(name)+".pickle", "wb")
    pickle.dump(m, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    if not os.path.exists("pickles/"+str(direct)+"/label_subst.pickle"):
        file_ls = open("pickles/"+str(direct)+"/label_subst.pickle", "wb")
        pickle.dump(label_subst, file_ls, protocol=pickle.HIGHEST_PROTOCOL)
        file_ls.close()
    if not os.path.exists("pickles/relations_train.pickle"):
        file_ls = open("pickles/relations_train.pickle", "wb")
        pickle.dump(relations_train, file_ls, protocol=pickle.HIGHEST_PROTOCOL)
        file_ls.close()
    if not os.path.exists("pickles/relations_dev.pickle"):
        file_ls = open("pickles/relations_dev.pickle", "wb")
        pickle.dump(relations_dev, file_ls, protocol=pickle.HIGHEST_PROTOCOL)
        file_ls.close()
    if not os.path.exists("pickles/relations_test.pickle"):
        file_ls = open("pickles/relations_test.pickle", "wb")
        pickle.dump(relations_test, file_ls, protocol=pickle.HIGHEST_PROTOCOL)
        file_ls.close()
    if not os.path.exists("pickles/all_relations_train.pickle"):
        file_ls = open("pickles/all_relations_train.pickle", "wb")
        pickle.dump(all_relations_train, file_ls, protocol=pickle.HIGHEST_PROTOCOL)
        file_ls.close()
    if not os.path.exists("pickles/all_relations_dev.pickle"):
        file_ls = open("pickles/all_relations_dev.pickle", "wb")
        pickle.dump(all_relations_dev, file_ls, protocol=pickle.HIGHEST_PROTOCOL)
        file_ls.close()
    if not os.path.exists("pickles/parses.pickle"):
        file_ls = open("pickles/parses.pickle", "wb")
        pickle.dump(parses, file_ls, protocol=pickle.HIGHEST_PROTOCOL)
        file_ls.close()
    return None

def convert_relations(relations, label_subst, m):
    inputs = []
    outputs = []
    # Convert relations: word vectors from segment tokens, aggregate to fix-form vector per segment
    for i, rel in enumerate(relations):
        senses, arg1, arg2, context = rel
        if i % 1000 == 0:
            print(("Converting relation",i))
        for sense in [senses[0]]:
            # Get tokens and weights
            tokens1 = [(token, 1./(2**depth)) if depth is not None else (token, 0.25) for token, depth in arg1]
            # Get weighted token vectors
            vecs = np.transpose([m.wv[t]*w for t,w in tokens1 if m.wv.__contains__(t)] + [m.wv[t.lower()]*w for t,w in tokens1 if not m.wv.__contains__(t) and m.wv.__contains__(t.lower())])
            if len(vecs) == 0:
                vecs = m.wv['a']*0
            vec1 = np.array(list(map(np.average, vecs)))
            vec1prod = np.array(list(map(np.prod, vecs)))
            # Get vectors for tokens in context (before arg1)
            """context1 = np.transpose([m.wv[t] for t in context[0] if t in m] + [m.wv[t.lower()] for t in context[0] if t not in m and t.lower() in m])
            if len(context1) == 0:
                context1avg = vec1*0
            else:
                context1avg = np.array(map(np.average, context1))
            """
            # Get tokens and weights
            tokens2 = [(token, 1./(2**depth)) if depth is not None else (token, 0.25) for token, depth in arg2]
            # Get weighted token vectors
            vecs = np.transpose([m.wv[t]*w for t,w in tokens2 if m.wv.__contains__(t)] + [m.wv[t.lower()]*w for t,w in tokens2 if not m.wv.__contains__(t) and m.wv.__contains__(t.lower())])
            if len(vecs) == 0:
                vecs = m.wv['a']*0
            # Get vectors for tokens in context (after arg2)
            """context2 = np.transpose([m.wv[t] for t in context[1] if t in m] + [m.wv[t.lower()] for t in context[1] if t not in m and t.lower() in m])
            if len(context2) == 0:
                context2avg = vec2*0
            else:
                context2avg = np.array(map(np.average, context2))
            """
            vec2 = np.array(list(map(np.average, vecs)))
            vec2prod = np.array(list(map(np.prod, vecs)))
            final = np.concatenate([np.add(vec1prod,vec1), np.add(vec2prod,vec2)])
            if len(final) == 2*len(m.wv['a']):
                inputs.append(final)
            else:
                print(("Warning: rel %d has length %d" % (i, len(final))))
                if len(vec1) == 0:
                    print(("arg1", arg1))
                if len(vec2) == 0:
                    print(("arg2", arg2))
                break
            outputs.append(np.array(label_subst[sense]))
    ## Theanets training from this point on
    inputs = np.array(inputs)
    inputs = inputs.astype(np.float32)
    outputs = np.array(outputs)
    outputs = outputs.astype(np.int32)
    return (inputs, outputs)

# TODO: maybe remove stop words since we really consider bag of word without sequence
# TODO: add contextual information with exponential decay by default

def convert_relations_modified(relations, label_subst, m):
    inputs = []
    outputs = []
    stop = set(stopwords.words("english"))
    # Convert relations: word vectors from segment tokens, aggregate to fix-form vector per segment
    for i, rel in enumerate(relations):
        senses, arg1, arg2, context = rel
        if i % 1000 == 0:
            print(("Converting relation",i))
        for sense in [senses[0]]:
            # Get tokens and weights for arg1
            tokens1 = [(token, 1./(2**depth)) if depth is not None else (token, 0.25) for token, depth in arg1]
            # Get weighted token vectors for arg1
            vecs = np.transpose([m.wv[t]*w for t,w in tokens1 if m.wv.__contains__(t) and t not in stop] + [m.wv[t.lower()]*w for t,w in tokens1 if not m.wv.__contains__(t) and m.wv.__contains__(t.lower()) and t.lower() not in stop])
            if len(vecs) == 0:
                vecs = m.wv['a']*0
            vec1 = np.array(list(map(np.average, vecs)))
            vec1prod = np.array(list(map(np.prod, vecs)))
            # Get tokens and weights for arg2
            tokens2 = [(token, 1./(2**depth)) if depth is not None else (token, 0.25) for token, depth in arg2]
            # Get weighted token vectors for arg2
            vecs = np.transpose([m.wv[t]*w for t,w in tokens2 if m.wv.__contains__(t) and t not in stop] + [m.wv[t.lower()]*w for t,w in tokens2 if not m.wv.__contains__(t) and m.wv.__contains__(t.lower()) and t.lower() not in stop])
            if len(vecs) == 0:
                vecs = m.wv['a']*0
            vec2 = np.array(list(map(np.average, vecs)))
            vec2prod = np.array(list(map(np.prod, vecs)))
            # compress vectors into combined vector
            final = np.concatenate([np.add(vec1prod,vec1), np.add(vec2prod,vec2)])
            if len(final) == 2*len(m.wv['a']):
                inputs.append(final)
            else:
                print(("Warning: rel %d has length %d" % (i, len(final))))
                if len(vec1) == 0:
                    print(("arg1", arg1))
                if len(vec2) == 0:
                    print(("arg2", arg2))
                break
            outputs.append(np.array(label_subst[sense]))
    # Theanets training from this point on
    inputs = np.array(inputs)
    inputs = inputs.astype(np.float32)
    outputs = np.array(outputs)
    outputs = outputs.astype(np.int32)
    return (inputs, outputs)

####################################
# read/combine PDTB data (syntactic)
####################################

def read_file(filename, parses):
    """ Read relation data from JSON """
    relations = []
    all_relations = []
    for row in open(filename):
        rel = json.loads(row)
        doc = parses[rel['DocID']]
        arg1 = get_token_depths(rel['Arg1'], doc)
        arg2 = get_token_depths(rel['Arg2'], doc)
        context = None #get_context(rel, doc, context_size=1)
        # Use for word vector training
        all_relations.append((rel['Sense'], arg1, arg2))
        # Use for prediction (implicit relations only)
        if rel['Type'] in ['Implicit', 'EntRel']:#, 'AltLex']:
            relations.append((rel['Sense'], arg1, arg2, context))
    return (relations, all_relations)

class RelReader(object):
    """ Iterator for reading relation data """
    def __init__(self, segs, docvec=False):
        self.segs = segs
        self.docvec = docvec
    def __iter__(self):
        for file, i in zip(self.segs, list(range(len(self.segs)))):
            for sub in [0, 1]:
                text = [token for token, _ in self.segs[i][sub+1]]
                if self.docvec:
                    yield gensim.models.doc2vec.TaggedDocument(words=text, tags=i*2+sub)#[doclab])
                else:
                    yield text

class ParseReader(object):
    """ Iterator for reading parse data """
    def __init__(self, parses, docvec=False, offset=0):
        self.parses = parses
        self.docvec = docvec
        self.offset = offset
    def __iter__(self):
        i = -1
        for doc in self.parses:
            ####print "      Reading", doc
            for sent_i, sent in enumerate(self.parses[doc]['sentences']):
                tokens = [w for w, _ in sent['words']]
                i += 1
                if self.docvec:
                    yield gensim.models.doc2vec.TaggedDocument(words=tokens, tags=self.offset+i)#["%s_%d" % (doc, sent_i)])
                else:
                    yield tokens

def build_tree(dependencies):
    """ Build tree structure from dependency list """
    tree = collections.defaultdict(lambda: [])
    for rel, parent, child in dependencies:
        tree[parent].append(child)
    return tree

def traverse(tree, node='ROOT-0', depth=0):
    """ Traverse dependency tree, calculate token depths """
    tokens = []
    for child in tree[node]:
        tokens.append((child, depth))
        tokens += traverse(tree, child, depth+1)
    return tokens

def get_token_depths(arg, doc):
    """ Wrapper for token depth calculation """
    tokens = []
    depths = {}
    for _, _, _, sent_i, token_i in arg['TokenList']:
        if sent_i not in depths:
            depths[sent_i] = dict(traverse(build_tree(doc['sentences'][sent_i]['dependencies'])))
        token, _ = doc['sentences'][sent_i]['words'][token_i]
        try:
            tokens.append((token, depths[sent_i][token+'-'+str(token_i+1)]))
        except KeyError:
            tokens.append((token, None))
    return tokens

####################################
# context of arguments
####################################

def get_context(rel, doc, context_size=2):
    """ Get tokens from context sentences of arguments """
    pretext, posttext = [], []
    for context_i in reversed(list(range(context_size+1))):
        _, _, _, sent_i, _ = rel['Arg1']['TokenList'][0]
        for token_i, token in enumerate(doc['sentences'][sent_i-context_i]['words']):
            token, _ = token
            if context_i == 0 and token_i >= rel['Arg1']['TokenList'][0][-1]:
                break
            pretext.append(token)
    for context_i in range(context_size+1):
        _, _, _, sent_i, _ = rel['Arg2']['TokenList'][-1]
        try:
            for token_i, token in enumerate(doc['sentences'][sent_i+context_i]['words']):
                token, _ = token
                if context_i == 0 and token_i <= rel['Arg2']['TokenList'][-1][-1]:
                    continue
                posttext.append(token)
        except IndexError:
            pass
    return (pretext, posttext)
