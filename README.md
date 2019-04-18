# Further work on Frankfurt Shallow Discourse Parser

This project is a fork of the official Frankfurt Shallow Discourse Parser and involves building/experimenting on implicit sense labelling components. Development is underway.

To set up the pre-commit hook that will keep python dependencies up-to-date and automatically download pre-trained google word vectors, please run the following:

```shell
$ ./init.sh
```

## The Frankfurt Shallow Discourse Parser

Developed at the Applied Computational Linguistics Lab (ACoLi), Goethe University Frankfurt am Main, Germany.

This repository hosts the shallow discourse parser described in: [Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling](http://aclweb.org/anthology/K16-2005)

Niko Schenk, Christian Chiarcos, Samuel Rönnqvist, Kathrin Donandt, Evgeny A. Stepanov and Giuseppe Riccardi. "Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling". In *Proceedings of the Twentieth Conference on Computational Natural Language Learning - Shared Task, CoNLL 2016*. 2016.

```
@inproceedings{schenk-EtAl:2016:CoNLL-STSDP,
  author    = {Niko Schenk, Christian Chiarcos, Samuel Rönnqvist, Kathrin Donandt, Evgeny A. Stepanov,  Giuseppe Riccardi},
  title     = {{Do We Really Need All Those Rich Linguistic Features? A Neural Network-Based Approach to Implicit Sense Labeling}},
  booktitle = {Proceedings of the Twentieth Conference on Computational Natural Language Learning - Shared Task, CoNLL 2016},
  month     = {August},
  year      = {2016},
  address   = {Berlin, Germany},
  publisher = {Association for Computational Linguistics}
}
```

## Data Requirements

Please copy the following data into the data/ directory:

For English:

- Penn Discourse TreeBank (PDTB) 2.0, a 1-million-word Wall Street Journal corpus; there is a train directory and a dev directory (named en-01-12-16-train/ and en-01-12-16-dev/, respectively). They have to include the parses and relations files (normally called pdtb-parses.json/pdtb-relations.json or parses.json/relations.json) (http://www.cs.brandeis.edu/~clp/conll16st/rules.html). Simply, place these two folders into the data/ directory.
- GoogleNews-vectors-negative300 (https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?pref=2&pli=1)
