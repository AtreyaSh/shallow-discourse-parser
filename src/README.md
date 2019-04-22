## Modifications to the Frankfurt Shallow Discourse Parser (Schenk et al. 2016)

Our modifications performed optimally in the word-embeddings component of the Frankfurt Shallow Discourse Parser. Our best model exceeded the baseline model by an F1-score of 0.46%. More detailed information on our models/runs can be found below.

## Table of Contents

* [Word-Embedding Models](#Word-Embedding-Models)
* [Usage](#Usage)
* [Results](#Results)
* [Further Work](#Further-Work)

## Word-Embedding Models

Since our results showed particular improvements upon refining word embeddings, we present here the code mainly for our word embeddings modifications. We therefore still use the baseline theanets neural architecture from Schenk et al. 2016.

### Model Parameters

1. **negative**: integer with range 0 or above; refers to the number of negative samples used per positive-sampled word for training

2. **sg**: 1 if skip-gram architecture is used, 0 if common-bag-of-words (CBOW) architecture is used

3. **hs**: 1 if hierarchical softmax is used, 0 if negative sampling is used

4. **final dimension**: final dimensionality after concatenating aggregated word vectors

5. **aggregation**: 0 for baseline approach using averaging and products, 1 for replacing baseline products with variance, 2 for removing baseline products

6. **context**: 1 if contexts of implicit arguments are used in aggregation, 0 if contexts are ignored

7. **stop**: 1 if stopwords are excluded from aggregation, 0 if stopwords are untouched during aggregation

### Tested Models

Below are the word-embedding models that we tested with the corresponding parameters. M<sub>0</sub> corresponds to the baseline embedding that we obtained from the Frankfurt Shallow Discourse Parser. Therefore, this represents the baseline embedding for us to compare against. All other models are therefore computed by varying relevant parameters.

| Model | negative | sg | hs | final dimension | aggregation | context | stop |
|---|---|---|---|---|---|---|---|
| M<sub>0</sub> | 0 | 1 | 1 | 600 | 0 | 0 | 0 |
| M<sub>1</sub> | 10 | 1 | 0 | 600 | 0 | 0 | 0 |
| M<sub>2</sub> | 0 | 1 | 1 | 600 | 1 | 0 | 0 |
| M<sub>3</sub> | 0 | 1 | 1 | 600 | 0 | 0 | 1 |
| M<sub>4</sub> | 0 | 1 | 1 | 1200 | 1 | 0 | 0 |
| M<sub>5</sub> | 0 | 1 | 1 | 600 | 0 | 1 | 0 |
| M<sub>6</sub> | 0 | 1 | 1 | 600 | 2 | 1 | 0 |
| M<sub>7</sub> | 0 | 1 | 1 | 1200 | 1 | 1 | 0 |
| M<sub>8</sub> | 10 | 0 | 0 | 1200 | 1 | 1 | 0 |
| M<sub>9</sub> | 5 | 0 | 0 | 1200 | 1 | 1 | 0 |
| M<sub>10</sub> | 0 | 1 | 1 | 600 | 1 | 1 | 0 |
| M<sub>11</sub> | 20 | 1 | 0 | 600 | 0 | 0 | 0 |

## Usage

### Running train.py

For running models, we created a file with dedicated functions in `train.py`. The usage instructions for this file are shown below.

```
usage: train.py [-h] [--train TRAIN] [--dev DEV] [--test TEST] [--emb EMB]
                [--mode MODE] [--name NAME] [--debug]

optional arguments:
  -h, --help     show this help message and exit
  --train TRAIN  Path to train data folder, defaults to data/en.train/
  --dev DEV      Path to development data folder, defaults to data/en.dev/
  --test TEST    Path to test data folder, defaults to data/en.test/
  --emb EMB      Path to pretrained google embeddings, defaults to
                 data/GoogleNews-vectors-negative300.bin
  --mode MODE    Type of NN hyperparameter search, possibilities are 'single',
                 'grid', defaults to 'single'
  --name NAME    Word-embedding model to be used such as 'm_0', 'm_1', 'm_2'
                 ... 'm_11', defaults to 'm_1'
  --debug        Enter debugging mode
```

We implemented 2 modes of searching for neural network hyperparameters given a word-embedding model.

1. The "single" mode simply uses a user-defined set of parameters, which is essentially one run. These default to the parameters from our best run. The user can redefine these parameters within the source-code in `train.py`.

2. The "grid" mode defines a set of possible values for hyperparameters and essentially conducts a grid-wise search to find the best combination. A total of 72 models will be tested using this mode.

Note: An example of running this file:

```shell
$ python3 train.py --mode grid --name m_11
```

### Running compare_combination.py

After running `train.py`, one can compare different models to find which ones had the best performance. In order to compare two different models (eg. a modified model vs. baseline model), one can use `compare_combination.py`. This dedicated scripts runs the baseline and a possible optimal model repeatedly and logs their accuracies. These accuracies can then be used to detect statistically significant changes.

```
usage: compare_combination.py [-h] [--iterations ITERATIONS] -n1 NETWORK1 -n2
                              NETWORK2

optional arguments:
  -h, --help            show this help message and exit
  --iterations ITERATIONS
                        number of iterations for each network, defaults to 20

required named arguments:
  -n1 NETWORK1, --network1 NETWORK1
                        path to network 1
  -n2 NETWORK2, --network2 NETWORK2
                        path to network 2
```

Note: Here is an example run of this script:

```shell
$ python3 compare_combination.py -n1 ./pickles/2019_04_11_21_02_39_m_0/neuralnetwork_37.pickle \
-n2 ./pickles/2019_04_11_21_03_37_m_1/neuralnetwork_59.pickle
```

### Model/Results Logging

All executed models will be logged as pickles in the `pickles/` directory with the given date/time and model number for easy identification. A csv file with results of each model will also be produced for post-run analysis. Other files such as the input/output aggregated embeddings and word embedding model will also be written to pickle files for easy re-testing at a later time.

Our best model has been saved in the `pickles/*_best` directory. These include the input and output argument embeddings with the prefix `inout*`, the best word embedding model with the prefix `m*` and the best neural-network with the prefix `neuralnetwork*`. We also include the label substitution dictionary, which essentially translates PDTB discourse senses into unique integer classes, and the corresponding run-log in `Results.csv`.

## Results

The results of running the 12 models above, via a `grid` search producing 72 neural-networks per word-embedding model, have been visualized below. The graphs below show precision-recall curves with an approximate contour line for a 0.4 F1-score. We can observe the phenomenon of overfitting in cases where the train/dev clusters lie significantly far away from test clusters.

<img src="/img/models.png" width="800">

The best dev/test F1-scores per word-embedding model have been tabulated below.

| Model | negative | sg | hs | final dimension | aggregation | context | stop | best dev-F1 | best test-F1 |
|:---|:---|:---|:---|:---|:---|:---|:---|:---|:---|
| M<sub>0</sub> | 0 | 1 | 1 | 600 | 0 | 0 | 0 | *0.4319465* | *0.3593190* |
| **M<sub>1</sub>** | 10 | 1 | 0 | 600 | 0 | 0 | 0 | **0.4395718** | **0.3639238**\*\*\* |
| M<sub>2</sub> | 0 | 1 | 1 | 600 | 1 | 0 | 0 | 0.4256447 | 0.3543410 |
| M<sub>3</sub> | 0 | 1 | 1 | 600 | 0 | 0 | 1 | 0.4128714 | 0.3309730 |
| M<sub>4</sub> | 0 | 1 | 1 | 1200 | 1 | 0 | 0 | 0.4291707 | 0.3501314 |
| M<sub>5</sub> | 0 | 1 | 1 | 600 | 0 | 1 | 0 | 0.4379196 | 0.3570653 |
| M<sub>6</sub> | 0 | 1 | 1 | 600 | 2 | 1 | 0 | 0.3429012 | 0.2886952 |
| M<sub>7</sub> | 0 | 1 | 1 | 1200 | 1 | 1 | 0 | 0.3800511 | 0.3134248 |
| M<sub>8</sub> | 10 | 0 | 0 | 1200 | 1 | 1 | 0 | 0.3998472 | 0.3419556 |
| M<sub>9</sub> | 5 | 0 | 0 | 1200 | 1 | 1 | 0 | 0.3737086 | 0.3317736 |
| M<sub>10</sub> | 0 | 1 | 1 | 600 | 1 | 1 | 0 | 0.4321273 | 0.3514591 |
| **M<sub>11</sub>** | 20 | 1 | 0 | 600 | 0 | 0 | 0 | **0.4396398** | 0.3590001 |

Note: \*\*\*\{t-test; p < 0.001\}

### Observations

1. The models M<sub>1</sub> and M<sub>11</sub>, highlighted in bold, exceeded the baseline model M<sub>0</sub> in the dev and/or test dataset. We can observe that the better performing models both incorporated negative sampling into their workflow, while the baseline model did not. We can therefore conclude the importance of negative sampling in word embedding refinement.

2. We repeated the runs to compute 40 overall test F1-scores for models M<sub>0</sub> and M<sub>1</sub> using `compare_combination.py`. Based on the variations of test F1-scores, we concluded, using the Welch's t-test, that the mean test F1-score of M<sub>1</sub> is significantly larger than the mean test F1-score of M<sub>0</sub>.

3. We can observe that M<sub>3</sub> performed significantly poorly compared to other models. M<sub>3</sub> excluded stop words from argument vector aggregation. We can conclude that stop-words are of importance in aggregating word vectors to argument vectors.

4. We can observe that models M<sub>6</sub> and M<sub>7</sub> perform very poorly. These models were designed to be hybrid models which incorporated numerous components from other models. We can conclude that the combination of various individiually well-performing parameters, such as aggregation and context-dependence, does not necessarily lead to an overall improvement in performance. In the cases of M<sub>6</sub> and  M<sub>7</sub>, such hybrid combinations actually led to a drop in performance. This stresses the importance of doing grid-based (step-by-step) searches of word-embedding parameters instead of combined parameter changes.
