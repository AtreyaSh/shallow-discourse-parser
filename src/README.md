# Table of Contents

* [Word-Embedding Models](## Word-Embedding Models)
* [Usage](## Usage)
* [Results](## Results)

## Word-Embedding Models

Since our results showed particular improvements upon refining word embeddings, we present here the code mainly for our word embeddings modifications. We therefore still use the baseline theanets neural architecture from Schenk et al. 2016.

### Model Parameters

1. **negative**: integer with range 0 or above; refers to the number of negative samples used per positive-
sampled word for training

2. **sg**: 1 if skip-gram architecture is used, 0 if common-bag-of-words (CBOW) architecture is used

3. **hs**: 1 if hierarchical softmax is used, 0 if negative sampling is used

4. **final dimension**: final dimensionality after concatenating aggregated word vectors

5. **aggregation**: 0 for baseline approach using averaging and products, 1 for replacing baseline products with variance, 2 for removing baseline products

6. **context**: 1 if contexts of implicit arguments are used in aggregation, 0 if contexts are ignored

7. **stop**: 1 if stopwords are excluded from aggregation, 0 if stopwords are untouched during aggregation

### Tested Models

Below are the word-embedding models that we tested with the corresponding parameters. M<sub>0</sub> corresponds to the baseline embedding that we obtained from the Frankfurt Shallow Discourse Parser. Therefore, this is baseline embedding for us to compare against. All other models are therefore computed by varying relevant parameters.

| Model | negative | sg | hs | final dimension | aggregation | context | stop |
|---|---|---|---|---|
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
                 'grid' and 'combination', defaults to 'single'
  --name NAME    Word-embedding model to be used such as 'm_0', 'm_1', 'm_2'
                 ... 'm_11', defaults to 'm_1'
  --debug        Enter debugging mode
```

We implemented 3 modes of searching for neural network hyperparameters given a word-embedding model.

1. The "single" mode simply uses a user-defined set of parameters, which is essentially one run. These default to the parameters from our best run. The user can redefine these parameters within the source-code in `train.py`.

2. The "grid" mode defines a set of possible values for hyperparameters and essentially conducts a grid-wise search to find the best combination. A total of 72 models will be tested using this mode.

3. The "combination" mode tests each word embedding with a given set of parameters repeatedly instead of only once, in order to get the best results out of otherwise randomly assigned starting weights. This mode has not been heavily tested on our end, so do proceed with caution.

Note: An example of running this file:

```shell
python3 train.py --mode grid --name m_11
```

### Model/Results Logging

All models that are run will be logged as pickles in the `pickles/` directory with the given date/time and model number for easy identification. A csv file with results of each model will also be produced for post-run analysis. Other files such as the input/output aggregated embeddings and word embedding model will also be written to pickle files for easy re-testing at a later time.

Our best model has been saved in the `pickles/` directory. These include the input and output argument embeddings with the prefix `inout*`, the best word embedding model with the prefix `m*` and the best neural-network with the prefix `neuralnetwork*`. We also include the label substitution dictionary, which essentially translates PDTB sense into integer classes.

## Results

The results of running the 12 models above, via a `grid` search producing 72 neural-networks per word-embedding model, have been visualized below. The graphs below show precision-recall curves with an approximate contour line for a 0.4 F1-score. We can observe the phenomenon of overfitting in cases where the train/dev clusters lie significantly further away than test clusters.

<img src="/img/models.png" width="800">

The best dev/test F1-scores per word-embedding model have been tabulated below.

| Model | best dev-F1 | best test-F1 |
|---|---|---|---|---|
| M<sub>0</sub> | *0.4319465* | *0.3593190* |
| M<sub>1</sub> | **0.4395718** | **0.3639238** |
| M<sub>2</sub> | 0.4256447 | 0.3543410 |
| M<sub>3</sub> | 0.4128714 | 0.3309730 |
| M<sub>4</sub> | 0.4291707 | 0.3501314 |
| M<sub>5</sub> | 0.4379196 | 0.3570653 |
| M<sub>6</sub> | 0.3429012 | 0.2886952 |
| M<sub>7</sub> | 0.3800511 | 0.3134248 |
| M<sub>8</sub> | 0.3998472 | 0.3419556 |
| M<sub>9</sub> | 0.3737086 | 0.3317736 |
| M<sub>10</sub> | 0.4321273 | 0.3514591 |
| M<sub>11</sub> | **0.4396398** | 0.3590001 |

### Observations

1. The models highlighted in bold exceeded the baseline model M<sub>0</sub> in the dev and/or test dataset. We can observe that the better performing models both incorporated negative sampling into their workflow, while the baseline model did not. We can therefore conclude the importance of negative sampling in word embedding refinement.

2. We can also observe that M<sub>3</sub> performed significantly poorly compared to other models. M<sub>3</sub> excluded stop words from argument vector aggregation. We can conclude that stop-words are of importance in aggregating word vectors to argument vectors.

3. We can also observe that models M<sub>6</sub> and  M<sub>7</sub> perform very poorly. These models were designed to be hybrid models which incorporated numerous components from other models. We can conclude that the combination of various individiually well-performing parameters, such as aggregation and context-dependence, does not necessarily lead to an overall improvement in performance. In the cases of M<sub>6</sub> and  M<sub>7</sub>, such hybrid combinations actually led to a drop in performance. This stresses the importance of doing grid-based (step-by-step) searches of word-embedding parameters instead of combined parameter changes.
