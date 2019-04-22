import numpy as np
import matplotlib
matplotlib.use("TkAgg") # mac fix
import matplotlib.pyplot as plt
import csv, sys
from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from ast import literal_eval
# import some data to play with
"""
Tiny plotting scrapt that receives a csv containing y_true and y_pred
and plots them into a confusion matrix to root"""

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    classes = label_subst
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
        #    title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor", fontsize = 12)
    plt.setp(ax.get_yticklabels(), fontsize = 12)

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    
    return ax

np.set_printoptions(precision=5)
# data looks like this:
# name;y_true;y_pred;label_subst
counter = 0
with open(sys.argv[1]) as f:
    for line in f:
        name, y_true, y_pred, label_subst = line.split(";")
        y_true = literal_eval(y_true)
        y_pred = literal_eval(y_pred)
        freqs = np.bincount(y_true)
        label_subst = literal_eval(label_subst)
        ax = plot_confusion_matrix(y_true, y_pred, classes = label_subst, normalize=True,
                        title = "Confusion matrix for %s data" % name)
        ax.get_figure().savefig("confmat%s-%s.png" % (name, counter))
        counter += 1
            