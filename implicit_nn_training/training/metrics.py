import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn.utils.multiclass import unique_labels

from collections import defaultdict
class Metrics():
    """Service class that manages metrics for all dataset on one single training instance"""
    def __init__(self):
        self.datasets = [] # List to ensure same order of items TODO implement
        self.metrics = {
            "accs" : {},
            "recs" : {},
            "precs" : {},
            "f1s" : {}
        }


    def register_datasets(self, datasets):
        for name in datasets:
            self.register_dataset(name)
        
    def register_dataset(self, name):
        """Put a dataset """
        if any(name in metric.keys() for metric in self.metrics.values()):
            print("Dataset %s already registered!\nThis causes the metrics to be reset." % dataset)
        
        for metric in self.metrics.keys():
            # create a new empty list of metrics for the dataset
            self.metrics[metric][name] = []

    def update_metrics(self, model, dataset, name):
        y_true = np.argmax(dataset[1], axis = 1)
        y_pred = model.predict(dataset[0])
        y_pred = np.argmax(y_pred, axis=1)
        print(name)
        print(y_true)
        print(y_pred)
        print("___")
        self.metrics["accs"][name].append(accuracy_score(y_true, y_pred))
        self.metrics["recs"][name].append(recall_score(y_true, y_pred, average="weighted"))
        self.metrics["precs"][name].append(precision_score(y_true, y_pred, average="weighted"))
        self.metrics["f1s"][name].append(f1_score(y_true, y_pred, average="weighted"))

    def get_averages(self):
        """Iterates over existing dataset metrics, computes their averages and returns them
        in a flattened dictionary"""
        result = defaultdict(dict)
        for metric in self.metrics.keys():
            for dataset in self.metrics[metric].keys():
                result[metric][dataset] = np.average(self.metrics[metric][dataset])
        return result


    def get_final_as_tuple(self):
        r = self.get_averages()
        return tuple(r["accs"].values()), tuple(r["recs"]), tuple(r["precs"]), tuple(r["f1s"])


    def get_averages_ordered_by(self, names):
        r = self.get_averages()
        return (tuple([r["accs"][n] for n in names]), tuple([r["recs"][n] for n in names]), 
                tuple([r["precs"][n] for n in names]), tuple([r["f1s"][n] for n in names]))