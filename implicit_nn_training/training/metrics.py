import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from collections import defaultdict

class Metrics():
    
    """Service class that manages metrics for all dataset on one single training instance"""
    def __init__(self):
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
            print("Dataset %s already registered!\nThis causes the metrics to be reset." % name)
        
        for metric in self.metrics.keys():
            # create a new empty list of metrics for the dataset
            self.metrics[metric][name] = []

    def update_metrics(self, y_true, y_pred, name):
        self.metrics["accs"][name].append(accuracy_score(y_true, y_pred, normalize= True))
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

    def get_averages_ordered_by(self, names):
        r = self.get_averages()
        return [r["accs"][n] for n in names], [r["recs"][n] for n in names], [r["precs"][n] for n in names], [r["f1s"][n] for n in names]