import json
import pandas as pd

senses = []
datasets = ["blind-test", "dev", "test", "train"]
for dataset in datasets:
    with open("data/en.%s/relations.json" % dataset, mode = "r", encoding = "utf-8") as f:
        temp = []
        for line in f:
            data = json.loads(line)
            sense = data["Sense"]
            type = data["Type"]
            if type in ["EntRel", "Implicit"]:
                for s in sense:
                    senses.append((s, dataset))
    # senses[dataset] = temp


senses = pd.DataFrame(senses)
senses.columns = ["sense", "dataset"]
temp = senses.groupby(by = ["dataset","sense"]).count()
temp = senses
def fix_types(x):
    try:
        return int(x)
    except ValueError:
        return 0

temp["count"] = temp.groupby(by = ["dataset","sense"])["sense"].transform("count")
temp.drop_duplicates().reset_index()#.groupby("sense").first().reset_index()
senses = temp.pivot_table(index = "sense", columns = ["dataset"], values = "count").reset_index()
for dataset in datasets:
    senses[dataset] = senses[dataset].apply(fix_types)
print(senses.to_latex())
