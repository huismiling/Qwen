
import json

pipa_trainset = json.load(open("piqa_trainset.txt", "r"))
arc_trainset = json.load(open("arc_trainset.txt", "r"))
hellaswag_trainset = json.load(open("hellaswag_trainset.txt", "r"))

total_trainset = pipa_trainset+arc_trainset+hellaswag_trainset
print(len(total_trainset))

# json.dump(total_trainset, open("total_trainset.txt", "w"), indent=4)

pipa_devset = json.load(open("piqa_devset.txt", "r"))
arc_devset = json.load(open("arc_devset.txt", "r"))
hellaswag_devset = json.load(open("hellaswag_devset.txt", "r"))

total_devset = pipa_devset+arc_devset+hellaswag_devset
print(len(total_devset))
# json.dump(total_devset, open("total_devset.txt", "w"), indent=4)
