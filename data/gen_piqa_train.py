
import json

from sklearn.model_selection import train_test_split

out_format = json.loads(r'  {   "id": "identity_0",    "conversations": [       {          "from": "user",        "value": "你好"      },        {          "from": "assistant",           "value": "我是一个语言模型，我叫通义千问。"      }       ]       }' )

# {"id": "c36c629e-12e9-43cc-8936-e1a96d869ab0", 
#     "goal": "How do I ready a guinea pig cage for it's new occupants?", 
#     "sol1": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped paper strips, you will also need to supply it with a water bottle and a food dish.", 
#     "sol2": "Provide the guinea pig with a cage full of a few inches of bedding made of ripped jeans material, you will also need to supply it with a water bottle and a food dish."
# }
id_list=[]
def process_item(text, label):
    text = json.loads(text)
    label = int(label)
    out_format["id"] = text["id"]
    assert out_format["id"] not in id_list
    id_list.append(out_format["id"])
    out_format["conversations"][0]["value"] = "Question: " + text["goal"] + "\nAnswer:" 
    out_format["conversations"][1]["value"] = text["sol1"] if label==0 else text["sol2"]
    out_str = json.dumps(out_format)
    return out_str


train_texts = open("physicaliqa-train-dev/train.jsonl").readlines()
train_labels = open("physicaliqa-train-dev/train-labels.lst").readlines()

train_out=[]
for itext, ilabel in zip(train_texts, train_labels):
    out_str = process_item(itext, ilabel)
    train_out.append(out_str)

with open("piqa_trainset.txt", "w") as fout:
    fout.write('[\n')
    fout.write(', \n'.join(train_out))
    fout.write('\n]\n')

train_texts = open("physicaliqa-train-dev/dev.jsonl").readlines()
train_labels = open("physicaliqa-train-dev/dev-labels.lst").readlines()

train_out=[]
for itext, ilabel in zip(train_texts, train_labels):
    out_str = process_item(itext, ilabel)
    train_out.append(out_str)

with open("piqa_devset.txt", "w") as fout:
    fout.write('[\n')
    fout.write(', \n'.join(train_out))
    fout.write('\n]\n')
