
import json

from sklearn.model_selection import train_test_split

out_format = json.loads(r'  {   "id": "identity_0",    "conversations": [       {          "from": "user",        "value": "你好"      },        {          "from": "assistant",           "value": "我是一个语言模型，我叫通义千问。"      }       ]       }' )

'''{
  "id": "MCAS_2000_4_6",
  "question": {
    "stem": "Which technology was developed most recently?",
    "choices": [
      {
        "text": "cellular telephone",
        "label": "A"
      },
      {
        "text": "television",
        "label": "B"
      },
      {
        "text": "refrigerator",
        "label": "C"
      },
      {
        "text": "airplane",
        "label": "D"
      }
    ]
  },
  "answerKey": "A"
}'''
id_list = []
def process_item(text):
    text = json.loads(text)
    out_format["id"] = text["id"]
    assert out_format["id"] not in id_list
    id_list.append(out_format["id"])

    question = text["question"]
    labels = [itc['label'] for itc in question["choices"]]
    ansIdx = labels.index(text["answerKey"])
    out_format["conversations"][0]["value"] = "Question: " + question["stem"] + "\nAnswer:"
    out_format["conversations"][1]["value"] = question["choices"][ansIdx]['text']
    out_str = json.dumps(out_format)
    return out_str


train_texts = open("ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Train.jsonl").readlines()

train_out=[]
for itext in train_texts:
    out_str = process_item(itext)
    train_out.append(out_str)

with open("arc_trainset.txt", "w") as fout:
    fout.write('[\n')
    fout.write(', \n'.join(train_out))
    fout.write('\n]\n')

dev_texts = open("ARC-V1-Feb2018-2/ARC-Challenge/ARC-Challenge-Dev.jsonl").readlines()

dev_out=[]
for itext in dev_texts:
    out_str = process_item(itext)
    dev_out.append(out_str)

with open("arc_devset.txt", "w") as fout:
    fout.write('[\n')
    fout.write(', \n'.join(dev_out))
    fout.write('\n]\n')
