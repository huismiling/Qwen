
import json

from sklearn.model_selection import train_test_split

out_format = json.loads(r'  {   "id": "identity_0",    "conversations": [       {          "from": "user",        "value": "你好"      },        {          "from": "assistant",           "value": "我是一个语言模型，我叫通义千问。"      }       ]       }' )

id_list = []
'''{"ind": 24, 
    "activity_label": "Roof shingle removal", 
    "ctx_a": "A man is sitting on a roof.", 
    "ctx_b": "he", 
    "ctx": "A man is sitting on a roof. he", 
    "split": "val", 
    "split_type": "indomain", 
    "label": 3, 
    "endings": ["is using wrap to wrap a pair of skis.", 
                "is ripping level tiles off.", 
                "is holding a rubik's cube.", 
                "starts pulling up roofing on a roof."], 
    "source_id": "activitynet~v_-JhWjGDPHMY"}
'''
import re
def preprocess_text(text):
    text = text.strip()
    # NOTE: Brackets are artifacts of the WikiHow dataset portion of HellaSwag.
    text = text.replace(" [title]", ". ")
    text = re.sub("\\[.*?\\]", "", text)
    text = text.replace("  ", " ")
    return text

def process_item(text):
    text = json.loads(text)
    out_format["id"] = f"hellaswag_{text['source_id']}_{text['ind']}"
    assert out_format["id"] not in id_list
    id_list.append(out_format["id"])

    activity_label = text["activity_label"]
    ctx = text["ctx"]
    out_format["conversations"][0]["value"] = preprocess_text(f"{activity_label}: {ctx}")
    label = int(text["label"])
    out_format["conversations"][1]["value"] = preprocess_text(text["endings"][label])
    out_str = json.dumps(out_format)
    return out_str


train_texts = open("hellaswag/hellaswag_train.jsonl").readlines()

train_out=[]
for itext in train_texts:
    out_str = process_item(itext)
    train_out.append(out_str)

with open("hellaswag_trainset.txt", "w") as fout:
    fout.write('[\n')
    fout.write(', \n'.join(train_out))
    fout.write('\n]\n')

dev_texts = open("hellaswag/hellaswag_val.jsonl").readlines()

dev_out=[]
for itext in dev_texts:
    out_str = process_item(itext)
    dev_out.append(out_str)

with open("hellaswag_devset.txt", "w") as fout:
    fout.write('[\n')
    fout.write(', \n'.join(dev_out))
    fout.write('\n]\n')
