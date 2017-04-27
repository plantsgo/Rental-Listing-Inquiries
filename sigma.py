#encoding=utf8

import json
import pandas as pd

def json2csv(data_file):
    with open(data_file) as f:
        train = f.read().encode("utf8")
    dic = {}
    train = json.loads(train)
    keys = train.keys()
    id_list = train[keys[0]].keys()
    dic["id"] = id_list

    data = pd.DataFrame(dic)
    for key in keys:
        word_list = []
        for i in id_list:
            word = train[key][i]
            try:
                word = word.replace("\r", "")
            except:
                pass
            word_list.append(word)
        data[key] = word_list

    return data

def get_data():
    train_data=json2csv("train.json")
    test_data=json2csv("test.json")
    return train_data,test_data

if __name__=="__main__":
    train_data, test_data=get_data()
    train_data.to_csv("train.csv",index=None,encoding="utf8")
    test_data.to_csv("test.csv",index=None,encoding="utf8")
