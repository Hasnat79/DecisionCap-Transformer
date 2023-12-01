import json
import os
def load(file):
    with open(file,'r') as f:
        data = json.load(f)

    return data

def write(file,data):
    with open(file,'w') as f:
        json.dump(data,f,indent=4)

file = "./train_val_videodatainfo.json"
out_file = "./fixed_train_val_videodatainfo.json"

data = load(file)
write(out_file,data)