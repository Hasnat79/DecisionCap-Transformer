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


data = load(file)
# write(out_file,data)
# cat = [i for i in range(20)]

#  "category": 3,

# print(len(dataset['videos']))

category_distribution = {}
data_ = data['videos']
# Count the occurrences of each category
for video in data_:
    category = int(video["category"])
    if category in category_distribution:
        category_distribution[category] += 1
    else:
        category_distribution[category] = 1

# Print the distribution
for category, count in category_distribution.items():
    print(f"Category {category}: {count} videos")