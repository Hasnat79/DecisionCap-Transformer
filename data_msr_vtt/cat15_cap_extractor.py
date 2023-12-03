# for clip text embeddings : https://github.com/huggingface/transformers/issues/21465#issuecomment-1419080756
# for clip image features : https://huggingface.co/docs/transformers/model_doc/clip
    # search: get_image_features

import os
import json 
import numpy as np

cat_15_video_cap_data = {}
msr_trainval_dataset_path = "./train_val_annotation/fixed_train_val_videodatainfo.json"
with open(msr_trainval_dataset_path,'r') as f: 
    data = json.load(f)

#all cat 15 video dictionaries filtered
cat15_video_d = [video_d for video_d in data['videos'] if video_d['category'] == 15] #total: 112
# cat15_video_d_train = [video_d for video_d in cat15_video_d if video_d['split'] == 'train'] #totaal: 104
# cat15_video_d_val = [video_d for video_d in cat15_video_d if video_d['split'] == "validate"]# total: 8

#saving the cat 15 video ids, 
cat15_video_ids = [video_d['video_id'] for video_d in cat15_video_d]

# extract captions for cat15 videos,total: 112*20(20 cap/ video) = 2240
cat15_cap_d = [sent_d for sent_d in data['sentences'] if sent_d['video_id'] in cat15_video_ids]

# print(len(cat15_cap_d)) #2240 : 112*20(each video-->20 caption)

#extracting {video_id: [caption, captions,...]}
for cap_info in cat15_cap_d: 
    # print(cap_info)
    video_id = cap_info['video_id']
    caption = cap_info['caption']
    # print(video_id)
    # print(caption)

    if video_id in cat15_video_ids:
        
        if video_id not in cat_15_video_cap_data:
            cat_15_video_cap_data[video_id] = []
        
        cat_15_video_cap_data[video_id].append(caption)
    # break
# print(len(cat_15_video_cap_data['video203']))

with open("./msr_vtt_category_15_videoId_cap_data.json", 'w') as f: 
    json.dump(cat_15_video_cap_data,f,indent=4)



