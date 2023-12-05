# for clip text embeddings : https://github.com/huggingface/transformers/issues/21465#issuecomment-1419080756
# for clip image features : https://huggingface.co/docs/transformers/model_doc/clip
    # search: get_image_features

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import json
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, TFCLIPModel
from nltk.tokenize import word_tokenize
from pathlib import Path
import torch
from tqdm import tqdm
from gensim.models import Word2Vec

from cider_metric.cider import cal_cider_score



def save_json(file, data):
    with open(file,'w') as f: 
        json.dump(data,f, indent=4)

def load_json(file):
    with open(file,'r') as f:
        return json.load(f)

# ----------------------------------------------------
def get_video_id_to_tokenized_caps(cat15_video_cap_data):
    video_id_tokenized_cap = {}
    data = cat15_video_cap_data


    for k,v in data.items():
        all_caps  = [caption for caption in v]
        tokenized_caps_ = [ word_tokenize(cap.lower()) for cap in all_caps]
        video_id_tokenized_cap[k] = tokenized_caps_


    # print(video_id_tokenized_cap)
    # print(len(video_id_tokenized_cap['video1919']))
    save_json("./video_id_tokenized_cap.json",video_id_tokenized_cap)
def create_action_vector (tokenizer, word):
    output = tokenizer(word, padding=True, return_tensors="pt")
    return output['input_ids'].numpy()[0]

def get_clip_text_feat(model,tokenizer, word):
    inputs = tokenizer(word, padding=True, return_tensors="tf")



    text_features = model.get_text_features(**inputs)
    # print(f"text feature shape: {text_features.shape}")  # 1x512
    return text_features

def get_visual_feat(data):
    cat15_visual_video_id_feat ={}
    video_ids = data.keys()

    # print(len(video_ids))
    cat15_encodes_all_frames_dir = Path("./cat15_encode_all_frames")
    for id in video_ids:
        print(id)
        for i in cat15_encodes_all_frames_dir.glob("*.pt"):

            # print(i)
            file_id = str(i).split("\\")[-1].split(".")[0]
            # print(f"file_id: {file_id}")
            if id == file_id:

                loaded_tensor = torch.load(i)
                loaded_tensor = loaded_tensor.numpy().tolist()

                # Print the loaded PyTorch tensor
                # print(type(loaded_tensor))
                # print(loaded_tensor)
                # print(loaded_tensor.shape)
                cat15_visual_video_id_feat[id] =loaded_tensor
    save_json("./cat15_visual_video_id_feat.json", cat15_visual_video_id_feat)
    # print(cat15_visual_video_id_feat)

if __name__ == '__main__':

    with open('./msr_vtt_category_15_videoId_cap_data.json', 'r') as f:  
        cat15_video_cap_data = json.load(f)

    model = TFCLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    video_id_tokenized_cap = get_video_id_to_tokenized_caps (cat15_video_cap_data)
    video_id_tokenized_cap = load_json("./video_id_tokenized_cap.json")

    cat15_visual_video_id_feat = get_visual_feat(cat15_video_cap_data)
    cat15_visual_video_id_feat = load_json("./cat15_visual_video_id_feat.json")

    msr_vtt_cat15_d4rl_dataset = {}
    actions =[]
    terminals = []
    rewards = []
    observations = []
    video_ids = []
    action_tokens = []
    with tqdm(total = len(video_id_tokenized_cap.keys())) as pbar:
        for video_id,caps in video_id_tokenized_cap.items():
            # print(video_id,len(caps))
            # print(caps[0])

            for tokenized_cap in caps:

                    # action and terminals steps

                    for i in range(len(tokenized_cap)):
                        if i<len(tokenized_cap)-1:
                            word = tokenized_cap[i+1]
                            # print(word)
                            action_token = create_action_vector(tokenizer,word).tolist()
                            action_vector = [0, 1, 0]
                            terminals.append(False)
                        else:
                            word = "."
                            action_vector=[0,0,1]
                            action_token = create_action_vector(tokenizer,word).tolist()
                            terminals.append(True)

                        actions.append(action_vector)
                        action_tokens.append(action_token)





                    # reward steps & video_ids

                    for i in range(len(tokenized_cap)):

                        # cider generation example
                        ref_tokens = tokenized_cap
                        cand_tokens = tokenized_cap[:i+1]
                        # print(f"ref tokens: {tokenized_cap}\n candidate tokens: {cand_tokens}")
                        # print(f"reward:{cal_cider_score(ref_tokens,cand_tokens)} ")
                        rewards.append(cal_cider_score(ref_tokens,cand_tokens))
                        # 1. "it" , "it is a dog"
                        # 2. "it is ", "it is a dog"
                        video_ids.append(video_id)

                    # observation steps
                    word_count = len(tokenized_cap)
                    encodes = cat15_visual_video_id_feat[video_id]
                    # print(len(encodes)) #420
                    # print(type(encodes)) #<class 'list'>
                    # print(np.array(encodes).shape) #(420, 1, 512)
                    # print()
                    samples =np.round(np.linspace(0,len(encodes)-1, word_count))

                    sampled_encodes = [encodes[int(sample)] for sample in samples]
                    # print(f"len(word_count): {word_count}") #12
                    # print(f"len(sampled_encodes): {len(sampled_encodes)}") #12
                    # print(sampled_encodes)
                    temp_word_list =[]

                    for word,encode in zip(tokenized_cap,encodes):
                        # print(word)
                        # print(encode)
                        # print(np.array(encode).shape)
                        temp_word_list.append(word)
                        word = " ".join(temp_word_list)
                        # print(word)
                        visual_feat = np.array(encode)
                        text_feat = get_clip_text_feat(model, tokenizer, word)
                        # print(type(text_feat)) #<class 'tensorflow.python.framework.ops.EagerTensor'>
                        text_feat = text_feat.numpy()
                        # print(text_feat.tolist())
                        # print(text_feat.shape)
                        observation = np.concatenate((visual_feat,text_feat), axis=1)
                        # print(observation.shape) #1,1024
                        observation = observation.tolist()
                        # print(len(observation)) #12
                        observations.append(observation)




            pbar.update(1)

        # print(len(observations))
        # print(len(video_ids))

    msr_vtt_cat15_d4rl_dataset['actions'] = actions
    msr_vtt_cat15_d4rl_dataset['terminals'] =terminals
    msr_vtt_cat15_d4rl_dataset['rewards'] = rewards
    msr_vtt_cat15_d4rl_dataset['observations'] = observations
    msr_vtt_cat15_d4rl_dataset['video_ids'] = video_ids
    msr_vtt_cat15_d4rl_dataset['action_tokens'] = action_tokens

    save_json("../gym/data/msr_vtt_cat15_d4rl_dataset.json", msr_vtt_cat15_d4rl_dataset)
    # print(msr_vtt_cat15_d4rl_dataset)
    # print(actions)
    print(np.array(actions).shape)
    # print(terminals)
    print(np.array(terminals).shape)
    # print(np.array(action_tokens).shape)
    print(np.array(rewards).shape)
    print(rewards)
    print(np.array(observations).shape)
    print(np.array(video_ids).shape)
    # print(np.array(rewards))
    # # print(actions.shape)
    # print(len(actions))
    # print(steps)
    # assert len(actions) == steps
















    # this var will save "video_id": {
    #   actions= [[...]] #total size of words in a caption
    #   observations 
    #   
    # }
    # video_episode_dictionary = {}
    # # creating vocabulary from data file
    # get_vocab(cat15_video_cap_data)
    # # generating word vectors for all the words from the vocab
    # get_word_vectors()

    # # testing pseudo prediction and decoding
    # print(vocabs[1])
    # #
    # action_vector = make_action_step(vocabs[1])
    # # print(action_vector)
    # # selection of action by simple argmax
    # idx = np.argmax(action_vector) #idx --> 1 (expected)
    # print(idx)
    # # decoding from vocab file
    # print(vocabs[idx]) # same word expected vocabs[1]

    # print(len(vocabs))