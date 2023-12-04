import numpy as np

import collections
import pickle
import json
from tqdm import tqdm


dataset_path = "./msr_vtt_cat15_d4rl_dataset.json"
with open(dataset_path,'r') as f:
    msr_vtt_d4rl_dataset = json.load(f)

print(msr_vtt_d4rl_dataset.keys())
datasets =[]

N = np.array(msr_vtt_d4rl_dataset['rewards']).shape[0] #total steps 21503
print(f"total steps: {N}")




# defaultdict(<class 'list'>, {'observations': [array([ 0.01983191, -0.08950131, -0.00319691, -0.03601582,  0.06619842,
#        -0.09322933,  0.06039053, -0.08753889, -0.01877608, -0.16759904,
#        -0.07047922,  0.07015167,  0.0302026 ,  0.05502646,  0.11365079,
#         0.06842492, -0.13811582], dtype=float32)], 'next_observations': [array([-3.8486063e-03, -5.2394319e-02,  8.3050327e-03, -2.5620908e-01,
#        -2.9927862e-01,  9.2399962e-02, -3.3266103e-01,  8.7400869e-02,
#        -1.9692373e-02, -7.2151250e-01,  9.5597059e-01, -5.3302956e-01,
#        -5.4545808e+00, -8.7090406e+00,  4.5068407e+00, -9.2885571e+00,
#         4.7328596e+00], dtype=float32)], 'actions': [array([-0.22293739, -0.7359478 , -0.8599511 ,  0.29579234, -0.8416547 ,
#         0.43432042], dtype=float32)], 'rewards': [-0.20008013], 'terminals': [False]})

#----------------------------------------------
# actions [array([49406,   275, 49407])]  # ok actions
# terminals [False] # ok
#  rewards [0.0] # ok
#  observations [array([ 0.34689599,  0.1342805 ,  0.01525949, ..., -0.46926677, -0.05507876,  0.06334257])]  ok
# video_ids [array('video1919', dtype='<U9')]

episode_step = 0
paths = []
data_ = collections.defaultdict(list)
print(msr_vtt_d4rl_dataset['video_ids'])
with tqdm(total =N) as pbar:
    for i in range(N):
        done_bool = msr_vtt_d4rl_dataset['terminals'][i]
        # print(done_bool)
        #  action vectors are created more than size 3 due to words like 2016 --> array([49406,   273,   271,   272,   277, 49407])
        # if found, skipping the whole episode
        if len(msr_vtt_d4rl_dataset['actions'][i]) == 3:
            for k in ['actions', 'terminals', 'rewards', 'observations', 'video_ids']:
                # print(np.array(msr_vtt_d4rl_dataset[k][i]))
                if k =="terminals" or k == 'rewards':
                    data_[k].append(msr_vtt_d4rl_dataset[k][i])
                elif k == "observations":
                    data_[k].append(np.array(msr_vtt_d4rl_dataset[k][i][0]))
                elif k == 'actions':
                    # print(len(msr_vtt_d4rl_dataset[k][i]))

                    # action vectors are created more than size 3 due to words like 2016 --> array([49406,   273,   271,   272,   277, 49407])
                        # print("gotchaaaaaaaa")
                        # print(f"episode_step: {i}")
                        # print(f"video idx: {i//20}")
                        # print(f"video_id: {msr_vtt_d4rl_dataset['video_ids'][i//20]}") #video4052
                        # skipping these steps | impact on the performance of the model : unknown

                    data_[k].append(np.array(msr_vtt_d4rl_dataset[k][i]))


            # print(data_)

            #testing data_ shapes
            # for k,v in data_.items():
            #     if k in ['actions', 'observations']:
            #         print(k, v[0].shape)

            if done_bool:
                # episode_step = 0
                episode_data = {}
                for k in data_:
                    # data[]
                    # print(k)
                    # print(data_[k])
                    # print(f"step: {i}")
                    # print(f"video_id: {msr_vtt_d4rl_dataset['video_ids'][i//20]}")
                    # print(type(data_[k][0]))
                    # if k =="actions":
                    #     print(data_[k][0].shape)
                    episode_data[k] = np.array(data_[k])
                paths.append(episode_data)
                data_ = collections.defaultdict(list)

                episode_step+=1

            pbar.update(1)



# print(paths)
returns = np.array([np.sum(p['rewards']) for p in paths])
num_samples = np.sum([p['rewards'].shape[0] for p in paths])
print(f'Number of samples collected: {num_samples}')
print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
print(f'total episodes: {episode_step}')
name = "./msr_vtt_cat15_d4rl_dataset"
with open(f'{name}.pkl', 'wb') as f:
    pickle.dump(paths, f)
