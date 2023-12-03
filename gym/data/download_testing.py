
 # dict_keys(['actions',
	# 		'infos/action_log_probs',
	# 		'infos/qpos', 'infos/qvel',
	# 		'metadata/algorithm', 'metadata/iteration',
	# 		'metadata/policy/fc0/bias', 'metadata/policy/fc0/weight',
	# 		  'metadata/policy/fc1/bias', 'metadata/policy/fc1/weight',
	# 		  'metadata/policy/last_fc/bias', 'metadata/policy/last_fc/weight',
	# 		  'metadata/policy/last_fc_log_std/bias',
	# 		  'metadata/policy/last_fc_log_std/weight',
	# 		  'metadata/policy/nonlinearity', 'metadata/policy/output_distribution',
	# 		  'next_observations', 'observations', 'rewards', 'terminals', 'timeouts'])





import gym
import numpy as np

import collections
import pickle

import d4rl


datasets = []

for env_name in ['hopper']:
	for dataset_type in ['medium']:
		name = f'{env_name}-{dataset_type}-v2'
		env = gym.make(name)
		dataset = env.get_dataset()
		print(f"keys: {dataset.keys()}")
		# print(dataset)
		print(dataset['actions'].shape)
		print(type(dataset['actions'][0]))
		print(dataset['rewards'].shape)

		N = dataset['rewards'].shape[0]
		print(f'N: {N}')
		data_ = collections.defaultdict(list)

		n=0
		for i in range (N):
			if dataset['terminals'][i]:
				print(dataset['terminals'][i])
				print(f"step: {i} Observations")
				# print(len(dataset['terminals']))
				print(len(dataset['observations'][i]))
				print(dataset['observations'][i])
				print(f"step: {i} Next Observations")
				# print(len(dataset['next_observations'][i]))
				print(dataset['next_observations'][i])
				print(dataset['actions'][i])
				print('-'*70)
				print(dataset['terminals'][i+1])
				print(f"step: {i+1} Observations")
				# print(len(dataset['terminals']))
				print(len(dataset['observations'][i]))
				print(dataset['observations'][i+1])
				print(f"step: {i+1} Next Observations")
				# print(len(dataset['next_observations'][i]))
				print(dataset['next_observations'][i+1])
				print(dataset['actions'][i])
				# n+=1
				# if n==2:break
				break
				
		# use_timeouts = False
		# if 'timeouts' in dataset:
		# 	use_timeouts = True

		# episode_step = 0
		# paths = []
		# for i in range(N):
		# 	print(f"episode: {episode_step}")
		# 	done_bool = bool(dataset['terminals'][i])
		# 	if use_timeouts:
		# 		final_timestep = dataset['timeouts'][i]
		# 	else:
		# 		final_timestep = (episode_step == 1000-1)
		# 	for k in ['observations', 'next_observations', 'actions', 'rewards', 'terminals']:
		# 		data_[k].append(dataset[k][i])
		# 	# print(f'data: ------------\n {data_}')
		# 	if done_bool or final_timestep:
		# 		episode_step = 0
		# 		episode_data = {}
		# 		for k in data_:
		# 			episode_data[k] = np.array(data_[k])
		# 		paths.append(episode_data)
		# 		data_ = collections.defaultdict(list)
		# 	episode_step += 1
				
	


		# returns = np.array([np.sum(p['rewards']) for p in paths])
		# num_samples = np.sum([p['rewards'].shape[0] for p in paths])
		# print(f'Number of samples collected: {num_samples}')
		# print(f'Trajectory returns: mean = {np.mean(returns)}, std = {np.std(returns)}, max = {np.max(returns)}, min = {np.min(returns)}')
		# print(f'paths: ====================\n{paths}')
		# with open(f'{name}.pkl', 'wb') as f:
		# 	pickle.dump(paths, f)
	