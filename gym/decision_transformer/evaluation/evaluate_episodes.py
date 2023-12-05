import numpy as np
import torch
import random



def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate_episode_rtg(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1000.,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',
    ):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    # state = env.reset()
    # print(state)
    # [1.25366579e+00 - 2.81061058e-03  2.02695562e-03 - 3.36053777e-03
    #  - 1.64279658e-03  2.91281450e-03  4.61904991e-03  8.10681572e-04
    #  - 1.94458413e-03 - 1.33148197e-04  1.19520414e-05]

    #env reset
    ## selecting a random episode to evaluate on
    rand_start = random.randint(1,2000) #randome episode
    # print(f"rand_start: {rand_start}")
    # print(env[rand_start])
    state = env[rand_start]['observations'][0]

    #print(state)#[-0.02100435  0.42834368 -0.09245506 ... -0.7490868  -0.108756650.39127845]
    # print(type(state)) #<class 'numpy.ndarray'>


    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    #decoding episode
    predicted_word_tokens = []
    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        #model predicting action
        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return.to(dtype=torch.float32),
            timesteps.to(dtype=torch.long),
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()


        #PREDICTING  action prob of current action list of current state
        # print(f"action: {action}")
        # print(type(action))

        # state, reward, done, _ = env.step(action)
        # print("state: ")
        # print(state)
        # # [1.14985098e+00 - 2.08796491e-01  1.53788230e-03 - 6.48324935e-01
        # #  4.64601937e-01  2.42331666e-01 - 7.87986333e-01 - 1.48095924e+00
        # #  - 1.59453952e-02 - 3.90510349e+00  8.67828997e-01]
        # print("reward")
        # print(reward) #1.2425690849682944
        # print("done")
        # print(done) #True
        #--------------------------------------
        #env step
        # rand_start
        word_token_idx = np.argmax(action)
        action_word_token = env[rand_start]['actions'][t][word_token_idx]
        # print(f"word_token_idx: {word_token_idx}")
        # print(f"action word: {action_word_token}")
        state = env[rand_start]['observations'][t]
        reward = env[rand_start]['rewards'][t]
        done  = env[rand_start]['terminals'][t]




        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        if mode != 'delayed':
            pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0,-1]
        target_return = torch.cat(
            [target_return, pred_return.reshape(1, 1)], dim=1)
        timesteps = torch.cat(
            [timesteps,
             torch.ones((1, 1), device=device, dtype=torch.long) * (t+1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length
