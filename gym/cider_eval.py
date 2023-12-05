import json

from cider_metric.cider import cal_cider_score
from transformers import AutoTokenizer, TFCLIPModel
from decision_transformer.models.decision_transformer import DecisionTransformer
import pickle
import torch
import argparse
import numpy as np


tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")

def decode(token_list):
    return tokenizer.decode(token_list)





def get_st_mean_std(trajectories):
    states, traj_lens, returns = [], [], []
    for path in trajectories:
        # print(path)
        states.append(path['observations'])
        traj_lens.append(len(path['observations']))
        returns.append(path['rewards'].sum())
    states = np.concatenate(states, axis=0)
    # print(type(states)) #<class 'numpy.ndarray'>
    # print(states.shape) #(21018, 1024)
    state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
    return state_mean,state_std

def run (variant):
    max_ep_len = 20
    dataset_path = "./data/msr_vtt_cat15_d4rl_dataset.pkl"

    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    state_dim = trajectories[0]['observations'].shape[1] #1024
    act_dim = trajectories[0]['actions'].shape[1] #3
    model = DecisionTransformer(
        state_dim=state_dim,
        act_dim=act_dim,
        max_length=variant['K'],
        max_ep_len=max_ep_len,
        hidden_size=variant['embed_dim'],
        n_layer=variant['n_layer'],
        n_head=variant['n_head'],
        n_inner=4 * variant['embed_dim'],
        activation_function=variant['activation_function'],
        n_positions=1024,
        resid_pdrop=variant['dropout'],
        attn_pdrop=variant['dropout'],
    )
    scale = 1
    state_mean, state_std = get_st_mean_std(trajectories)
    target_return = 3.92
    device = 'cuda'

    file_path = "E:\\642\Final_project\decision-transformer\gym\\results\decision_cap_model_mid.pth"
    model.load_state_dict(torch.load(file_path))
    model.eval()
    model.to(device=device)
    # print(f"state mean: {state_mean}")
    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    msr_cat15_video_id_cap_datapath = "../data_msr_vtt/msr_vtt_category_15_videoId_cap_data.json"
    with open(msr_cat15_video_id_cap_datapath,'r') as f :
        msr_cat15_video_id_cap = json.load(f)

    # ['actions', 'terminals', 'rewards', 'observations', 'video_ids']
    for t in range(len(trajectories)):
        print(f"episode: {t}")
        state = trajectories[t]['observations'][0]
        states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
        actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
        rewards = torch.zeros(0, device=device, dtype=torch.float32)
        ep_return = target_return
        print(ep_return)
        target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

        token_list =[]
        for step in range(max_ep_len):
            print(f"step: {step}")

            # add padding
            actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
            rewards = torch.cat([rewards, torch.zeros(1, device=device)])
            print(f"states: {states}")
            # model predicting action
            action = model.get_action(
                (states.to(dtype=torch.float32) - state_mean) / state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
            )
            actions[-1] = action
            action = action.detach().cpu().numpy()
            print(action)
            # word token decode
            word_token_idx = np.argmax(action)
            action_word_token = trajectories[t]['actions'][step][word_token_idx]
            token_list.append(action_word_token)

            state = trajectories[t]['observations'][step]
            print(f"state ** : {state}")
            reward = trajectories[t]['rewards'][step]
            done = trajectories[t]['terminals'][step]

            cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
            states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
            # states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward
            pred_return = target_return[0, -1] - (reward / scale)
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)



            if done:
                target_return = 3.92
                break

            # print(state)

        print(decode(token_list))
        break


    states = torch.from_numpy(np.array(state)).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)


    # print(path.keys())
    # print(path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='msr_vtt')  # msr_vtt, hopper
    parser.add_argument('--dataset', type=str, default='cat15')  # medium, medium-replay, medium-expert, expert, cat15
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=20)
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--model_type', type=str, default='dt')  # dt for decision transformer, bc for behavior cloning
    parser.add_argument('--embed_dim', type=int, default=128)
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=100)
    parser.add_argument('--max_iters', type=int, default=1051)
    parser.add_argument('--num_steps_per_iter', type=int,
                        default=20)  # total steps 21018 / 20 (steps/iter or 20 episode per iteration), total iteration 1051
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=False)

    args = parser.parse_args()
    run(variant = vars(args))

