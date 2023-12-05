# DecisionCap-Transformer 

### Overview
**Project Name:** DecisionCap-Transformer

**Objective:** The DecisionCap-Transformer project aims to leverage the novel offline reinforcement learning algorithm, Decision Transformer, for the challenging task of video captioning. Video captioning involves generating textual descriptions for videos, enabling viewers to grasp the video's content without watching it entirely. This approach enhances video searching efficiency and provides valuable insights for the audience.

**Algorithm Used:** Decision Transformer (Chen et al., 2021) - An offline reinforcement learning algorithm that differs from traditional RL algorithms by employing the return-to-go metric instead of rewards.

### Getting Started
**Environment Setup**

1. Create a Conda environment using the provided `conda_env_dec_cap.yml` file:
```bash 
conda env create -f conda_env_dec_cap.yml
```
2. Follow the instructions in the [msr_vtt_readme.md](data_msr_vtt%2Fmsr_vtt_readme.md) for additional setup related to the MSR-VTT dataset.
3. Navigate to the gym/data folder and run [msr_vtt_d4rl_datasets.py](gym%2Fdata%2Fmsr_vtt_d4rl_datasets.py) file:
This will generate a pickle file named `msr_vtt_cat15_d4rl_dataset.pkl`.
```bash 
python msr_vtt_d4rl_datasets.py
```

### Training The model 
Run the training script [experiment_decision_cap.py](gym%2Fexperiment_decision_cap.py) to train the Decision Transformer model on the prepared dataset. Adjust the parameters using argparse arguments as specified in the script.
```bash
python experiment_decision_cap.py
```

### Additional Notes
- Ensure all dependencies are installed and the Conda environment is activated before running any scripts.
- For further details on the Decision Transformer algorithm, refer to the original paper by Chen et al. (2021).

## Reference
Chen, L., Lu, K., Rajeswaran, A., Lee, K., Grover, A., Laskin, M., Abbeel, P., Srinivas, A., Mordatch, I.: Decision Transformer: Reinforcement Learning via Sequence Modeling, http://arxiv.org/abs/2106.01345, (2021)
