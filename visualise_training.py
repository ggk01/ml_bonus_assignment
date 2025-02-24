import matplotlib.pyplot as plt
import numpy as np
import pickle

def plot_energy_per_node(energy_data):
    """Plots energy values per node per episode."""
    plt.figure(figsize=(10, 5))
    for node, energies in energy_data.items():
        plt.plot(range(1, len(energies) + 1), energies, label=f'Node {node}')
    
    plt.xlabel('Episode')
    plt.ylabel('Energy Value')
    plt.title('Energy Value per Node per Episode')
    plt.legend()
    plt.grid()
    plt.show()

def plot_cumulative_rewards(single_agent_rewards, reward_data_1=None, reward_data_2=None):
    """Plots cumulative rewards for single or multi-agent scenarios."""
    plt.figure(figsize=(10, 5))
    
    if single_agent_rewards:
        plt.plot(range(1, len(single_agent_rewards) + 1), np.cumsum(single_agent_rewards), label='Single Agent')
    if reward_data_1 and reward_data_2:
        plt.plot(range(1, len(reward_data_1) + 1), np.cumsum(reward_data_1), label='Agent 1')
        plt.plot(range(1, len(reward_data_2) + 1), np.cumsum(reward_data_2), label='Agent 2')
    
    plt.xlabel('Timestep')
    plt.ylabel('Cumulative Reward')
    plt.title('Cumulative Reward Over Time')
    plt.legend()
    plt.grid()
    plt.show()

if __name__ == "__main__":
    # Load real training data
    with open("training_data.pkl", "rb") as f:
        training_data = pickle.load(f)
    
    energy_data = training_data["energy_data"]
    single_agent_rewards = training_data.get("single_agent_rewards", [])
    reward_data_1 = training_data.get("reward_data_1", None)
    reward_data_2 = training_data.get("reward_data_2", None)
    
    plot_energy_per_node(energy_data)
    plot_cumulative_rewards(single_agent_rewards, reward_data_1, reward_data_2)
