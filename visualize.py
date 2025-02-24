import matplotlib.pyplot as plt
import numpy as np

def plot_rewards(episode_rewards):
    """Plots the average total reward per episode."""
    plt.figure()
    plt.plot(range(len(episode_rewards)), episode_rewards)
    plt.xlabel("Episode")
    plt.ylabel("Average Reward")
    plt.title("Agent Reward Over Episodes")
    plt.show()

def plot_q_values(q_table):
    """Plots the Q-values for each node-color pair."""
    states = list(q_table.keys())
    q_values = [max(q_table[s].values()) if q_table[s] else 0 for s in states]
    
    plt.figure()
    plt.bar(range(len(states)), q_values)
    plt.xlabel("State Index")
    plt.ylabel("Max Q-value")
    plt.title("Max Q-value per State")
    plt.show()
