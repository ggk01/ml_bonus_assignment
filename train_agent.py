from graph_env import GraphColoringEnv
from q_learning_agent import QLearningAgent
import random
import pickle

def train_agent(episodes=10, show_graph=True):
    env = GraphColoringEnv(initial_nodes=3, max_colors=3)
    agent = QLearningAgent(num_nodes=3, num_colors=3, epsilon=0.7, epsilon_decay=0.02)
    max_steps = 1000  # Prevent infinite loops
    failed_episodes = 0  # Track failed episodes
    
    energy_data = {}  # Stores energy values per node per episode
    single_agent_rewards = []  # Stores cumulative rewards per episode
    
    for episode in range(1, episodes + 1):  # ✅ Start episodes from 1
        print(f"🚀 Starting Episode {episode}")
        state = env.reset()
        done = False
        steps = 0  # Step counter
        success = False
        episode_reward = 0  # Track total reward for the episode
    
        while not done and steps < max_steps:
            uncolored_nodes = [n for n in env.graph.nodes if env.node_colors[n] is None]
            if not uncolored_nodes:
                print(f"Episode {episode}: No uncolored nodes left! Expanding graph...")
                env.expand_graph()
                success = True
                done = True
                continue

            random.shuffle(uncolored_nodes)
            
            # ✅ Find a valid node and action
            found_valid = False
            for node in uncolored_nodes:
                valid_colors = [color for color in range(env.max_colors) if env.is_valid_color(node, color)]
                if valid_colors:
                    action = random.choice(valid_colors)
                    found_valid = True
                    break

            if not found_valid:
                print(f"Episode {episode}: No valid moves found! Skipping step.")
                continue

            print(f"Episode {episode}, Step {steps}: Trying Node {node} with Color {action}")

            next_state, reward = env.step(node, action)
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            episode_reward += reward
            
            if node not in energy_data:
                energy_data[node] = []
            energy_data[node].append(reward)
            
            if show_graph:
                env.render(step=steps, message=f"Episode {episode}, Step {steps}")

            steps += 1
        
        if steps >= max_steps:
            print(f"❌ Episode {episode}: Step limit reached, marking as failed.")
            failed_episodes += 1
        
        single_agent_rewards.append(episode_reward)
        agent.decay_epsilon()
        print(f"Completed Episode {episode}, Epsilon: {agent.epsilon}")
        
        if episode % 10 == 0 or episode == episodes:
            print(f"Training Progress: {episode}/{episodes} episodes completed")
    
    print(f"Training Summary: {failed_episodes}/{episodes} episodes failed due to constraints.")
    
    # ✅ Store training data for visualization
    training_data = {
        "energy_data": energy_data,
        "single_agent_rewards": single_agent_rewards
    }
    with open("training_data.pkl", "wb") as f:
        pickle.dump(training_data, f)
    
    return agent

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--show_graph", action="store_true", help="Show graph visualization at each step")
    args = parser.parse_args()
    
    trained_agent = train_agent(show_graph=args.show_graph)
    print("Training complete. Data saved for visualization.")
