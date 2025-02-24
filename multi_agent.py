from graph_env import GraphColoringEnv
from q_learning_agent import QLearningAgent
import random
import time

def train_multi_agent(episodes=50):
    env = GraphColoringEnv(initial_nodes=3, max_colors=3)
    agent_1 = QLearningAgent(num_nodes=3, num_colors=3, epsilon=0.7, epsilon_decay=0.02)
    agent_2 = QLearningAgent(num_nodes=3, num_colors=3, epsilon=0.7, epsilon_decay=0.02)
    max_steps = 1000
    failed_episodes = 0
    
    for episode in range(1, episodes + 1):  # ✅ Start episodes from 1
        print(f"\n🚀 Starting Episode {episode}")
        state = env.reset()
        done = False
        steps = 0
        success = False
        
        while not done and steps < max_steps:
            uncolored_nodes = [n for n in env.graph.nodes if env.node_colors[n] is None]
            if not uncolored_nodes:
                print(f"✅ Episode {episode}: No uncolored nodes left! Expanding graph...")
                env.expand_graph()
                success = True
                done = True
                continue
            
            random.shuffle(uncolored_nodes)
            
            for agent, agent_name in zip([agent_1, agent_2], ["Agent 1", "Agent 2"]):
                if not uncolored_nodes:
                    break  # If all nodes are colored, stop selecting
                
                node = random.choice(uncolored_nodes)
                valid_colors = [color for color in range(env.max_colors) if env.is_valid_color(node, color)]
                
                if valid_colors:
                    action = random.choice(valid_colors)
                    print(f"🎯 {agent_name}: Trying Node {node} with Color {action}")
                    next_state, reward = env.step(node, action)
                    agent.update_q_value(state, action, reward, next_state)
                    state = next_state
                    
                    if reward > 0:
                        print(f"✅ {agent_name}: Successfully colored Node {node} with Color {action}")
                    else:
                        print(f"❌ {agent_name}: Invalid move on Node {node} with Color {action}")
                
                time.sleep(1)  # ✅ Pause to visualize step
                steps += 1
        
        if steps >= max_steps:
            print(f"❌ Episode {episode}: Step limit reached, marking as failed.")
            failed_episodes += 1
        
        agent_1.decay_epsilon()
        agent_2.decay_epsilon()
        print(f"✅ Completed Episode {episode}, Epsilon (Agent 1): {agent_1.epsilon}, Epsilon (Agent 2): {agent_2.epsilon}")
        
        if episode % 5 == 0 or episode == episodes:
            print(f"📊 Training Progress: {episode}/{episodes} episodes completed.")
    
    print(f"🏁 Training Summary: {failed_episodes}/{episodes} episodes failed due to constraints.")
    return agent_1, agent_2

if __name__ == "__main__":
    trained_agent_1, trained_agent_2 = train_multi_agent()
    print("Multi-Agent Training Complete.")
