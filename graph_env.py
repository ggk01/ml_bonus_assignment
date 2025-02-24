import random
import networkx as nx
import matplotlib.pyplot as plt
import time

class GraphColoringEnv:
    def __init__(self, initial_nodes=5, max_colors=3, show_graph=True):
        self.graph = self.generate_fixed_graph(initial_nodes)
        self.node_colors = {node: None for node in self.graph.nodes}
        self.max_colors = max_colors
        self.reward_correct = 1
        self.reward_wrong = -1
        self.steps = 0
        self.show_graph = show_graph
        self.fixed_edges = list(self.graph.edges)  # ✅ Store fixed edges to prevent changes

    def generate_fixed_graph(self, num_nodes):
        """Generates a fixed connected graph that does not change during an episode."""
        while True:
            graph = nx.erdos_renyi_graph(n=num_nodes, p=0.5, seed=42)
            if nx.is_connected(graph):
                return graph

    def reset(self):
        """Resets the environment for a new episode while preserving the initial graph structure."""
        self.node_colors = {node: None for node in self.graph.nodes}  # ✅ Reset only colors
        self.steps = 0
        self.graph.clear_edges()
        self.graph.add_edges_from(self.fixed_edges)  # ✅ Restore original edges
        
        if self.show_graph:
            self.render(step=0, message="Initial Graph")
        return self.get_state()

    def get_state(self):
        """Returns the current state (colors assigned to nodes)."""
        return tuple(self.node_colors.values())

    def is_valid_color(self, node, color):
        """Checks if a color assignment is valid based on graph constraints."""
        if self.node_colors[node] is not None:
            return False  # ✅ Prevent re-coloring already colored nodes
        
        neighbor_colors = {self.node_colors[n] for n in self.graph.neighbors(node) if self.node_colors[n] is not None}
        return color not in neighbor_colors  # ✅ Ensure no adjacent nodes have the same color

    def step(self, node, color):
        """Executes an action (coloring a node), renders the graph if enabled, and returns the new state."""
        if self.node_colors[node] is not None:
            if self.show_graph:
                self.render(step=self.steps, message=f"❌ Bad Move: Node {node} Already Colored")
            return self.get_state(), self.reward_wrong  # Node already colored
        
        if self.is_valid_color(node, color):
            self.node_colors[node] = color
            reward = self.reward_correct
            if self.show_graph:
                self.render(step=self.steps, message=f"✅ Good Move: Node {node} -> Color {color}")
            time.sleep(0.5)  # ✅ Pause to show the coloring step
        else:
            reward = self.reward_wrong
            if self.show_graph:
                self.render(step=self.steps, message=f"❌ Bad Move: Node {node} -> Color {color} (Invalid)")
            time.sleep(0.5)
        
        self.steps += 1
        return self.get_state(), reward


    def expand_graph(self):
        """Expands the current graph by adding a new node and random connections."""
        if len(self.graph.nodes) >= 20:  # Prevent infinite growth
            return  

        new_node = len(self.graph.nodes)  # Add a new node with a unique ID
        self.graph.add_node(new_node)
        self.node_colors[new_node] = None  # New node starts uncolored

        # Randomly connect the new node to existing nodes (max 50% of current nodes)
        existing_nodes = list(self.graph.nodes)
        num_connections = min(len(existing_nodes) // 2, random.randint(1, 3))  # Limit new connections

        connections = random.sample(existing_nodes, num_connections)
        for node in connections:
            self.graph.add_edge(new_node, node)

        self.fixed_edges = list(self.graph.edges)  # Store edges for reference

        if self.show_graph:
            self.render(step=self.steps, message=f"Graph Expanded: Added Node {new_node}")

        return self.get_state()


    def render(self, step, message=""): 
        """Visualizes the graph step by step, ensuring all nodes are shown before moving on if enabled."""
        if not self.show_graph:
            return
        
        color_list = ["red", "blue", "green", "yellow", "purple", "orange"]  # ✅ Define color mappings
        
        color_map = [
            color_list[self.node_colors[node]] if self.node_colors[node] is not None else "gray"
            for node in self.graph.nodes
        ]
        
        plt.figure(figsize=(6, 4))
        nx.draw(self.graph, with_labels=True, node_color=color_map, edge_color='black', linewidths=1, edgecolors='black', cmap=plt.get_cmap("Set1"))
        plt.title(f"Step {step}: {message}")
        plt.show()
        time.sleep(0.5)  # ✅ Pause after each render to visualize steps clearly
