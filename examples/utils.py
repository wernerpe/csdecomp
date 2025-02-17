import os
import numpy as np
from pydrake.all import (
    Rgba,
    MultibodyPlant,
    Context
)

CSD_EXAMPLES_ROOT = os.path.dirname(__file__)

def get_fake_voxel_map():
    np.random.seed(3)
    N = 20
    radius = 0.04
    cols = [Rgba(1,0,1,1)]*N
    min_ = np.array([-0.1, -1.2, 0])
    max_ = np.array([1.2, 1.2, 1.5])
    diff = max_-min_
    locs = np.random.rand(N,3)*diff.reshape(1,3) +min_.reshape(1,3)
    return locs, cols, radius


def densify_waypoints(waypoints_q : list[np.ndarray], 
                      plant : MultibodyPlant, 
                      plant_context : Context, 
                      frame_name = "kinova::end_effector_link", 
                      densify = 200):
    dists = []
    dense_waypoints = []
    for idx in range(len(waypoints_q[:-1])):
        a = waypoints_q[idx]
        b = waypoints_q[idx+1]
        t = np.linspace(1,0, 10)
        locs_endeff = []
        dists_endeff = []
        for tval in t:
            a = a*tval + b*(1-tval)
            qa = a        
            plant.SetPositions(plant_context, qa)
            tf_tot= plant.EvalBodyPoseInWorld(plant_context, plant.GetBodyByName("end_effector_link"))
            tf = tf_tot.translation() + tf_tot.GetAsMatrix4()[:3,:3][:,1] *0.13
            locs_endeff.append(tf)
        for i in range(len(locs_endeff)-1):
            dists_endeff.append(np.linalg.norm(locs_endeff[i]- locs_endeff[i+1]))
        d = np.sum(dists_endeff)
        t = np.linspace(1,0,int(densify*d))
        for tval in t:
            dense_waypoints.append(waypoints_q[idx]*tval + waypoints_q[idx+1]*(1-tval))
    return dense_waypoints

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import pycsdecomp

def write_graph_summary(g: nx.Graph):
    summary = []
    summary.append(f"Graph Summary for Dynamic Roadmap")
    summary.append(f"=====================================")
    
    # Basic graph information
    summary.append(f"Number of nodes: {g.number_of_nodes()}")
    summary.append(f"Number of edges: {g.number_of_edges()}")
    
    # Connected components
    components = list(nx.connected_components(g))
    summary.append(f"Number of connected components: {len(components)}")
    
    # Largest component details
    largest_component = max(components, key=len)
    summary.append(f"Largest component size: {len(largest_component)} nodes")
    
    # Degree information
    degrees = [d for n, d in g.degree()]
    avg_degree = np.mean(degrees)
    max_degree = max(degrees)
    min_degree = min(degrees)
    summary.append(f"Average node degree: {avg_degree:.2f}")
    summary.append(f"Maximum node degree: {max_degree}")
    summary.append(f"Minimum node degree: {min_degree}")
    
    # Path-related metrics
    if nx.is_connected(g):
        pass
        # diameter = nx.diameter(g)
        # avg_shortest_path = nx.average_shortest_path_length(g)
        # summary.append(f"Graph diameter: {diameter}")
        # summary.append(f"Average shortest path length: {avg_shortest_path:.2f}")
    else:
        summary.append("Graph is not connected. Diameter and average shortest path are undefined.")
    
    # Clustering coefficient
    # avg_clustering = nx.average_clustering(g)
    # summary.append(f"Average clustering coefficient: {avg_clustering:.4f}")
    
    # Identify potential bottlenecks
    cut_vertices = list(nx.articulation_points(g))
    summary.append(f"Number of articulation points (potential bottlenecks): {len(cut_vertices)}")
    
    # Density
    density = nx.density(g)
    summary.append(f"Graph density: {density:.4f}")
    
    # Isolated nodes
    isolated_nodes = list(nx.isolates(g))
    summary.append(f"Number of isolated nodes: {len(isolated_nodes)}")
    
    # Component size distribution
    component_sizes = [len(c) for c in components]
    summary.append("Component size distribution:")
    summary.append(f"  Min: {min(component_sizes)}")
    summary.append(f"  Max: {max(component_sizes)}")
    summary.append(f"  Mean: {np.mean(component_sizes):.2f}")
    summary.append(f"  Median: {np.median(component_sizes):.2f}")
    
    # Print summary
    print("\n".join(summary))
    
    return summary

def plot_graph_from_adjacency_list(adj_list, max_nodes = 1000):
    # Create a new graph
    G = nx.Graph()

    # Add edges to the graph
    i = 0
    added_edges = []
    for node, neighbors in adj_list.items():
        i+=1
        for neighbor in neighbors:
            G.add_edge(node, neighbor)
        if i>max_nodes:
            break
    print("done adding nodes")
    # Draw the graph
    plt.figure(figsize=(10, 8))
    pos = nx.spring_layout(G, k=0.5, iterations=100)
    nx.draw_networkx_nodes(G, pos)
    nx.draw_networkx_edges(G, pos, edge_color='k', width=2, alpha=1)  
    # Add edge labels
    # edge_labels = {(u, v): f"{u}-{v}" for u, v in G.edges()}
    # nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)

    plt.title("Graph with Spring Layout")
    # plt.axis('off')
    # plt.tight_layout()
    plt.show()



def get_drm_summary(drm : pycsdecomp.DRM, do_plot = False, max_nodes_to_plot = 1000):
    G = nx.Graph()

    # Add edges to the graph
    edges_added = []
    for node, neighbors in drm.node_adjacency_map.items():
        for neighbor in neighbors:
            e_id = f"{np.min([node,neighbor])},{np.max([node,neighbor])}"
            if e_id not in edges_added:
                edges_added.append(e_id)
                G.add_edge(node, neighbor)

    write_graph_summary(G)
    if do_plot:
        plot_graph_from_adjacency_list(drm.node_adjacency_map, max_nodes_to_plot)