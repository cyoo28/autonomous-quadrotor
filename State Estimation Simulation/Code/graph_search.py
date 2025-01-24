from heapq import heappush, heappop
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap

def graph_search(world, resolution, margin, start, goal, astar):
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    # Setup neighbor indices and start node
    neighbors = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1],
                          [1, 1, 0], [1, -1, 0], [1, 0, 1], [1, 0, -1],
                          [-1, 1, 0], [-1, -1, 0], [-1, 0, 1], [-1, 0, -1],
                          [0, 1, 1], [0, 1, -1], [0, -1, 1], [0, -1, -1],
                          [1, 1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, -1],
                          [-1, -1, 1], [-1, 1, -1], [1, -1, -1], [-1, -1, -1]])

    nodes_expanded = 0
    unexpanded = [] # Priority queue to keep track of unexpanded nodes
    parents = {start_index: None}  # Used to keep track of parent nodes
    gscore = {start_index: 0}  # Used to keep track of cost to visit nodes
    heappush(unexpanded, (gscore[start_index],start_index))

    while unexpanded:
        current_cost, current_index = heappop(unexpanded)
        nodes_expanded += 1

        # Check if goal has been reached
        if current_index == goal_index:
            # A path was found
            path = goal
            while current_index != start_index:
                current_index = parents[current_index]
                path = np.vstack([occ_map.index_to_metric_center(current_index),path])
            path = np.vstack([start,path])
            return np.array(path), nodes_expanded

        for x, y, z in neighbors:
            next_node_index = tuple(current_index + np.array([x, y, z]))

            # If node is not valid, move all the way to the back of the queue
            if occ_map.is_occupied_index(next_node_index):
                next_gscore = gscore[current_index] + np.inf
            # If the node is valid then calculate the cost to get there
            else:
                next_gscore = gscore[current_index] + np.sqrt((current_index[0] - next_node_index[0])**2 + (current_index[1] - next_node_index[1])**2 + (current_index[2] - next_node_index[2])**2)
            # If the cost has not been calculated before or is lower than the lowest cost to get there in the past
            # update the path
            if next_node_index not in gscore or next_gscore < gscore[next_node_index]:
                next_cost = next_gscore + 1.9*astar * np.sqrt((next_node_index[0] - goal_index[0])**2 + (next_node_index[1] - goal_index[1])**2 + (next_node_index[1] - goal_index[1])**2)
                heappush(unexpanded, (next_cost, next_node_index))
                gscore[next_node_index] = next_gscore
                parents[next_node_index] = current_index

    # No path was found
    return None, nodes_expanded
