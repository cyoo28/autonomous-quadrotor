from heapq import heappush, heappop
import numpy as np

from flightsim.world import World

from .occupancy_map import OccupancyMap

class Node:
    def __init__(self,index,cost,father,astar,goal):
        self.index = index
        self.cost = cost
        self.father = father
        self.astar = astar
        self.goal = goal
        self.a_cost = cost + np.linalg.norm(np.array(goal) - np.array(index))

    def __lt__(self,other):
        if self.astar:
            return self.a_cost<other.a_cost
        else:
            return self.cost<other.cost

def graph_search(world, resolution, margin, start, goal, astar):
    occ_map = OccupancyMap(world, resolution, margin)
    # Retrieve the index in the occupancy grid matrix corresponding to a position in space.
    start_index = tuple(occ_map.metric_to_index(start))
    goal_index = tuple(occ_map.metric_to_index(goal))

    pq = []
    start_node = Node(start_index,0,None,astar,goal_index)  # cost and index

    path = goal.reshape(1,3)

    nodes_expanded = 0
    closed = set()  # closed set
    opened = {start_index:start_node}  # opened set
    # opened = set()
    # vis = np.zeros((500,500,500))
    heappush(pq,start_node)

    while pq:
        cur_node = heappop(pq)
        closed.add(cur_node.index)
        nodes_expanded += 1
        cur_node_x = cur_node.index[0]
        cur_node_y = cur_node.index[1]
        cur_node_z = cur_node.index[2]
        # vis[cur_node_x,cur_node_y,cur_node_z] = 1
        if cur_node.index == goal_index:

            while cur_node.father:
                path = np.insert(path,0,occ_map.index_to_metric_center(cur_node.index).reshape(1,3),axis = 0)
                cur_node = cur_node.father

            break

        neighbor_nodes = [(cur_node_x+1,cur_node_y, cur_node_z),(cur_node_x+1,cur_node_y+1,cur_node_z),
                          (cur_node_x+1,cur_node_y+1, cur_node_z+1),(cur_node_x+1,cur_node_y+1, cur_node_z-1),
                          (cur_node_x+1,cur_node_y-1, cur_node_z),(cur_node_x+1,cur_node_y-1, cur_node_z+1),
                          (cur_node_x+1,cur_node_y-1, cur_node_z-1),(cur_node_x+1,cur_node_y, cur_node_z+1),
                          (cur_node_x+1,cur_node_y, cur_node_z-1),(cur_node_x,cur_node_y+1, cur_node_z),
                          (cur_node_x,cur_node_y+1, cur_node_z+1),(cur_node_x,cur_node_y+1, cur_node_z-1),
                          (cur_node_x,cur_node_y-1, cur_node_z),(cur_node_x,cur_node_y-1, cur_node_z+1),
                          (cur_node_x,cur_node_y-1, cur_node_z-1),(cur_node_x,cur_node_y, cur_node_z+1),
                          (cur_node_x,cur_node_y, cur_node_z-1),(cur_node_x-1,cur_node_y, cur_node_z),
                          (cur_node_x-1,cur_node_y+1,cur_node_z),(cur_node_x-1,cur_node_y+1, cur_node_z+1),
                          (cur_node_x-1,cur_node_y+1, cur_node_z-1),(cur_node_x-1,cur_node_y-1, cur_node_z),
                          (cur_node_x-1,cur_node_y-1, cur_node_z-1),(cur_node_x-1,cur_node_y, cur_node_z+1),
                          (cur_node_x-1,cur_node_y, cur_node_z-1),(cur_node_x-1,cur_node_y-1, cur_node_z+1)]

        for neighbor in neighbor_nodes:
            if not occ_map.is_occupied_index(neighbor) and neighbor not in closed:
                cost = cur_node.cost + np.linalg.norm(np.array(cur_node.index) - np.array(neighbor))
                if neighbor not in opened:
                    new_node = Node(neighbor,cost,cur_node,astar,goal_index)
                    opened[neighbor] = new_node
                    heappush(pq,new_node)
                else:
                    if opened[neighbor].cost>cost:
                        opened[neighbor].cost = cost
                        opened[neighbor].father = cur_node

    path[0] = start
    if len(pq) == 0:
        return None,0

    print(nodes_expanded)
    return path,nodes_expanded
