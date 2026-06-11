import sys
import math
import heapq
from collections import deque


class BaseGraph:

    def __init__(self):
        self._adjacency_list: list[list[tuple[float, int]]] = [] 

    # ********* Graph Traversals *********
    def bfs(self):
        V = len(self._adjacency_list)
        visited: list[bool] = [False] * V
        parent: list[int] = [-1] * V  
        connected_components: list[list[int]] = []  
        queue: deque[int] = deque()
        
        # Handling Disconnected Graphs => O(V + E) TC
        for start_node in range(V):
            if not visited[start_node]:
                # Starting the BFS on a new Connected Component
                visited[start_node] = True
                connected_components.append([])
                queue.append(start_node) 
                
                while queue:
                    current_node = queue.popleft()  
                    # BFS traversal started on current_node
                    connected_components[-1].append(current_node) 
                    for weight, neighbor in self._adjacency_list[current_node]:  # O(degree(current_node))
                        if not visited[neighbor]:
                            visited[neighbor] = True # node visited but not traversed yet
                            parent[neighbor] = current_node  
                            queue.append(neighbor)  
                    # BFS traversal completed on current_node
        
        return parent, connected_components

    def dfs(self):
        V = len(self._adjacency_list)
        sys.setrecursionlimit(V + 1000)

        visited: list[bool] = [False] * V
        parent: list[int] = [-1] * V
        connected_components: list[list[int]] = []
        
        def dfs_helper(current_node: int, parent_node: int):
            visited[current_node] = True
            parent[current_node] = parent_node
            # DFS traversal on current_node started
            connected_components[-1].append(current_node) # Pre-Order Traversal
            for weight, neighbor in self._adjacency_list[current_node]:
                if not visited[neighbor]:
                    dfs_helper(neighbor, current_node)
            # DFS traversal on current_node completed
            # connected_components[-1].append(current_node) # Post-Order Traversal

        # Handling Disconnected Graphs => O(V + E) TC
        for start_node in range(V):
            if not visited[start_node]:
                # Starting the DFS on a new Connected Component
                connected_components.append([])
                dfs_helper(start_node, -1)
        
        return parent, connected_components

    # ********* SSSP Methods *********
    def _reconstruct_path(self, source_node: int, destination_node: int, parent: list[int]):
        path: list[int] = []
        
        # If destination is not reachable from source
        if parent[destination_node] == -1:
            return path
        
        # Reconstruct path by following parent pointers
        current_node = destination_node
        while current_node != source_node:
            path.append(current_node)
            current_node = parent[current_node]
        path.append(source_node)
        path.reverse()
        return path

    def sssp_bfs(self, source_node: int, destination_node: int) -> tuple[list[int], int] | None:
        
        if source_node == destination_node:
            return [source_node], 0.0

        V = len(self._adjacency_list)
        if V == 0:
            return
        visited: list[bool] = [False] * V
        parent: list[int] = [-1] * V
        queue: deque[int] = deque()

        # Start BFS traversal from source node
        visited[source_node] = True 
        parent[source_node] = -1 
        queue.append(source_node)
        
        while queue:
            current_node = queue.popleft()
            # Early termination if destination is reached
            if current_node == destination_node:
                break
            # BFS traversal on current_node started
            for weight, neighbor in self._adjacency_list[current_node]: # O(degree(current_node))
                if not visited[neighbor]: 
                    visited[neighbor] = True # node visited but not traversed yet
                    parent[neighbor] = current_node 
                    queue.append(neighbor)
            # BFS traversal on current_node completed
        
        # Reconstruct path using helper method
        shortest_path: list[int] = self._reconstruct_path(source_node, destination_node, parent)
        if not shortest_path:
            return [], math.inf
        
        # For unweighted graphs, distance = number of edges = len(path) - 1
        # For equally weighted graphs, we need to get the weight from an edge
        # Since BFS doesn't track edge weights, we'll use path length - 1 (assumes weight 1 per edge)
        # For actual weighted graphs, use Dijkstra's algorithm
        return shortest_path, len(shortest_path) - 1

    def sssp_dijkstra(self, source_node: int, destination_node: int) -> tuple[list[int], float] | None:
        V = len(self._adjacency_list)
        if V == 0:
            return
        
        # If destination is not reachable from source
        if destination_node == source_node:
            return [source_node], 0
        
        parent: list[int] = [-1] * V
        distances: list[int] = [int(1e9)] * V
        priority_queue: list[tuple[float, int]] = []

        parent[source_node] = -1
        distances[source_node] = 0
        heapq.heappush(priority_queue, (0, source_node))  # O(log V)

        while priority_queue:  # O(V) iterations (each node processed at most once)
            current_distance, current_node = heapq.heappop(priority_queue)  # O(log V)
            # Relax edges from current node - O(degree(current_node))
            for neighbor_distance, neighbor in self._adjacency_list[current_node]:
                new_distance = current_distance + neighbor_distance
                if new_distance < distances[neighbor]:
                    parent[neighbor] = current_node
                    distances[neighbor] = new_distance
                    heapq.heappush(priority_queue, (new_distance, neighbor))  # O(log V)
        # Reconstruct path using helper method - O(path_length) = O(V) worst case
        shortest_path: list[int] = self._reconstruct_path(source_node, destination_node, parent)
        if not shortest_path:
            return [], math.inf

        return shortest_path, distances[destination_node]

    # RARELY ASKED ********* APSP Method *********
    def floyd_warshall(self, adjacency_matrix: list[list[float]]):
        V = len(self._adjacency_list)
        if V == 0:
            return
        
        # Initialize distance matrix: 0 for diagonal (self-loops), infinity else where => O(V^2) TC, O(V^2) SC
        distance_matrix: list[list[float]] = [
            [0.0 if row == column else math.inf for column in range(V)] 
            for row in range(V)
        ]
        # Initialize parent matrix: -1 indicates no path, diagonal is -1 (no node is parent for itself) => O(V^2) TC, O(V^2) SC
        parent_matrix: list[list[int]] = [[-1] * V for _ in range(V)]

        # Fill distance matrix with direct edge weights
        for source_idx in range(V): # O(V^2) TC
            for destination_idx in range(V):
                if adjacency_matrix[source_idx][destination_idx] != math.inf:
                    distance_matrix[source_idx][destination_idx] = adjacency_matrix[source_idx][destination_idx]
                    parent_matrix[source_idx][destination_idx] = source_idx
        
        # Floyd-Warshall algorithm: consider all intermediate nodes
        # Key insight: For each intermediate node k, update all pairs (i, j)
        for k in range(V): # Intermediate node k
            for i in range(V): # Source node i
                for j in range(V): # Destination node j
                    # Skip if no path through intermediate k
                    if (distance_matrix[i][k] != math.inf and distance_matrix[k][j] != math.inf):
                        new_distance = distance_matrix[i][k] + distance_matrix[k][j]
                        if distance_matrix[i][j] > new_distance:
                            distance_matrix[i][j] = new_distance
                            # Update parent: When going through intermediate k, the parent of destination 
                            # in the path source -> destination is the parent of destination in path intermediate -> destination
                            # This is the standard Floyd-Warshall parent update rule
                            parent_matrix[i][j] = parent_matrix[k][j]
        
        return distance_matrix, parent_matrix


class DirectedGraph(BaseGraph):

    def __init__(self):
        super().__init__()

    # ********* Graph Representation Based Methods *********
    def build_graph_from_edges(self, V: int, edges: list[tuple[int, int, float]]):
        self._adjacency_list: list[list[tuple[float, int]]] = [[] for _ in range(V)]
        for source, destination, weight in edges:
            self._adjacency_list[source].append((weight, destination))

    def build_graph_from_matrix(self, adjacency_matrix: list[list[float]]):
        V = len(adjacency_matrix)
        if V == 0 or any(len(row) != V for row in adjacency_matrix) or all(ele == math.inf for row in adjacency_matrix for ele in row):
            raise ValueError("Adjacency matrix must be non-empty and square")
        
        self._adjacency_list: list[list[tuple[float, int]]] = [[] for _ in range(V)]
        
        # Following Zero based node naming convention
        for i in range(V): # O(V^2) TC, O(V + E) SC
            for j in range(V):
                if adjacency_matrix[i][j] != math.inf and i != j:
                    weight = adjacency_matrix[i][j]
                    self._adjacency_list[i].append((weight, j))

    # ********* Node & Edge Operations *********
    def get_out_degree(self, node: int):
        return len(self._adjacency_list[node])

    def get_in_degree(self, node: int):
        in_degree_count = 0
        V = len(self._adjacency_list)
        
        for source_node in range(V):
            for weight, destination_node in self._adjacency_list[source_node]:
                if destination_node == node:
                    in_degree_count += 1
                    break
        
        return in_degree_count

    # ********* Graph Traversal Based Methods *********
    def is_cyclic_dfs(self):
        V = len(self._adjacency_list)
        visited: list[bool] = [False] * V
        # To track nodes currently on the call stack to detect back edges (cycles)
        nodes_on_call_stack: list[bool] = [False] * V
        def dfs_helper(current_node: int):
            visited[current_node] = True
            nodes_on_call_stack[current_node] = True
            for weight, neighbor in self._adjacency_list[current_node]:
                if not visited[neighbor]:
                    if dfs_helper(neighbor):
                        return True
                # If neighbor is on the call stack, we found a back edge (cycle)
                elif nodes_on_call_stack[neighbor]:
                    return True
            nodes_on_call_stack[current_node] = False
            return False

        # Handle disconnected graphs
        for start_node in range(V):
            if not visited[start_node]:
                if dfs_helper(start_node):
                    return True
        return False

    def topological_sort_kahn_s_bfs(self):
        queue: deque[int] = deque()
        
        V = len(self._adjacency_list)
        in_degree: list[int] = [0] * V
        topological_order: list[int] = []
        
        # Calculate in-degrees of all nodes => O(V + E) TC
        for node in range(V):
            for weight, neighbor in self._adjacency_list[node]:
                in_degree[neighbor] += 1
        
        # Initialize queue with nodes having in-degree 0 (handles disconnected graphs) => O(V) TC
        for node in range(V):
            if in_degree[node] == 0:
                queue.append(node)
        
        # Process nodes in topological order using BFS => O(V + E) TC
        while queue:
            current_node = queue.popleft()
            topological_order.append(current_node)
            for weight, neighbor in self._adjacency_list[current_node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If all nodes were processed, the graph is a DAG
        if len(topological_order) == V:
            return topological_order
        
        return None

    # RARELY ASKED
    def kosaraju(self):
        # Create reversed adjacency list without modifying the original graph
        V = len(self._adjacency_list)
        rev_adj_lst: list[list[tuple[float, int]]] = [[] for _ in range(V)]
        for node in range(V):
            for weight, neighbor in self._adjacency_list[node]:
                rev_adj_lst[neighbor].append((weight, node))
        
        # First DFS: find finish times on original graph
        visited: list[bool] = [False] * V
        finish_order: list[int] = []
        def dfs1(current_node: int):
            visited[current_node] = True
            for weight, neighbor in self._adjacency_list[current_node]:
                if not visited[neighbor]:
                    dfs1(neighbor)
            finish_order.append(current_node)
        
        for node in range(V):
            if not visited[node]:
                dfs1(node)
        
        # Second DFS: traverse in reverse finish order on reversed graph
        visited: list[bool] = [False] * V
        strongly_connected_components: list[list[int]] = []
        def dfs2(current_node: int):
            visited[current_node] = True
            strongly_connected_components[-1].append(current_node)
            for weight, neighbor in rev_adj_lst[current_node]:
                if not visited[neighbor]:
                    dfs2(neighbor)
        
        for start_node in finish_order[::-1]:
            if not visited[start_node]:
                strongly_connected_components.append([])
                dfs2(start_node)

        return strongly_connected_components

    # RARELY ASKED ********* SSSP Methods *********
    def bellman_ford(self, source_node: int, destination_node: int):
        V = len(self._adjacency_list)
        if V == 0:
            return
        
        if source_node == destination_node:
            return [source_node], 0.0

        parent: list[int] = [-1] * V
        distance: list[float] = [math.inf] * V

        distance[source_node] = 0  
        parent[source_node] = -1  

        # Bellman-Ford algorithm: relax edges V-1 times - O(V * E) TC
        # Key insight: After V-1 iterations, all shortest paths should be found
        # In the V-th iteration, if any edge is relaxed => a negative cycle exists
        for iteration in range(V):  # O(V-1) iterations for relaxation
            for node in range(V):  # O(V * E)
                if distance[node] != math.inf:
                    for weight, neighbor in self._adjacency_list[node]:  # O(degree(node)) = O(E) total
                        new_distance = distance[node] + weight 
                        if new_distance < distance[neighbor]:  
                            distance[neighbor] = new_distance 
                            parent[neighbor] = node 
                            # If we can still relax after V-1 iterations, a negative cycle exists
                            if iteration == V - 1:
                                return None
        
        # Reconstruct path using helper method - O(V) worst case
        # Check if destination is reachable (distance is finite)
        if distance[destination_node] == math.inf:
            return [], math.inf
        
        shortest_path = self._reconstruct_path(source_node, destination_node, parent)  # O(V)
        if not shortest_path:
            return [], math.inf

        return shortest_path, distance[destination_node]


class DisjointSet:

    def __init__(self, num_of_nodes: int):
        self.parent: list[int] = list(range(num_of_nodes))
        self.size: list[int] = [1] * num_of_nodes

    def find_ultimate_parent(self, node: int):
        # Base case: node is its own parent (root)
        if self.parent[node] == node:
            return node
        
        # Path compression: make parent point directly to root
        # This flattens the tree structure for future lookups
        self.parent[node] = self.find_ultimate_parent(self.parent[node])
        return self.parent[node]

    def union_by_size(self, u: int, v: int):
        # Get the ultimate parent (roots) of both nodes
        root_u = self.find_ultimate_parent(u)
        root_v = self.find_ultimate_parent(v)

        # Return if nodes already belong to the same component
        if root_u == root_v:
            return

        # Union by size: attach smaller component to larger component
        # This keeps the tree height minimal
        if self.size[root_u] < self.size[root_v]:
            self.parent[root_u] = root_v
            self.size[root_v] += self.size[root_u]
        else:
            self.parent[root_v] = root_u
            self.size[root_u] += self.size[root_v]


class UndirectedGraph(BaseGraph):

    def __init__(self):
        super().__init__()

    # ********* Graph Representation Based Methods *********
    def build_graph_from_edges(self, V: int, edges: list[tuple[int, int, float]]):
        self._adjacency_list: list[list[tuple[float, int]]] = [[] for _ in range(V)]
        for source, destination, weight in edges:
            self._adjacency_list[source].append((weight, destination))
            self._adjacency_list[destination].append((weight, source))

    def build_graph_from_matrix(self, adjacency_matrix: list[list[float]]):
        V = len(adjacency_matrix)
        if V == 0 or any(len(row) != V for row in adjacency_matrix) or all(ele == math.inf for row in adjacency_matrix for ele in row):
            raise ValueError("Adjacency matrix must be non-empty and square")
        
        self._adjacency_list: list[list[tuple[float, int]]] = [[] for _ in range(V)]
        
        # Following Zero based node naming convention
        for i in range(V): # O(V^2 / 2) TC, O(V + E) SC
            for j in range(i + 1, V):
                if adjacency_matrix[i][j] != math.inf and i != j:
                    weight = adjacency_matrix[i][j]
                    self._adjacency_list[i].append((weight, j))
                    self._adjacency_list[j].append((weight, i))

    # ********* Node & Edge Operations *********
    def get_degree(self, node: int):
        # For undirected graphs, degree equals the number of neighbors
        return len(self._adjacency_list[node])

    # ********* Graph Traversal Based Methods *********
    def is_cyclic_dfs(self):
        V = len(self._adjacency_list)
        visited: list[bool] = [False] * V
        def dfs_helper(current_node: int, parent_node: int):
            visited[current_node] = True
            for weight, neighbor in self._adjacency_list[current_node]:
                if not visited[neighbor]:
                    if dfs_helper(neighbor, current_node):
                        return True
                # If neighbor is already visited and is not the parent of the current node =>we found a back edge (cycle)
                elif neighbor != parent_node:
                    return True
            return False

        # Handle disconnected graphs
        for start_node in range(V):
            if not visited[start_node]:
                if dfs_helper(start_node, -1):
                    return True
        return False

    def is_bipartite_bfs(self):
        V = len(self._adjacency_list)
        colors: list[bool | None] = [None] * V  # Also serves as visited set
        queue: deque[int] = deque()

        # Handle disconnected graphs
        for start_node in range(V):
            if colors[start_node] is None:
                queue.append(start_node)
                colors[start_node] = True

                # BFS on a new connected component
                while queue:
                    current_node = queue.popleft()
                    for weight, neighbor in self._adjacency_list[current_node]:
                        if colors[neighbor] is None:
                            colors[neighbor] = not colors[current_node]
                            queue.append(neighbor)
                        # If neighbor has same color as current node, graph is not bipartite
                        elif colors[neighbor] == colors[current_node]:
                            return False
        
        return True

    # ********* Minimum Spanning Tree Methods *********
    def mst_prim_s_algorithm(self):
        V = len(self._adjacency_list)
        if V == 0:
            return [], 0.0
        
        visited: list[bool] = [False] * V
        priority_queue: list[tuple[float, int, int]] = [] # (edge_weight, source_node, destination_node)
        mst_weight: float = 0.0
        mst_edges: list[tuple[float, int, int]] = []
        
        # Start with an arbitrary node (use dummy parent -1 for the start node)
        start_node: int = 0
        heapq.heappush(priority_queue, (0, -1, start_node))
        
        while priority_queue: # O(E) iterations
            # Pop the cheapest edge connecting the MST to a new node (TC: O(log E) per iteration)
            current_weight, parent_node, current_node = heapq.heappop(priority_queue)
            
            if not visited[current_node]:
                visited[current_node] = True
                mst_weight += current_weight
                if parent_node != -1:
                    mst_edges.append((current_weight, parent_node, current_node))
                
                # Add all adjacent edges from this newly visited node
                for weight, neighbor in self._adjacency_list[current_node]:
                    if not visited[neighbor]:
                        # TC: O(log E) per push operation
                        heapq.heappush(priority_queue, (weight, current_node, neighbor))
        
        return mst_edges, mst_weight

    def mst_kruskal_s_algorithm(self):
        V = len(self._adjacency_list)
        if V == 0:
            return [], 0.0
        
        # Only include edges where source < destination to avoid duplicates in undirected graphs
        # This ensures each edge is processed exactly once instead of twice
        sorted_edges: list[tuple[float, int, int]] = sorted(
            [
                (weight, source, destination)
                for source, neighbors in enumerate(self._adjacency_list) 
                for weight, destination in neighbors
                if source < destination  # Only process each edge once
            ], 
            key=lambda edge: edge[0]
        )  # O(E log E) TC, O(E) SC
        mst_weight: float = 0.0
        mst_edges: list[tuple[float, int, int]] = []
        
        disjoint_set: DisjointSet = DisjointSet(V)

        for weight, source, destination in sorted_edges:  # O(E * 4α) TC
            # Add edge if it doesn't create a cycle (nodes are in different components)
            source_root: int = disjoint_set.find_ultimate_parent(source)
            destination_root: int = disjoint_set.find_ultimate_parent(destination)
            if source_root != destination_root:
                mst_weight += weight
                mst_edges.append((weight, source, destination))
                disjoint_set.union_by_size(source, destination)
        
        return mst_edges, mst_weight

    # RARELY ASKED ********* Critical Connections, Articulation Points Methods *********
    def get_bridges(self):
        V = len(self._adjacency_list)
        visited: list[bool] = [False] * V
        bridges: list[tuple[float, int, int]] = []
        discovery_time: list[int] = [0] * V  # To store the discovery time of nodes
        low_dis_time: list[int] = [0] * V  # To store the lowest discovery time of the nodes
        timer: int = 1  # To keep track of the time of insertion of nodes
        def dfs_helper(current_node: int, parent_node: int):
            nonlocal timer
            visited[current_node] = True
            discovery_time[current_node] = low_dis_time[current_node] = timer
            timer += 1

            for weight, neighbor in self._adjacency_list[current_node]:
                if not visited[neighbor]:
                    dfs_helper(neighbor, current_node)
                    low_dis_time[current_node] = min(low_dis_time[current_node], low_dis_time[neighbor])
                    # If the lowest time of insertion of the current_node is > the time of insertion of the neighbor => The edge represents a bridge
                    if low_dis_time[neighbor] > discovery_time[current_node]:
                        bridges.append((weight, current_node, neighbor))
                elif neighbor != parent_node:
                    low_dis_time[current_node] = min(low_dis_time[current_node], discovery_time[neighbor])
        
        # Start DFS traversal from all nodes in the graph to handle disconnected graphs
        for start_node in range(V):
            if not visited[start_node]:
                dfs_helper(start_node, -1)
        
        return bridges

    def get_articulation_points(self):
        V = len(self._adjacency_list)
        visited: list[bool] = [False] * V
        articulation_points: set[int] = set()
        discovery_time: list[int] = [-1] * V
        low_dis_time: list[int] = [-1] * V
        timer: int = 1
        def dfs_helper(current_node: int, parent_node: int):
            nonlocal timer
            visited[current_node] = True
            discovery_time[current_node] = low_dis_time[current_node] = timer
            timer += 1
            children: int = 0  # To count the number of children of the current_node
            for weight, neighbor in self._adjacency_list[current_node]:
                if not visited[neighbor]:
                    dfs_helper(neighbor, current_node)
                    # Update the lowest time of insertion for the current_node 
                    low_dis_time[current_node] = min(low_dis_time[current_node], low_dis_time[neighbor])
                    # If the lowest time of insertion of the current_node is found to be greater than the time of insertion of the neighbor and it is not the starting node
                    if low_dis_time[neighbor] >= discovery_time[current_node] and parent_node != -1:
                        # Mark the current_node as an articulation point
                        articulation_points.add(current_node)
                    children += 1
                elif neighbor != parent_node:
                    low_dis_time[current_node] = min(low_dis_time[current_node], discovery_time[neighbor])
            # If the current_node is a starting node and has more than one child 
            if parent_node == -1 and children > 1:
                # Mark the current_node as an articulation point
                articulation_points.add(current_node)
        
        # Start DFS traversal from all nodes in the graph to handle disconnected graphs
        for start_node in range(V):
            if not visited[start_node]:
                dfs_helper(start_node, -1)
        
        # sorted can handle any iterable and sets also
        return sorted(articulation_points)
