import math
import heapq
from enum import Enum
from collections import deque


class GraphType(Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"


class BaseGraph:

    def __init__(self, graph_type: GraphType):
        self.graph_type: GraphType = graph_type
        self.V: int = 0
        self.edges: list[tuple[float, int, int]] = []

    # ********* Graph Representation Based Methods *********
    def build_graph_from_matrix(self, adjacency_matrix: list[list[float]]):
        """
            Builds adjacency list representation from an adjacency matrix. Converts 2D matrix to adjacency list format.

            Args:
                adjacency_matrix: 2D matrix where matrix[i][j] represents edge weight from node i to node j

            Returns:
                None (modifies self._adjacency_list)

            Time Complexity: O(V^2)
            Space Complexity: O(V + E)

            Edge Cases:
                - Empty graph (V = 0)
                - Self-loops (handled by skipping i == j in directed graphs)
                - No parallel edges (matrix representation doesn't support them)
        """
        # Set V from matrix size and validate matrix is square
        self.V = len(adjacency_matrix)
        if self.V > 0 and any(len(row) != self.V for row in adjacency_matrix):
            raise ValueError("Adjacency matrix must be square")
        
        self._adjacency_list: list[list[tuple[float, int]]] = [[] for _ in range(self.V)]
        
        # Following Zero based node naming convention
        if self.graph_type == GraphType.UNDIRECTED:
            for i in range(self.V): # O(V^2 / 2) TC, O(V + E) SC
                for j in range(i + 1, self.V):
                    if adjacency_matrix[i][j] != math.inf:
                        self._adjacency_list[i].append((adjacency_matrix[i][j], j))
                        self._adjacency_list[j].append((adjacency_matrix[i][j], i))
        else:
            for i in range(self.V): # O(V^2) TC, O(V + E) SC
                for j in range(self.V):
                    if adjacency_matrix[i][j] != math.inf and i != j:
                        self._adjacency_list[i].append((adjacency_matrix[i][j], j)) 

    # ********* Graph Traversals *********
    def bfs(self):
        """
            Performs Breadth First Search (BFS) traversal. Explores graph level by level using a queue, similar to level-order tree traversal.

            Returns:
                tuple: (parent, graph_components) where parent[i] = j means node i's parent is j (-1 for roots), and graph_components is list of traversal orders per connected component

            Time Complexity: O(V + E)
            Space Complexity: O(V)

            Edge Cases:
                - Empty graph
                - Disconnected graphs (handles all components)
                - Single node graph
        """
        visited: list[bool] = [False] * self.V
        parent: list[int] = [-1] * self.V  
        graph_components: list[list[int]] = []  
        queue: deque[int] = deque()
        
        # Handling Disconnected Graphs => O(V + E) TC
        for start_node in range(self.V):
            if not visited[start_node]:
                visited[start_node] = True
                graph_components.append([]) 
                queue.append(start_node) 
                
                # BFS on a new connected component
                while queue:
                    # BFS traversal on current_node is started
                    current_node = queue.popleft()  
                    for weight, neighbor in self._adjacency_list[current_node]:  # O(degree(current_node))
                        if not visited[neighbor]:
                            visited[neighbor] = True # node visited but not traversed yet
                            parent[neighbor] = current_node  
                            queue.append(neighbor)  
                    # BFS on current_node is completed
                    graph_components[-1].append(current_node) 
        
        return parent, graph_components

    def dfs(self):
        """
            Performs Depth First Search (DFS) traversal recursively. Explores as deep as possible before backtracking, similar to pre-order tree traversal.

            Returns:
                tuple: (parent, graph_components) where parent[i] = j means node i's parent is j (-1 for roots), and graph_components is list of traversal orders per connected component

            Time Complexity: O(V + E)
            Space Complexity: O(V) + recursion stack O(V)

            Edge Cases:
                - Empty graph
                - Disconnected graphs (handles all components)
                - Single node graph
                - Deep graphs (stack overflow risk for very deep graphs)
        """
        visited: list[bool] = [False] * self.V
        parent: list[int] = [-1] * self.V
        graph_components: list[list[int]] = []
        
        def dfs_helper(current_node: int, parent_node: int):
            visited[current_node] = True
            parent[current_node] = parent_node
            graph_components[-1].append(current_node) # Pre-Order Traversal
            # DFS traversal on current_node started
            for weight, neighbor in self._adjacency_list[current_node]:
                if not visited[neighbor]:
                    dfs_helper(neighbor, current_node)
            # DFS traversal on current_node completed
            # graph_components[-1].append(current_node) # Post-Order Traversal

        # Handling Disconnected Graphs => O(V + E) TC
        for start_node in range(self.V):
            if not visited[start_node]:
                graph_components.append([])
                dfs_helper(start_node, -1)
        
        return parent, graph_components

    # # BaseGraph: Redundant method, use dfs instead
    # def dfs_iterative(self):
    #     """
    #         Performs Depth First Search (DFS) traversal iteratively using a stack. Explores as deep as possible before backtracking, similar to pre-order tree traversal.

    #         Returns:
    #             tuple: (parent, graph_components) where parent[i] = j means node i's parent is j (-1 for roots), and graph_components is list of traversal orders per connected component

    #         Time Complexity: O(V + E)
    #         Space Complexity: O(V)

    #         Edge Cases:
    #             - Empty graph
    #             - Disconnected graphs (handles all components)
    #             - Single node graph
    #     """
    #     visited: list[bool] = [False] * self.V
    #     parent: list[int] = [-1] * self.V  
    #     graph_components: list[list[int]] = []  
    #     stack: list[int] = []

    #     # Handling Disconnected Graphs => O(V + E) TC
    #     for start_node in range(self.V):
    #         if not visited[start_node]:
    #             visited[start_node] = True
    #             graph_components.append([])
    #             stack.append(start_node)

    #             # DFS on a new connected component
    #             while stack:
    #                 # DFS traversal on current_node is started
    #                 current_node = stack.pop() 
    #                 graph_components[-1].append(current_node) 
    #                 for weight, neighbor in self._adjacency_list[current_node]: # O(degree(current_node))
    #                     if not visited[neighbor]:
    #                         visited[neighbor] = True  # node visited but not traversed yet
    #                         parent[neighbor] = current_node 
    #                         stack.append(neighbor) 
    #                 # DFS on current_node is not completed yet
        
    #     return parent, graph_components

    # ********* SSSP Methods *********
    def _reconstruct_path(self, source_node: int, destination_node: int, visited: list[bool], parent: list[int]):
        # If destination is not reachable from source
        if not visited[destination_node]:
            return []
        
        # Reconstruct path by following parent pointers
        path: list[int] = []
        current_node = destination_node
        while current_node != source_node:
            path.append(current_node)
            current_node = parent[current_node]
        path.append(source_node)
        path.reverse()
        return path

    def sssp_bfs(self, source_node: int, destination_node: int):
        """
            Finds shortest path in unweighted or equally weighted graphs using BFS. BFS guarantees shortest path by exploring level by level.

            Args:
                source_node: Starting node
                destination_node: Target node
                
            Returns:
                tuple: (shortest_path, shortest_path_distance) or ([], math.inf) if no path exists

            Time Complexity: O(V + E)
            Space Complexity: O(V)

            Edge Cases:
                - Source equals destination (returns [source], 0)
                - No path exists (returns [], math.inf)
                - Graph with negative weights (incorrect results, use Dijkstra/Bellman-Ford)
                - Unequal edge weights (incorrect results, use Dijkstra)
        """
        if source_node == destination_node:
            return [source_node], 0.0

        visited: list[bool] = [False] * self.V
        parent: list[int] = [-1] * self.V
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
        shortest_path = self._reconstruct_path(source_node, destination_node, visited, parent)
        if not shortest_path:
            return [], math.inf
        
        # For unweighted graphs, distance = number of edges = len(path) - 1
        # For equally weighted graphs, we need to get the weight from an edge
        # Since BFS doesn't track edge weights, we'll use path length - 1 (assumes weight 1 per edge)
        # For actual weighted graphs, use Dijkstra's algorithm
        return shortest_path, len(shortest_path) - 1

    def sssp_dijkstra(self, source_node: int, destination_node: int):
        """
            Finds shortest path using Dijkstra's algorithm. Greedy algorithm that processes closest unvisited node first using priority queue. Works only with non-negative edge weights.

            Args:
                source_node: Starting node
                destination_node: Target node
                
            Returns:
                tuple: (shortest_path, shortest_path_distance) or ([], math.inf) if no path exists
                    
            Time Complexity: O((V + E) log V)
            Space Complexity: O(V + E)

            Edge Cases:
                - Source equals destination (returns [source], 0)
                - No path exists (returns [], math.inf)
                - Negative edge weights (incorrect results, use Bellman-Ford)
                - Disconnected graph (handles correctly)
        """
        if source_node == destination_node:
            return [source_node], 0.0

        visited: list[bool] = [False] * self.V
        parent: list[int] = [-1] * self.V
        distance: list[float] = [math.inf] * self.V
        priority_queue: list[tuple[float, int]] = []
        
        parent[source_node] = -1  
        distance[source_node] = 0  
        heapq.heappush(priority_queue, (0, source_node))  # O(log V)

        while priority_queue:  # O(V) iterations (each node processed at most once)
            current_distance, current_node = heapq.heappop(priority_queue)  # O(log V)
            
            # Skip if already processed (redundant entry in priority queue)
            if not visited[current_node]:  # O(1)
                # Mark node as visited after popping from queue (before processing)
                visited[current_node] = True
                # Early termination optimization if destination is reached
                if current_node == destination_node:
                    break
                # Relax edges from current node - O(degree(current_node))
                for weight, neighbor in self._adjacency_list[current_node]:  # O(degree(current_node))
                    if not visited[neighbor]:  # O(1)
                        new_distance = current_distance + weight
                        if new_distance < distance[neighbor]:
                            parent[neighbor] = current_node
                            distance[neighbor] = new_distance
                            heapq.heappush(priority_queue, (new_distance, neighbor))  # O(log V)
        
        # Reconstruct path using helper method - O(path_length) = O(V) worst case
        shortest_path = self._reconstruct_path(source_node, destination_node, visited, parent)  # O(V)
        if not shortest_path:
            return [], math.inf

        return shortest_path, distance[destination_node]

    # # RARELY ASKED ********* APSP Methods *********
    # def floyd_warshall(self, adjacency_matrix: list[list[float]]):
    #     """
    #         Finds all-pairs shortest paths using Floyd-Warshall algorithm. Dynamic programming approach that considers all intermediate nodes. Handles negative edge weights but not negative cycles.

    #         Args:
    #             adjacency_matrix: 2D matrix where matrix[i][j] represents edge weight from node i to node j

    #         Returns:
    #             tuple: (distance_matrix, parent_matrix) or ([], []) if graph is empty

    #         Time Complexity: O(V^3)
    #         Space Complexity: O(V^2)

    #         Edge Cases:
    #             - Empty graph (returns [], [])
    #             - Negative cycles (doesn't detect, may produce incorrect results)
    #             - Self-loops (handled with 0 distance on diagonal)
    #             - No path between nodes (math.inf in distance matrix)
    #     """
    #     if self.V == 0:
    #         return [], []
        
    #     # Initialize distance matrix: 0 for diagonal (self-loops), infinity else where => O(V^2) TC, O(V^2) SC
    #     distance_matrix: list[list[float]] = [
    #         [0.0 if row == column else math.inf for column in range(self.V)] 
    #         for row in range(self.V)
    #     ]
    #     # Initialize parent matrix: None indicates no path, diagonal is None (no node is parent for itself) => O(V^2) TC, O(V^2) SC
    #     parent_matrix: list[list[int | None]] = [[None] * self.V for _ in range(self.V)]

    #     # Fill distance matrix with direct edge weights
    #     if self.graph_type == GraphType.UNDIRECTED:
    #         for source_idx in range(self.V): # O(V^2 / 2) TC
    #             for destination_idx in range(source_idx + 1, self.V):
    #                 if adjacency_matrix[source_idx][destination_idx] != math.inf:
    #                     distance_matrix[source_idx][destination_idx] = adjacency_matrix[source_idx][destination_idx]
    #                     distance_matrix[destination_idx][source_idx] = adjacency_matrix[source_idx][destination_idx]
    #                     parent_matrix[source_idx][destination_idx] = source_idx
    #                     parent_matrix[destination_idx][source_idx] = destination_idx
    #     else:
    #         for source_idx in range(self.V): # O(V^2) TC
    #             for destination_idx in range(self.V):
    #                 if adjacency_matrix[source_idx][destination_idx] != math.inf:
    #                     distance_matrix[source_idx][destination_idx] = adjacency_matrix[source_idx][destination_idx]
    #                     parent_matrix[source_idx][destination_idx] = source_idx
        
    #     # Floyd-Warshall algorithm: consider all intermediate nodes
    #     # Key insight: For each intermediate node k, update all pairs (i, j)
    #     for k in range(self.V): # Intermediate node k
    #         for i in range(self.V): # Source node i
    #             for j in range(self.V): # Destination node j
    #                 # Skip if no path through intermediate k
    #                 if (distance_matrix[i][k] != math.inf and distance_matrix[k][j] != math.inf):
    #                     new_distance = distance_matrix[i][k] + distance_matrix[k][j]
    #                     if distance_matrix[i][j] > new_distance:
    #                         distance_matrix[i][j] = new_distance
    #                         # Update parent: When going through intermediate k, the parent of destination 
    #                         # in the path source -> destination is the parent of destination in path intermediate -> destination
    #                         # This is the standard Floyd-Warshall parent update rule
    #                         parent_matrix[i][j] = parent_matrix[k][j]
        
    #     return distance_matrix, parent_matrix


class DirectedGraph(BaseGraph):

    def __init__(self):
        super().__init__(GraphType.DIRECTED)

    # ********* Node & Edge Operations *********
    def get_out_degree(self, node: int):
        """
            Returns the out-degree of a node (number of outgoing edges).

            Args:
                node: Node to get out-degree for

            Returns:
                int: Number of outgoing edges from the node

            Time Complexity: O(1)
            Space Complexity: O(1)

            Edge Cases:
                - Invalid node index (IndexError)
                - Node with no outgoing edges (returns 0)
        """
        return len(self._adjacency_list[node])

    def get_in_degree(self, node: int):
        """
            Returns the in-degree of a node (number of incoming edges) by counting edges pointing to the node.

            Args:
                node: Node to get in-degree for

            Returns:
                int: Number of incoming edges to the node

            Time Complexity: O(V + E)
            Space Complexity: O(1)

            Edge Cases:
                - Invalid node index (IndexError)
                - Node with no incoming edges (returns 0)
        """
        in_degree_count = 0
        
        for current_node in range(self.V):
            for weight, neighbor in self._adjacency_list[current_node]:
                if neighbor == node:
                    in_degree_count += 1
        return in_degree_count

    # ********* Graph Traversal Based Methods *********
    def is_cyclic_dfs(self):
        """
            Checks if directed graph contains a cycle using DFS. Detects back edges by tracking nodes on recursion stack.

            Returns:
                bool: True if cycle exists, False otherwise

            Time Complexity: O(V + E)
            Space Complexity: O(V) + recursion stack O(V)

            Edge Cases:
                - Empty graph (returns False)
                - Disconnected graphs (checks all components)
                - Self-loops (detected as cycles)
                - Deep graphs (stack overflow risk)
        """
        visited: list[bool] = [False] * self.V
        # To track nodes currently on the call stack to detect back edges (cycles)
        nodes_on_call_stack: list[bool] = [False] * self.V
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
        for start_node in range(self.V):
            if not visited[start_node]:
                if dfs_helper(start_node):
                    return True
        return False

    def topological_sort_kahn_s_bfs(self):
        """
            Performs topological sorting using Kahn's algorithm. Processes nodes with in-degree 0 first using BFS.

            Returns:
                list[int]: Topologically sorted nodes, or None if graph contains a cycle

            Time Complexity: O(V + E)
            Space Complexity: O(V)

            Edge Cases:
                - Empty graph (returns [])
                - Graph with cycle (returns None)
                - Disconnected DAG (handles all components)
                - Single node (returns [0])
        """
        queue: deque[int] = deque()
        
        in_degree: list[int] = [0] * self.V
        topological_order: list[int] = []
        
        # Calculate in-degrees of all nodes => O(V + E) TC
        for node in range(self.V):
            for weight, neighbor in self._adjacency_list[node]:
                in_degree[neighbor] += 1
        
        # Initialize queue with nodes having in-degree 0 (handles disconnected graphs) => O(V) TC
        for node in range(self.V):
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
        if len(topological_order) == self.V:
            return topological_order
        
        return None

    # # RARELY ASKED
    # def kosaraju(self):
    #     """
    #         Finds all strongly connected components (SCCs) using Kosaraju's algorithm. Two-pass DFS: first on original graph to get finish times, then on reversed graph in reverse finish order.

    #         Returns:
    #             list: List of lists, where each inner list contains nodes in one SCC

    #         Time Complexity: O(V + E)
    #         Space Complexity: O(V + E)

    #         Edge Cases:
    #             - Empty graph (returns [])
    #             - Disconnected graph (finds SCCs in each component)
    #             - Graph with no cycles (each node is its own SCC)
    #     """
    #     # Create reversed adjacency list without modifying the original graph
    #     rev_adj_lst: list[list[tuple[float, int]]] = [[] for _ in range(self.V)]
    #     for node in range(self.V):
    #         for weight, neighbor in self._adjacency_list[node]:
    #             rev_adj_lst[neighbor].append((weight, node))
        
    #     # First DFS: find finish times on original graph
    #     visited: list[bool] = [False] * self.V
    #     finish_order: list[int] = []
    #     def dfs1(current_node: int):
    #         visited[current_node] = True
    #         for weight, neighbor in self._adjacency_list[current_node]:
    #             if not visited[neighbor]:
    #                 dfs1(neighbor)
    #         finish_order.append(current_node)
        
    #     for node in range(self.V):
    #         if not visited[node]:
    #             dfs1(node)
        
    #     # Second DFS: traverse in reverse finish order on reversed graph
    #     visited: list[bool] = [False] * self.V
    #     strongly_connected_components: list[list[int]] = []
    #     def dfs2(current_node: int):
    #         visited[current_node] = True
    #         strongly_connected_components[-1].append(current_node)
    #         for weight, neighbor in rev_adj_lst[current_node]:
    #             if not visited[neighbor]:
    #                 dfs2(neighbor)
        
    #     for start_node in finish_order[::-1]:
    #         if not visited[start_node]:
    #             strongly_connected_components.append([])
    #             dfs2(start_node)

    #     return strongly_connected_components

    # # RARELY ASKED ********* SSSP Methods *********
    # def bellman_ford(self, source_node: int, destination_node: int):
    #     """
    #         Finds shortest path using Bellman-Ford algorithm. Relaxes all edges V-1 times. Can handle negative weights and detects negative cycles.

    #         Args:
    #             source_node: Starting node
    #             destination_node: Target node
            
    #         Returns:
    #             tuple: (shortest_path, shortest_path_distance) if path exists and no negative cycle, ([], math.inf) if no path, None if negative cycle detected

    #         Time Complexity: O(V * E)
    #         Space Complexity: O(V)

    #         Edge Cases:
    #             - Source equals destination (returns [source], 0)
    #             - No path exists (returns [], math.inf)
    #             - Negative cycle reachable from source (returns None)
    #             - Negative weights without cycles (handles correctly)
    #     """
    #     if source_node == destination_node:
    #         return [source_node], 0.0

    #     parent: list[int] = [-1] * self.V
    #     distance: list[float] = [math.inf] * self.V

    #     distance[source_node] = 0  
    #     parent[source_node] = -1  

    #     # Bellman-Ford algorithm: relax edges V-1 times - O(V * E) TC
    #     # Key insight: After V-1 iterations, all shortest paths should be found
    #     # In the V-th iteration, if any edge is relaxed => a negative cycle exists
    #     for iteration in range(self.V):  # O(V-1) iterations for relaxation
    #         for node in range(self.V):  # O(V * E)
    #             if distance[node] != math.inf:
    #                 for weight, neighbor in self._adjacency_list[node]:  # O(degree(node)) = O(E) total
    #                     new_distance = distance[node] + weight 
    #                     if new_distance < distance[neighbor]:  
    #                         distance[neighbor] = new_distance 
    #                         parent[neighbor] = node 
    #                         # If we can still relax after V-1 iterations, a negative cycle exists
    #                         if iteration == self.V - 1:
    #                             return None
        
    #     # Reconstruct path using helper method - O(V) worst case
    #     # Check if destination is reachable (distance is finite)
    #     if distance[destination_node] == math.inf:
    #         return [], math.inf
        
    #     # Create visited array based on reachability from source
    #     visited = [distance[i] != math.inf for i in range(self.V)]
    #     shortest_path = self._reconstruct_path(source_node, destination_node, visited, parent)  # O(V)
    #     if not shortest_path:
    #         return [], math.inf

    #     return shortest_path, distance[destination_node]


# class DisjointSet:

#     def __init__(self, num_of_nodes: int):
#         self.parent: list[int] = list(range(num_of_nodes))
#         self.size: list[int] = [1] * num_of_nodes

#     def find_ultimate_parent(self, node: int):
#         """
#             Finds the root of a node using path compression. Flattens tree structure for faster future lookups.

#             Args:
#                 node: Node to find root for
                
#             Returns:
#                 int: Root node of the component containing the given node
            
#             Time Complexity: O(α(V)) amortized (near constant)
#             Space Complexity: O(1) + recursion stack O(V)

#             Edge Cases:
#                 - Node is its own parent (returns itself)
#                 - Deep tree structures (path compression optimizes)
#         """
#         # Base case: node is its own parent (root)
#         if self.parent[node] == node:
#             return node
        
#         # Path compression: make parent point directly to root
#         # This flattens the tree structure for future lookups
#         self.parent[node] = self.find_ultimate_parent(self.parent[node])
#         return self.parent[node]

#     def union_by_size(self, u: int, v: int):
#         """
#             Unions two components using union by size. Attaches smaller component to larger one to minimize tree height.

#             Args:
#                 u: First node
#                 v: Second node

#             Returns:
#                 None

#             Time Complexity: O(α(V)) amortized (near constant)
#             Space Complexity: O(1) + recursion stack O(V)

#             Edge Cases:
#                 - Nodes already in same component (no-op)
#                 - Invalid node indices (IndexError)
#         """
#         # Get the ultimate parents (roots) of both nodes
#         root_u = self.find_ultimate_parent(u)
#         root_v = self.find_ultimate_parent(v)

#         # Return if nodes already belong to the same component
#         if root_u == root_v:
#             return

#         # Union by size: attach smaller component to larger component
#         # This keeps the tree height minimal
#         if self.size[root_u] < self.size[root_v]:
#             self.parent[root_u] = root_v
#             self.size[root_v] += self.size[root_u]
#         else:
#             self.parent[root_v] = root_u
#             self.size[root_u] += self.size[root_v]


class UndirectedGraph(BaseGraph):

    def __init__(self):
        super().__init__(GraphType.UNDIRECTED)

    # ********* Node & Edge Operations *********
    def get_degree(self, node: int):
        """
            Returns the degree of a node (number of edges incident to it). For undirected graphs, equals number of neighbors.

            Args:
                node: Node to get degree for

            Returns:
                int: Number of edges incident to the node

            Time Complexity: O(1)
            Space Complexity: O(1)

            Edge Cases:
                - Invalid node index (IndexError)
                - Isolated node (returns 0)
        """
        # For undirected graphs, degree equals the number of neighbors
        return len(self._adjacency_list[node])

    # ********* Graph Traversal Based Methods *********
    def is_cyclic_dfs(self):
        """
            Checks if undirected graph contains a cycle using DFS. Detects back edges (edges to already visited non-parent nodes).

            Returns:
                bool: True if cycle exists, False otherwise

            Time Complexity: O(V + E)
            Space Complexity: O(V) + recursion stack O(V)

            Edge Cases:
                - Empty graph (returns False)
                - Disconnected graphs (checks all components)
                - Tree structure (returns False)
                - Deep graphs (stack overflow risk)
        """
        visited: list[bool] = [False] * self.V
        def dfs_helper(current_node: int, parent_node: int):
            visited[current_node] = True
            for weight, neighbor in self._adjacency_list[current_node]:
                if not visited[neighbor]:
                    if dfs_helper(neighbor, current_node):
                        return True
                # If neighbor is already visited and is not the child of the current node, we found a back edge (cycle)
                elif neighbor != parent_node:
                    return True
            return False

        # Handle disconnected graphs
        for start_node in range(self.V):
            if not visited[start_node]:
                if dfs_helper(start_node, -1):
                    return True
        return False

    # def is_bipartite_bfs(self):
    #     """
    #         Checks if graph is bipartite using BFS coloring. Assigns colors to nodes and checks for conflicts (adjacent nodes with same color).

    #         Returns:
    #             bool: True if bipartite, False otherwise

    #         Time Complexity: O(V + E)
    #         Space Complexity: O(V)

    #         Edge Cases:
    #             - Empty graph (returns True)
    #             - Disconnected graphs (all components must be bipartite)
    #             - Odd-length cycles (returns False)
    #             - Tree structure (returns True)
    #     """
    #     colors: list[bool | None] = [None] * self.V  # Also serves as visited set
    #     queue: deque[int] = deque()

    #     # Handle disconnected graphs
    #     for start_node in range(self.V):
    #         if colors[start_node] is None:
    #             queue.append(start_node)
    #             colors[start_node] = True

    #             # BFS on a new connected component
    #             while queue:
    #                 current_node = queue.popleft()
    #                 for weight, neighbor in self._adjacency_list[current_node]:
    #                     if colors[neighbor] is None:
    #                         colors[neighbor] = not colors[current_node]
    #                         queue.append(neighbor)
    #                     # If neighbor has same color as current node, graph is not bipartite
    #                     elif colors[neighbor] == colors[current_node]:
    #                         return False
        
    #     return True

    # # ********* Minimum Spanning Tree Methods *********
    # def mst_prim_s_algorithm(self):
    #     """
    #         Finds Minimum Spanning Tree (MST) using Prim's algorithm. Grows MST by greedily adding cheapest edge connecting MST to unvisited vertex.

    #         Returns:
    #             tuple: (mst_edges, mst_weight) where mst_edges is list of (weight, source, dest) tuples

    #         Time Complexity: The Lazy Version - O(E log E) but The Eager Version - O(E log V)
    #         Space Complexity: O(E)

    #         Edge Cases:
    #             - Empty graph (returns ([], 0))
    #             - Disconnected graph (returns MST of first component only)
    #             - Single node (returns ([], 0))
    #             - Negative edge weights (works correctly)
    #     """
    #     if self.V == 0:
    #         return [], 0.0
        
    #     visited: list[bool] = [False] * self.V
    #     priority_queue: list[tuple[float, int, int]] = [] # (edge_weight, source_node, destination_node)
    #     mst_weight: float = 0.0
    #     mst_edges: list[tuple[float, int, int]] = []
        
    #     # Start with an arbitrary node (use dummy parent -1 for the start node)
    #     start_node: int = 0
    #     heapq.heappush(priority_queue, (0, -1, start_node))
        
    #     while priority_queue: # O(E) iterations
    #         # Pop the cheapest edge connecting the MST to a new node (TC: O(log E) per iteration)
    #         current_weight, parent_node, current_node = heapq.heappop(priority_queue)
            
    #         if not visited[current_node]:
    #             visited[current_node] = True
    #             mst_weight += current_weight
    #             if parent_node != -1:
    #                 mst_edges.append((current_weight, parent_node, current_node))
                
    #             # Add all adjacent edges from this newly visited node
    #             for weight, neighbor in self._adjacency_list[current_node]:
    #                 if not visited[neighbor]:
    #                     # TC: O(log E) per push operation
    #                     heapq.heappush(priority_queue, (weight, current_node, neighbor))
        
    #     return mst_edges, mst_weight

    # def mst_kruskal_s_algorithm(self):
    #     """
    #         Finds Minimum Spanning Tree (MST) using Kruskal's algorithm. Sorts edges by weight and adds them if they don't create cycles (using Union-Find).

    #         Returns:
    #             tuple: (mst_edges, mst_weight) where mst_edges is list of (weight, source, dest) tuples

    #         Time Complexity: O(E log E)
    #         Space Complexity: O(E)

    #         Edge Cases:
    #             - Empty graph (returns ([], 0))
    #             - Disconnected graph (returns MST of connected components)
    #             - Single node (returns ([], 0))
    #             - Negative edge weights (works correctly)
    #     """
    #     if self.V == 0:
    #         return [], 0.0
        
    #     # Only include edges where source < destination to avoid duplicates in undirected graphs
    #     # This ensures each edge is processed exactly once instead of twice
    #     sorted_edges: list[tuple[float, int, int]] = sorted(
    #         [
    #             (weight, source, destination)
    #             for source, neighbors in enumerate(self._adjacency_list) 
    #             for weight, destination in neighbors
    #             if source < destination  # Only process each edge once
    #         ], 
    #         key=lambda edge: edge[0]
    #     )  # O(E log E) TC, O(E) SC
    #     mst_weight: float = 0.0
    #     mst_edges: list[tuple[float, int, int]] = []
        
    #     disjoint_set: DisjointSet = DisjointSet(self.V)

    #     for weight, source, destination in sorted_edges:  # O(E * 4α) TC
    #         # Add edge if it doesn't create a cycle (nodes are in different components)
    #         source_root: int = disjoint_set.find_ultimate_parent(source)
    #         destination_root: int = disjoint_set.find_ultimate_parent(destination)
    #         if source_root != destination_root:
    #             mst_weight += weight
    #             mst_edges.append((weight, source, destination))
    #             disjoint_set.union_by_size(source, destination)
        
    #     return mst_edges, mst_weight

    # # RARELY ASKED ********* Critical Connections, Articulation Points Methods *********
    # def get_bridges(self):
    #     """
    #         Finds all bridges (critical connections) using Tarjan's algorithm. A bridge is an edge whose removal increases number of connected components.

    #         Returns:
    #             list: List of (weight, source, destination) tuples representing bridges

    #         Time Complexity: O(V + E)
    #         Space Complexity: O(V) + recursion stack O(V)

    #         Edge Cases:
    #             - Empty graph (returns [])
    #             - Disconnected graph (finds bridges in each component)
    #             - Tree structure (all edges are bridges)
    #             - Graph with no bridges (returns [])
    #     """
    #     visited: list[bool] = [False] * self.V
    #     bridges: list[tuple[float, int, int]] = []
    #     discovery_time: list[int] = [0] * self.V  # To store the discovery time of nodes
    #     low_dis_time: list[int] = [0] * self.V  # To store the lowest discovery time of the nodes
    #     timer: int = 1  # To keep track of the time of insertion of nodes
    #     def dfs_helper(current_node: int, parent_node: int):
    #         nonlocal timer
    #         visited[current_node] = True
    #         discovery_time[current_node] = low_dis_time[current_node] = timer
    #         timer += 1

    #         for weight, neighbor in self._adjacency_list[current_node]:
    #             if not visited[neighbor]:
    #                 dfs_helper(neighbor, current_node)
    #                 low_dis_time[current_node] = min(low_dis_time[current_node], low_dis_time[neighbor])
    #                 # If the lowest time of insertion of the current_node is > the time of insertion of the neighbor => The edge represents a bridge
    #                 if low_dis_time[neighbor] > discovery_time[current_node]:
    #                     bridges.append((weight, current_node, neighbor))
    #             elif neighbor != parent_node:
    #                 low_dis_time[current_node] = min(low_dis_time[current_node], discovery_time[neighbor])
        
    #     # Start DFS traversal from all nodes in the graph to handle disconnected graphs
    #     for start_node in range(self.V):
    #         if not visited[start_node]:
    #             dfs_helper(start_node, -1)
        
    #     return bridges

    # def get_articulation_points(self):
    #     """
    #         Finds all articulation points using Tarjan's algorithm. An articulation point is a node whose removal increases number of connected components.

    #         Returns:
    #             list: Sorted list of articulation point node indices

    #         Time Complexity: O(V + E)
    #         Space Complexity: O(V) + recursion stack O(V)

    #         Edge Cases:
    #             - Empty graph (returns [])
    #             - Disconnected graph (finds articulation points in each component)
    #             - Tree structure (all non-leaf nodes are articulation points)
    #             - Graph with no articulation points (returns [])
    #     """
    #     visited: list[bool] = [False] * self.V
    #     articulation_points: set[int] = set()
    #     discovery_time: list[int] = [-1] * self.V
    #     low_dis_time: list[int] = [-1] * self.V
    #     timer: int = 1
    #     def dfs_helper(current_node: int, parent_node: int):
    #         nonlocal timer
    #         visited[current_node] = True
    #         discovery_time[current_node] = low_dis_time[current_node] = timer
    #         timer += 1
    #         children: int = 0  # To count the number of children of the current_node
    #         for weight, neighbor in self._adjacency_list[current_node]:
    #             if not visited[neighbor]:
    #                 dfs_helper(neighbor, current_node)
    #                 # Update the lowest time of insertion for the current_node 
    #                 low_dis_time[current_node] = min(low_dis_time[current_node], low_dis_time[neighbor])
    #                 # If the lowest time of insertion of the current_node is found to be greater than the time of insertion of the neighbor and it is not the starting node
    #                 if low_dis_time[neighbor] >= discovery_time[current_node] and parent_node != -1:
    #                     # Mark the current_node as an articulation point
    #                     articulation_points.add(current_node)
    #                 children += 1
    #             elif neighbor != parent_node:
    #                 low_dis_time[current_node] = min(low_dis_time[current_node], discovery_time[neighbor])
    #         # If the current_node is a starting node and has more than one child 
    #         if parent_node == -1 and children > 1:
    #             # Mark the current_node as an articulation point
    #             articulation_points.add(current_node)
        
    #     # Start DFS traversal from all nodes in the graph to handle disconnected graphs
    #     for start_node in range(self.V):
    #         if not visited[start_node]:
    #             dfs_helper(start_node, -1)

    #     # sorted can handle any iterable and sets also
    #     return sorted(articulation_points)
