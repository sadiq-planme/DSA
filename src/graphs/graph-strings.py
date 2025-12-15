import math
import heapq
from enum import Enum
from collections import defaultdict, deque


class GraphType(Enum):
    DIRECTED = "directed"
    UNDIRECTED = "undirected"


class BaseGraph:

    def __init__(self, graph_type: GraphType):
        self.graph_type: GraphType = graph_type
        self._adjacency_list: defaultdict[str, list[tuple[float, str]]] = defaultdict(list)
        self._nodes: set[str] = set()
        self.edges: list[tuple[float, str, str]] = []

    def add_edge(self, source: str, destination: str, weight: float = 1.0):
        """
            Add an edge between source and destination nodes. Automatically adds nodes if they don't exist.
            For undirected graphs, adds bidirectional edges.
            
            Args:
                source: Source node
                destination: Destination node
                weight: Edge weight (default: 1.0)
            
            Time Complexity: O(1) amortized
            Space Complexity: O(1)
            
            Edge Cases: Handles new nodes automatically, duplicate edges allowed
        """
        # Add nodes if they don't exist
        self._nodes.add(source)
        self._nodes.add(destination)

        # Add edges to adjacency list
        self._adjacency_list[source].append((weight, destination))
        if self.graph_type == GraphType.UNDIRECTED:
            self._adjacency_list[destination].append((weight, source))

    # ********* Graph Traversals *********
    def bfs(self):
        """
            Performs Breadth First Search (BFS) traversal. Explores level by level, similar to level-order tree traversal.
            Handles disconnected graphs by traversing each connected component.
            
            Returns:
                tuple: (parent, graph_components) - parent dict mapping each node to its parent (None for roots),
                graph_components list where each element is traversal order of one connected component
            
            Time Complexity: O(V + E)
            Space Complexity: O(V)
            
            Edge Cases: Empty graph, disconnected graph, single node
        """
        visited: set[str] = set()
        parent: dict[str, str | None] = {}  
        graph_components: list[list[str]] = []  
        queue: deque[str] = deque()  

        # Handling Disconnected Graphs => O(V + E) TC
        for start_node in self._nodes:
            if start_node not in visited:  # O(1) average case
                visited.add(start_node)  # O(1) amortized
                parent[start_node] = None  
                graph_components.append([]) 
                queue.append(start_node) 
                
                # starting BFS on a new connected component
                while queue:
                    current_node = queue.popleft()  
                    # BFS traversal on current_node started
                    for weight, neighbor in self._adjacency_list.get(current_node, []):  # O(degree(current_node))
                        if neighbor not in visited:
                            visited.add(neighbor)  # node visited but not traversed yet => O(1) amortized
                            parent[neighbor] = current_node  
                            queue.append(neighbor)  
                    # BFS on current_node completed
                    graph_components[-1].append(current_node)
        
        return parent, graph_components

    # # BaseGraph: Redundant method, use dfs instead
    # def dfs_iterative(self):
    #     """
    #         Performs Depth First Search (DFS) traversal iteratively using a stack. Explores as deep as possible before backtracking.
    #         Handles disconnected graphs by traversing each connected component.
            
    #         Returns:
    #             tuple: (parent, graph_components) - parent dict mapping each node to its parent (None for roots),
    #             graph_components list where each element is traversal order of one connected component
            
    #         Time Complexity: O(V + E)
    #         Space Complexity: O(V)
            
    #         Edge Cases: Empty graph, disconnected graph, single node
    #     """
    #     visited: set[str] = set()
    #     parent: dict[str, str | None] = {}
    #     graph_components: list[list[str]] = []
    #     stack: list[str] = []

    #     # Handling Disconnected Graphs => O(V + E) TC
    #     for start_node in self._nodes:
    #         if start_node not in visited: # O(1) average case
    #             visited.add(start_node) # O(1) amortized
    #             parent[start_node] = None
    #             graph_components.append([])
    #             stack.append(start_node)

    #             # DFS on a new connected component
    #             while stack:
    #                 current_node = stack.pop() 
    #                 graph_components[-1].append(current_node) 
    #                 # DFS traversal on current_node is started
    #                 for weight, neighbor in self._adjacency_list.get(current_node, []): # O(degree(current_node))
    #                     if neighbor not in visited:
    #                         visited.add(neighbor) # node visited but not traversed yet
    #                         parent[neighbor] = current_node 
    #                         stack.append(neighbor) 
    #                 # DFS on current_node is not completed yet
        
    #     return parent, graph_components

    def dfs(self):
        """
            Performs Depth First Search (DFS) traversal recursively. Explores as deep as possible before backtracking.
            Handles disconnected graphs by traversing each connected component.
            
            Returns:
                tuple: (parent, graph_components) - parent dict mapping each node to its parent (None for roots),
                graph_components list where each element is traversal order of one connected component
            
            Time Complexity: O(V + E)
            Space Complexity: O(V) + recursion stack O(V) worst case
            
            Edge Cases: Empty graph, disconnected graph, single node, deep recursion may cause stack overflow
        """
        visited: set[str] = set()
        parent: dict[str, str | None] = {}
        graph_components: list[list[str]] = []
        def dfs_helper(current_node: str, parent_node: str | None):
            visited.add(current_node)
            parent[current_node] = parent_node
            graph_components[-1].append(current_node) # Pre-Order Traversal
            # DFS traversal on current_node started
            for weight, neighbor in self._adjacency_list.get(current_node, []):
                if neighbor not in visited:
                    dfs_helper(neighbor, current_node)
            # DFS traversal on current_node completed
            # graph_components[-1].append(current_node) # Post-Order Traversal

        # Handling Disconnected Graphs => O(V + E) TC
        for start_node in self._nodes:
            if start_node not in visited:
                graph_components.append([])
                dfs_helper(start_node, None)
        
        return parent, graph_components

    # ********* SSSP Methods *********
    def _reconstruct_path(self, source_node: str, destination_node: str, visited: set[str], parent: dict[str, str | None]):
        """
            Helper method to reconstruct path from source to destination using parent pointers.
            
            Args:
                source_node: Source node
                destination_node: Destination node
                visited: Set of visited nodes
                parent: Dictionary mapping each node to its parent
            
            Returns:
                list: Path from source to destination, empty list if no path exists
            
            Time Complexity: O(V) worst case
            Space Complexity: O(V)
            
            Edge Cases: Destination not reachable, source equals destination, invalid parent pointers
        """
        # If destination is not reachable from source
        if destination_node not in visited:
            return []
        
        # Reconstruct path by following parent pointers
        path: list[str] = []
        current_node = destination_node
        while current_node != source_node:
            path.append(current_node)
            current_node = parent.get(current_node)
        path.append(source_node)
        path.reverse()
        return path

    def sssp_bfs(self, source_node: str, destination_node: str):
        """
            Finds shortest path using BFS. Works only for unweighted or equally weighted graphs.
            BFS guarantees shortest path in unweighted graphs by exploring level by level.
            
            Args:
                source_node: Starting node
                destination_node: Target node
            
            Returns:
                tuple: (shortest_path, shortest_path_distance) - path as list of nodes and total distance,
                ([], math.inf) if no path exists
            
            Time Complexity: O(V + E)
            Space Complexity: O(V)
            
            Edge Cases: Source equals destination, no path exists, disconnected graph, only works for equal weights
        """
        if source_node == destination_node:
            return [source_node], 0.0

        visited: set[str] = set()
        parent: dict[str, str | None] = {}
        queue: deque[str] = deque()

        # Start BFS traversal from source node
        visited.add(source_node)  # O(1) amortized
        parent[source_node] = None 
        queue.append(source_node)
        
        while queue:
            current_node = queue.popleft()
            # Early termination if destination is reached
            if current_node == destination_node:
                break
            # BFS traversal on current_node started
            for weight, neighbor in self._adjacency_list.get(current_node, []): # O(degree(current_node))
                if neighbor not in visited: 
                    visited.add(neighbor)
                    parent[neighbor] = current_node 
                    queue.append(neighbor)
            # BFS traversal on current_node completed
        
        # Reconstruct path using helper method
        shortest_path: list[str] = self._reconstruct_path(source_node, destination_node, visited, parent)
        if not shortest_path:
            return [], math.inf
        
        # For unweighted graphs, distance is the number of edges (path length - 1)
        # BFS works for unweighted graphs, so we don't need to multiply by weight
        return shortest_path, float(len(shortest_path) - 1)

    def sssp_dijkstra(self, source_node: str, destination_node: str):
        """
            Finds shortest path using Dijkstra's algorithm. Greedy algorithm that always processes closest unvisited node first.
            Only works with non-negative edge weights. Uses priority queue for efficient node selection.
            
            Args:
                source_node: Starting node
                destination_node: Target node
            
            Returns:
                tuple: (shortest_path, shortest_path_distance) - path as list of nodes and total distance,
                ([], math.inf) if no path exists
            
            Time Complexity: O((V + E) log V) with binary heap
            Space Complexity: O(V)
            
            Edge Cases: Source equals destination, no path exists, negative weights (incorrect results), disconnected graph
        """
        if source_node == destination_node:
            return [source_node], 0.0

        visited: set[str] = set()  # grows to O(V)
        parent: dict[str, str | None] = {}  # grows to O(V)
        distance: defaultdict[str, float] = defaultdict(lambda: math.inf) # grows to O(V)
        priority_queue: list[tuple[float, str]] = []  # can grow to O(V + E) worst case
        
        parent[source_node] = None  
        distance[source_node] = 0  
        heapq.heappush(priority_queue, (0, source_node))  # O(log V)

        while priority_queue:  # O(V) iterations (each node processed at most once)
            current_distance, current_node = heapq.heappop(priority_queue)  # O(log V)
            
            # Skip if already processed (redundant entry in priority queue)
            if current_node not in visited:  # O(1) average case
                # Mark node as visited after popping from queue (before processing)
                visited.add(current_node)
                # Early termination optimization if destination is reached
                if current_node == destination_node:  # Early termination optimization
                    break
                # Relax edges from current node - O(degree(current_node))
                for weight, neighbor in self._adjacency_list.get(current_node, []):  # O(degree(current_node))
                    if neighbor not in visited:  # O(1) average case
                        new_distance = current_distance + weight
                        if new_distance < distance[neighbor]:
                            parent[neighbor] = current_node
                            distance[neighbor] = new_distance
                            heapq.heappush(priority_queue, (new_distance, neighbor))  # O(log V)
        
        # Reconstruct path using helper method - O(path_length) = O(V) worst case
        shortest_path: list[str] = self._reconstruct_path(source_node, destination_node, visited, parent)  # O(V)
        if not shortest_path:  # O(1)
            return [], math.inf

        return shortest_path, distance[destination_node]  # O(1)

    # # RARELY ASKED ********* APSP Methods *********
    # def floyd_warshall(self):
    #     """
    #         Finds all-pairs shortest paths using Floyd-Warshall algorithm. Dynamic programming approach that considers
    #         all intermediate nodes. Handles negative edge weights and detects negative cycles.
            
    #         Returns:
    #             tuple: (distance_matrix, parent_matrix) -
    #             distance_matrix[i][j] is shortest distance from node i to j, parent_matrix for path reconstruction.
    #             Returns ([], []) if graph is empty
            
    #         Time Complexity: O(V^3)
    #         Space Complexity: O(V^2)
            
    #         Edge Cases: Empty graph, negative cycles, disconnected graph, self-loops
    #     """
    #     num_nodes = len(self._nodes)
    #     if num_nodes == 0:
    #         return [], []
        
    #     # Create node to index mapping for consistent indexing => O(V) TC, O(V) SC
    #     sorted_nodes = sorted(self._nodes)
    #     self._node_to_index: dict[str, int] = {node: index for index, node in enumerate(sorted_nodes)}
    #     self._index_to_node: dict[int, str] = {index: node for node, index in self._node_to_index.items()}
        
    #     # Initialize distance matrix: 0 for diagonal (self-loops), infinity elsewhere => O(V^2) TC, O(V^2) SC
    #     distance_matrix: list[list[float]] = [
    #         [0.0 if row == column else math.inf for column in range(num_nodes)] 
    #         for row in range(num_nodes)
    #     ]
    #     # Initialize parent matrix: None indicates no path, diagonal is None (no node is parent for itself) => O(V^2) TC, O(V^2) SC
    #     parent_matrix: list[list[str | None]] = [[None] * num_nodes for _ in range(num_nodes)]

    #     # Fill distance matrix with direct edge weights => O(V + E) TC
    #     for source in self._nodes:
    #         source_idx = self._node_to_index[source]
    #         for weight, destination in self._adjacency_list[source]:
    #             destination_idx = self._node_to_index[destination]
                
    #             # Add the edge to the distance matrix and parent matrix
    #             distance_matrix[source_idx][destination_idx] = weight
    #             parent_matrix[source_idx][destination_idx] = source
                
    #             # If the graph is undirected, we need to add the edge in the opposite direction
    #             if self.graph_type == GraphType.UNDIRECTED:
    #                 distance_matrix[destination_idx][source_idx] = weight
    #                 parent_matrix[destination_idx][source_idx] = destination
        
    #     # Floyd-Warshall algorithm: consider all intermediate nodes
    #     # Key insight: For each intermediate node k, update all pairs (i, j)
    #     for k in range(num_nodes): # Intermediate node k
    #         for i in range(num_nodes): # Source node i
    #             for j in range(num_nodes): # Destination node j
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
    def get_out_degree(self, node: str):
        """
            Returns the out-degree (number of outgoing edges) of a node.
            
            Args:
                node: Node to get out-degree for
            
            Returns:
                int: Number of outgoing edges from the node
            
            Time Complexity: O(1) average case
            Space Complexity: O(1)
            
            Edge Cases: Node doesn't exist (returns 0), node with no outgoing edges
        """
        return len(self._adjacency_list.get(node, [])) # O(1) average case

    def get_in_degree(self, node: str):
        """
            Returns the in-degree (number of incoming edges) of a node. Note: O(V + E) complexity.
            For O(1) complexity, maintain a separate in-degree dictionary updated during edge operations.
            
            Args:
                node: Node to get in-degree for
            
            Returns:
                int: Number of incoming edges to the node
            
            Time Complexity: O(V + E)
            Space Complexity: O(1)
            
            Edge Cases: Node doesn't exist (returns 0), node with no incoming edges
        """
        in_degree_count: int = 0
        for source in self._nodes: # O(V + E) TC
            for weight, neighbor in self._adjacency_list.get(source, []): # O(degree(source)) = O(E) total
                if neighbor == node:
                    in_degree_count += 1
        return in_degree_count 

    # ********* Graph Traversal Based Methods *********
    def is_cyclic_dfs(self):
        """
            Detects cycles in directed graph using DFS. Tracks nodes on call stack to identify back edges.
            
            Returns:
                bool: True if graph contains a cycle, False otherwise
            
            Time Complexity: O(V + E)
            Space Complexity: O(V) + recursion stack O(V) worst case
            
            Edge Cases: Empty graph, disconnected graph, self-loops, multiple cycles
        """
        visited: set[str] = set()
        nodes_on_call_stack: set[str] = set()  # To track nodes on the call stack to detect back edges (cycles)
        def dfs_helper(current_node: str):
            visited.add(current_node)
            nodes_on_call_stack.add(current_node)
            for weight, neighbor in self._adjacency_list.get(current_node, []):
                if neighbor not in visited:
                    if dfs_helper(neighbor):
                        return True
                # If neighbor is on the call stack, we found a back edge (cycle)
                elif neighbor in nodes_on_call_stack:
                    return True
            nodes_on_call_stack.discard(current_node)
            return False

        # Handle disconnected graphs
        for start_node in self._nodes:
            if start_node not in visited:
                if dfs_helper(start_node):
                    return True
        return False

    def topological_sort_kahn_s_bfs(self):
        """
            Performs topological sorting using Kahn's algorithm. BFS-based approach that processes nodes with
            in-degree 0 first, then removes their outgoing edges and repeats.
            
            Returns:
                list: Nodes in topologically sorted order if graph is a DAG, None if cycle exists
            
            Time Complexity: O(V + E)
            Space Complexity: O(V)
            
            Edge Cases: Empty graph, cyclic graph (returns None), disconnected DAG, single node
        """
        queue: deque[str] = deque()
        in_degree: defaultdict[str, int] = defaultdict(int)
        topological_order: list[str] = []
        
        # Calculate in-degrees of all nodes => O(V + E) TC
        for node in self._nodes:
            for weight, neighbor in self._adjacency_list.get(node, []):
                in_degree[neighbor] += 1

        # Initialize queue with nodes having in-degree 0 (handles disconnected graphs)
        for node in self._nodes:
            if in_degree[node] == 0:
                queue.append(node)
        
        # Process nodes in topological order using BFS => O(V + E) TC
        while queue:
            current_node = queue.popleft()
            topological_order.append(current_node)
            for weight, neighbor in self._adjacency_list.get(current_node, []):
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)
        
        # If all nodes were processed, the graph is a DAG
        if len(topological_order) == len(self._nodes):
            return topological_order
        
        return None

    # # RARELY ASKED
    # def kosaraju(self):
    #     """
    #         Finds all strongly connected components (SCCs) using Kosaraju's algorithm. Two-pass DFS approach:
    #         first on original graph to get finish order, then on reversed graph in reverse finish order.
    #         An SCC is a maximal set where every vertex is reachable from every other vertex.
            
    #         Returns:
    #             list: List of lists, where each inner list contains nodes in one SCC
            
    #         Time Complexity: O(V + E)
    #         Space Complexity: O(V + E)
            
    #         Edge Cases: Empty graph, no SCCs (each node is its own SCC), single large SCC, disconnected graph
    #     """
    #     # 1. Create the reversed graph
    #     rev_adj_lst: defaultdict[str, list[tuple[float, str]]] = defaultdict(list)
    #     for node in self._nodes:
    #         for weight, neighbor in self._adjacency_list.get(node, []):
    #             rev_adj_lst[neighbor].append((weight, node))
        
    #     # 2. First DFS (on original graph) to get finish order
    #     visited: set[str] = set()
    #     finish_order: list[str] = []

    #     def dfs_1(current_node: str):
    #         visited.add(current_node)
    #         for weight, neighbor in self._adjacency_list.get(current_node, []):
    #             if neighbor not in visited:
    #                 dfs_1(neighbor)
    #         finish_order.append(current_node)  # Post order Traversal

    #     # Run the first pass to get the finish order => O(V + E) TC
    #     for start_node in self._nodes:
    #         if start_node not in visited:
    #             dfs_1(start_node)
        
    #     # 3. Second DFS (on reversed graph) in reverse finish order => O(V + E) TC
    #     visited.clear()
    #     strongly_connected_components: list[list[str]] = []

    #     def dfs_2(current_node: str):
    #         visited.add(current_node)
    #         strongly_connected_components[-1].append(current_node)
    #         for weight, neighbor in rev_adj_lst.get(current_node, []):
    #             if neighbor not in visited:
    #                 dfs_2(neighbor)

    #     for start_node in finish_order[::-1]:
    #         if start_node not in visited:
    #             strongly_connected_components.append([])
    #             dfs_2(start_node)

    #     return strongly_connected_components

    # # RARELY ASKED ********* SSSP Methods *********
    # def bellman_ford(self, source_node: str, destination_node: str):
    #     """
    #         Finds shortest path using Bellman-Ford algorithm. Relaxes all edges V-1 times. Can handle negative weights
    #         and detects negative cycles. Works with edge list or adjacency list representation.
            
    #         Args:
    #             source_node: Starting node
    #             destination_node: Target node
            
    #         Returns:
    #             tuple: (shortest_path, shortest_path_distance) if path exists and no negative cycle,
    #             ([], math.inf) if no path exists, None if negative cycle detected
            
    #         Time Complexity: O(V * E)
    #         Space Complexity: O(V)
            
    #         Edge Cases: Source equals destination, no path exists, negative cycle reachable from source,
    #         disconnected graph, early termination when no relaxation occurs
    #     """
    #     if source_node == destination_node:
    #         return [source_node], 0.0

    #     parent: dict[str, str | None] = {}  # O(1) space, grows to O(V)
    #     distance: defaultdict[str, float] = defaultdict(lambda: math.inf)  # O(1) space, grows to O(V)

    #     distance[source_node] = 0  
    #     parent[source_node] = None # None represents no parent

    #     # Bellman-Ford algorithm: relax edges V-1 times - O(V * E)
    #     # Key insight: After V-1 iterations, all shortest paths should be found
    #     num_nodes = len(self._nodes)  # O(1)
    #     for iteration in range(num_nodes):  # O(V) iterations
    #         relaxed = False  
    #         for node in self._nodes:  # O(V) iterations per outer iteration
    #             if distance[node] != math.inf:
    #                 for weight, neighbor in self._adjacency_list.get(node, []):  # O(degree(node)) = O(E) total
    #                     new_distance = distance[node] + weight 
    #                     if new_distance < distance[neighbor]:  
    #                         distance[neighbor] = new_distance 
    #                         parent[neighbor] = node 
    #                         relaxed = True 
    #                         if iteration == num_nodes - 1:
    #                             # If we can still relax, a negative cycle exists
    #                             return None
    #         # Early termination optimization: if no relaxation, we're done (no negative cycles)
    #         if not relaxed:
    #             break

    #     # Reconstruct path using helper method - O(V) worst case
    #     # Create visited set from nodes with finite distance (reachable nodes)
    #     visited = {node for node in distance if distance[node] != math.inf}
    #     shortest_path = self._reconstruct_path(source_node, destination_node, visited, parent)  # O(V)
    #     if not shortest_path:
    #         return [], math.inf

    #     return shortest_path, distance[destination_node]


# class DisjointSet:

#     def __init__(self):
#         self.parent: dict[str, str] = {} 
#         self.size: defaultdict[str, int] = defaultdict(lambda: 1) 

#     def find_ultimate_parent(self, node: str):
#         """
#             Finds the root of a node with path compression. Flattens tree structure for faster future lookups.
            
#             Args:
#                 node: Node to find root for
            
#             Returns:
#                 str: Root node of the component containing the given node
            
#             Time Complexity: O(α(V)) amortized (inverse Ackermann function, effectively constant)
#             Space Complexity: O(1) excluding recursion stack
            
#             Edge Cases: Node not in set (creates new singleton set), single node component, deep tree before compression
#         """
#         if node not in self.parent:
#             self.parent[node] = node
#             return node
        
#         # Base case: node is its own parent (root)
#         if self.parent[node] == node:
#             return node
        
#         # Path compression: make parent point directly to root
#         # This flattens the tree structure for future lookups
#         self.parent[node] = self.find_ultimate_parent(self.parent[node])
#         return self.parent[node]

#     def union_by_size(self, u: str, v: str):
#         """
#             Unions two components using union by size. Attaches smaller component to larger one to minimize tree height.
            
#             Args:
#                 u: First node
#                 v: Second node
            
#             Time Complexity: O(α(V)) amortized (two find operations)
#             Space Complexity: O(1) excluding recursion stack
            
#             Edge Cases: Nodes already in same component (no-op), nodes not in set (creates new components), single node unions
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
    def get_degree(self, node: str):
        """
            Returns the degree (number of adjacent edges) of a node in an undirected graph.
            
            Args:
                node: Node to get degree for
            
            Returns:
                int: Number of edges incident to the node
            
            Time Complexity: O(1) average case
            Space Complexity: O(1)
            
            Edge Cases: Node doesn't exist (returns 0), isolated node (returns 0)
        """
        # For undirected graphs, degree equals the number of neighbors
        return len(self._adjacency_list.get(node, [])) # O(1) average case

    # ********* Graph Traversal Based Methods *********
    def is_cyclic(self):
        """
            Detects cycles in undirected graph using DFS. Identifies back edges (edges to already visited non-parent nodes).
            
            Returns:
                bool: True if graph contains a cycle, False otherwise
            
            Time Complexity: O(V + E)
            Space Complexity: O(V) + recursion stack O(V) worst case
            
            Edge Cases: Empty graph, disconnected graph, tree (no cycles), self-loops, multiple cycles
        """
        visited: set[str] = set()

        def dfs_helper(current_node: str, parent_node: str | None):
            visited.add(current_node)
            for weight, neighbor in self._adjacency_list.get(current_node, []):
                if neighbor not in visited:
                    if dfs_helper(neighbor, current_node):
                        return True
                # If neighbor is already visited and is not the child of the current node, we found a back edge (cycle)
                elif neighbor != parent_node:
                    return True
            return False

        # Handle disconnected graphs
        for start_node in self._nodes:
            if start_node not in visited:
                if dfs_helper(start_node, None):
                    return True
        return False

    # def is_bipartite(self):
    #     """
    #         Checks if graph is bipartite using BFS coloring. A graph is bipartite if nodes can be divided into
    #         two sets such that no edges exist within the same set.
            
    #         Returns:
    #             bool: True if graph is bipartite, False otherwise
            
    #         Time Complexity: O(V + E)
    #         Space Complexity: O(V)
            
    #         Edge Cases: Empty graph, disconnected graph (each component must be bipartite), odd-length cycles (not bipartite),
    #         single node, tree (always bipartite)
    #     """
    #     colors: dict[str, bool] = {}  # Also serves as visited set
    #     queue: deque[str] = deque()

    #     # Handle disconnected graphs
    #     for start_node in self._nodes:
    #         if start_node not in colors:
    #             queue.append(start_node)
    #             colors[start_node] = True

    #             # BFS on a new connected component
    #             while queue:
    #                 current_node = queue.popleft()
    #                 for weight, neighbor in self._adjacency_list.get(current_node, []):
    #                     if neighbor not in colors:
    #                         colors[neighbor] = not colors[current_node]
    #                         queue.append(neighbor)
    #                     # If neighbor has same color as current node, graph is not bipartite
    #                     elif colors[neighbor] == colors[current_node]:
    #                         return False
        
    #     return True

    # # ********* Minimum Spanning Tree Methods *********
    # def mst_prim_s(self):
    #     """
    #         Finds Minimum Spanning Tree (MST) using Prim's algorithm. Greedy approach that grows MST by always
    #         adding the cheapest edge connecting MST to a vertex not yet in MST.
            
    #         Returns:
    #             tuple: (mst_edges, mst_weight) - list of edges in MST and total weight
            
    #         Time Complexity: O(E log E) with edge-based heap (can be O(E log V) with vertex-based heap)
    #         Space Complexity: O(E + V)
            
    #         Edge Cases: Empty graph, disconnected graph (finds MST of connected component), single node,
    #         graph with no edges, negative weights (works but may not be meaningful)
    #     """
    #     if not self._adjacency_list:
    #         return [], 0.0
        
    #     visited: set[str] = set()
    #     priority_queue: list[tuple[float, str | None, str]] = [] # (edge_weight, source_node, destination_node)
    #     mst_weight: float = 0
    #     mst_edges: list[tuple[float, str, str]] = []
        
    #     # Start with an arbitrary node (use dummy parent None for the start node)
    #     start_node = next(iter(self._adjacency_list))
    #     heapq.heappush(priority_queue, (0, None, start_node))
        
    #     while priority_queue: # O(E) iterations
    #         # Pop the cheapest edge connecting the MST to a new node (TC: O(log E) per iteration)
    #         current_weight, parent_node, current_node = heapq.heappop(priority_queue)
            
    #         if current_node not in visited:
    #             visited.add(current_node)
    #             mst_weight += current_weight
    #             if parent_node is not None:
    #                 mst_edges.append((current_weight, current_node, parent_node))
                
    #             # Add all adjacent edges from this newly visited node
    #             for weight, neighbor in self._adjacency_list.get(current_node, []):
    #                 if neighbor not in visited:
    #                     # TC: O(log E) per push operation
    #                     heapq.heappush(priority_queue, (weight, current_node, neighbor))
        
    #     return mst_edges, mst_weight

    # def mst_kruskal_s(self):
    #     """
    #         Finds Minimum Spanning Tree (MST) using Kruskal's algorithm. Sorts edges by weight and adds them
    #         if they don't create a cycle, using Union-Find to detect cycles efficiently.
            
    #         Returns:
    #             tuple: (mst_edges, mst_weight) - list of edges in MST and total weight
            
    #         Time Complexity: O(E log E) dominated by sorting
    #         Space Complexity: O(E + V)
            
    #         Edge Cases: Empty graph, disconnected graph (finds MST of each component), single node,
    #         graph with no edges, duplicate edges (processed once due to source < destination check)
    #     """
    #     # Only include edges where source < destination to avoid duplicates in undirected graphs
    #     # This ensures each edge is processed exactly once instead of twice
    #     sorted_edges: list[tuple[float, str, str]] = sorted(
    #         [
    #             (weight, source, destination) 
    #             for source, neighbors in self._adjacency_list.items() 
    #             for weight, destination in neighbors
    #             if source < destination  # Only process each edge once
    #         ], key=lambda edge: edge[0]
    #     ) # O(E log E) TC, O(E) SC
    #     mst_weight: float = 0.0
    #     mst_edges: list[tuple[float, str, str]] = []
    #     disjoint_set: DisjointSet = DisjointSet()

    #     for weight, source, destination in sorted_edges:  # O(E * 4α) TC
    #         # Add edge if it doesn't create a cycle (nodes are in different components)
    #         source_root = disjoint_set.find_ultimate_parent(source)
    #         destination_root = disjoint_set.find_ultimate_parent(destination)
    #         if source_root != destination_root:
    #             mst_weight += weight
    #             mst_edges.append((weight, source, destination))
    #             disjoint_set.union_by_size(source, destination)
        
    #     return mst_edges, mst_weight

    # # RARELY ASKED ********* Critical Connections, Articulation Points Methods *********
    # def get_bridges(self):
    #     """
    #         Finds all bridges (critical connections) using Tarjan's algorithm. A bridge is an edge whose removal
    #         increases the number of connected components. Uses DFS with discovery time and low-link values.
            
    #         Returns:
    #             list: List of bridges as tuples (weight, source, destination)
            
    #         Time Complexity: O(V + E)
    #         Space Complexity: O(V + E) including recursion stack
            
    #         Edge Cases: Empty graph, graph with no bridges, disconnected graph, single edge, tree (all edges are bridges)
    #     """
    #     visited: set[str] = set()
    #     bridges: list[tuple[float, str, str]] = []
    #     # To store the time of insertion (discovery time) of nodes
    #     discovery_time: defaultdict[str, int] = defaultdict(lambda: 0)  
    #     # To store the lowest time of insertion of the nodes
    #     low_dis_time: defaultdict[str, int] = defaultdict(lambda: 0)  
    #     timer: int = 1  # To keep track of the time of insertion of nodes
    #     def dfs_helper(current_node: str, parent: str | None):
    #         nonlocal timer
    #         visited.add(current_node)
    #         discovery_time[current_node] = low_dis_time[current_node] = timer
    #         timer += 1
    #         for weight, neighbor in self._adjacency_list.get(current_node, []):
    #             if neighbor not in visited:
    #                 dfs_helper(neighbor, current_node)
    #                 low_dis_time[current_node] = min(low_dis_time[current_node], low_dis_time[neighbor])
    #                 # If the lowest discovery time of the neighbor is > the discovery time of current current_node => (current_node, neighbor) forms a bridge
    #                 if low_dis_time[neighbor] > discovery_time[current_node]:
    #                     bridges.append((weight, current_node, neighbor))
    #             elif neighbor != parent:
    #                 low_dis_time[current_node] = min(low_dis_time[current_node], discovery_time[neighbor])
        
    #     # Start DFS traversal from all nodes in the graph to handle disconnected graphs
    #     for start_node in self._nodes:
    #         if start_node not in visited:
    #             dfs_helper(start_node, None)
        
    #     return bridges

    # def get_articulation_points(self):
    #     """
    #         Finds all articulation points using Tarjan's algorithm. An articulation point is a node whose removal
    #         increases the number of connected components. Uses DFS with discovery time and low-link values.
            
    #         Returns:
    #             list: Sorted list of articulation point nodes
            
    #         Time Complexity: O(V + E) + O(V log V) for sorting
    #         Space Complexity: O(V + E) including recursion stack
            
    #         Edge Cases: Empty graph, graph with no articulation points, disconnected graph, single node,
    #         tree (all non-leaf nodes are articulation points), complete graph (no articulation points)
    #     """
    #     visited: set[str] = set()
    #     articulation_points: set[str] = set()
    #     discovery_time: defaultdict[str, int] = defaultdict(lambda: -1)
    #     low_dis_time: defaultdict[str, int] = defaultdict(lambda: -1)
    #     timer: int = 1
    #     def dfs_helper(current_node: str, parent: str | None):
    #         # to specify the timer variable in the outer scope
    #         nonlocal timer
    #         visited.add(current_node)
    #         discovery_time[current_node] = low_dis_time[current_node] = timer
    #         timer += 1
    #         children: int = 0  # To count the number of children of the current_node
    #         for weight, neighbor in self._adjacency_list.get(current_node, []):
    #             if neighbor not in visited:
    #                 dfs_helper(neighbor, current_node)
    #                 # Update the lowest time of insertion for the current_node 
    #                 low_dis_time[current_node] = min(low_dis_time[current_node], low_dis_time[neighbor])
    #                 # If the lowest time of insertion of the current_node is found to be greater than the time of insertion of the neighbor and it is not the starting node
    #                 if low_dis_time[neighbor] >= discovery_time[current_node] and parent is not None:
    #                     # Mark the current_node as an articulation point
    #                     articulation_points.add(current_node)
    #                 children += 1
    #             elif neighbor != parent:
    #                 low_dis_time[current_node] = min(low_dis_time[current_node], discovery_time[neighbor])
            
    #         # If the current_node is a starting node and has more than one children 
    #         if parent is None and children > 1:
    #             # Mark the current_node as an articulation point
    #             articulation_points.add(current_node)
        
    #     # Start DFS traversal from all nodes in the graph to handle disconnected graphs
    #     for start_node in self._nodes:
    #         if start_node not in visited:
    #             dfs_helper(start_node, None)

    #     # sorted can handle any iterable and sets also
    #     return sorted(articulation_points)
