import heapq
from collections import deque


class UndirectedGraph:

    def __init__(self):
        self.adj_list: list[list[tuple[float, int]]] = []

    # TC: O(V+2E) & SC: O(V+2E)
    def build_from_edges(self, edges: list[tuple[int, int, float]], V: int) -> None:
        # invalid input
        if V <= 0: return

        # clear the last graph
        if self.adj_list:
            self.adj_list = []
        
        # create the new graph
        self.adj_list = [[] for _ in range(V)]
        for src, des, weight in edges:
            self.adj_list[src].append((weight, des))
            self.adj_list[des].append((weight, src))

    # TC: O(V^2) & SC: O(V+2E)
    def build_from_matrix(self, matrix: list[list[float]]) -> None:
        # invalid input
        V = len(matrix)
        if V == 0: return

        # clear the last graph
        if self.adj_list:
            self.adj_list = []
        
        # create the new graph
        self.adj_list = [[] for _ in range(V)]
        for src in range(V):
            for des in range(V):
                weight = matrix[src][des]
                if (weight != float('inf')) and (src != des):
                    self.adj_list[src].append((weight, des))

    # https://takeuforward.org/plus/dsa/problems/traversal-techniques?subject=dsa&approach=time-complexity-and-in-depth-theory&tab=editorial
    # TC: O(V+2E) & SC: O(4V)
    def bfs(self) -> tuple[list[list[int]], list[int]] | None:
        # invalid input
        V = len(self.adj_list)
        if (V == 0): return

        # initializations
        visited: list[bool] = [False] * V
        parents: list[int | None] = [None] * V
        components: list[list[int]] = []
        q: deque[int] = deque()

        # handling the disconnected graph
        for start in range(V):
            if not visited[start]:
                # starting the BFS on a new connected component
                visited[start] = True
                parents[start] = -1
                components.append([])
                q.append(start)

                while q:
                    curr = q.popleft()
                    components[-1].append(curr)
                    # BFS on curr node started
                    for _, neigh in self.adj_list[curr]:
                        if not visited[neigh]:
                            visited[neigh] = True
                            parents[neigh] = curr
                            q.append(neigh)
                    # BFS on curr node completed
                # BFS traversal on a new connected component completed
        
        return components, parents

    # https://takeuforward.org/plus/dsa/problems/traversal-techniques?subject=dsa&approach=time-complexity-and-in-depth-theory&tab=editorial
    # TC: O(V+2E) & SC: O(3V) + O(V) call stack
    def dfs(self) -> tuple[list[list[int]], list[int]] | None:
        # invalid input
        V = len(self.adj_list)
        if V == 0: return

        # initializations
        visited: list[bool] = [False] * V
        parents: list[int] = [-1] * V
        components: list[list[int]] = []

        def helper(curr: int, par: int):
            components[-1].append(curr)
            visited[curr] = True
            parents[curr] = par

            for _, neigh in self.adj_list(curr):
                if not visited[neigh]:
                    helper(neigh, curr)

        for start in range(V):
            if not visited[start]:
                # DFS traversal on a new connected component started
                components.append([])
                helper(start, -1)
                # DFS traversal on a new connected component completed

        return components, parents

    # TC: O(V+2E+k) & SC: O(3V+k) where k is length of path from src to des
    def sssp_with_bfs(self, src: int, des: int) -> tuple[list[int], int] | None:
        if src == des:
            return [src], 0

        # graph not built yet
        V = len(self.adj_list)
        if V == 0: return

        # initializations
        visited: list[bool] = [False] * V
        parents: list[int | None] = [None] * V
        q: deque[int] = deque()

        # initializations to start the BFS on a component with src node
        visited[src] = True
        parents[src] = -1
        q.append(src)

        while q:
            curr = q.popleft()
            
            # BFS traversal on the curr node started
            for _, neigh in self.adj_list[curr]:
                if not visited[neigh]:
                    visited[neigh] = True
                    parents[neigh] = curr
                    q.append(neigh)
            # BFS traversal on the curr node ended
        # BFS traversal on a connected component of the graph with src node has completed
        
        # if there is no way to reach the destination form source
        if parents[des] is None:
            return [], float('inf')

        # reversal traversal from destination to source with parent links 
        # to discover path from source node to destion node
        path: list[int] = [des]
        curr = des
        while curr != src:
            curr = parents[curr]
            path.append(curr)
        path.reverse()

        return path, len(path)-1

    # https://takeuforward.org/plus/dsa/problems/dijkstra's-algorithm?subject=dsa&approach=dfs&tab=editorial
    # TC: O((V + E)*log V) & SC: O(V + E)
    def dijkstra(self, src: int, des: int) -> tuple[list[int], float] | None:
        if src == des:
            return [src], 0.0
        
        # the graph is not built yet
        V = len(self.adj_list)
        if V == 0: return

        # initialization for BFS traversal on a component of the graph with src node
        distances: list[float] = [float('inf')] * V
        parents: list[int | None] = [None] * V
        pq: list[tuple[float, int]] = []

        distances[src] = 0
        parents[src] = -1
        heapq.heappush(pq, (0.0, src))

        while pq:
            curr_dis, curr = heapq.heappop(pq)
            if curr_dis > distances[curr]:
                continue

            # starting the BFS traversal on curr node
            for dis, neigh in self.adj_list[curr]:
                new_dis = dis + curr_dis
                if new_dis < distances[neigh]:
                    distances[neigh] = new_dis
                    parents[neigh] = curr
                    heapq.heappush(pq, (new_dis, neigh))
            # BFS traversal on curr node completed
        
        # if there is no way to traverse from src to des node
        if parents[des] is None:
            return [], float('inf')
        
        # discovering the src to des path by traversing back from des to src by 
        # using the parent links in the parents array
        path: list[int] = [des]
        curr = des
        while curr != src:
            curr = parents[curr]
            path.append(curr)
        path.reverse()

        return path, distances[des]
