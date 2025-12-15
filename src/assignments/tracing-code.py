class Solution:

    def isCycle(self, V: int, adj: list[list[int]]) -> bool:
        visited: list[bool] = [False] * V
        
        def dfs(current_node: int, parent: int) -> bool:
            visited[current_node] = True
            for neighbor in adj[current_node]:  # [1, 2], [2], [0]
                if not visited[neighbor]:  # True, True, False
                    if dfs(neighbor, current_node):  # dfs2(1, 0), dfs3(2, 1)
                        return True
                # if the neighbor is already visited &
                # the neighbor is not equals to the parent of the current_node
                # => we found a back edge
                elif neighbor != parent:  # True
                    return True
            return False

        for starting_node in range(V):
            if not visited[starting_node]:
                if dfs(starting_node, -1):  # dfs1(0, -1)
                    return True
        return False


print("Test Case 1:")
print(Solution().isCycle(3, [[1, 2], [2], [0]]))

# visited = [True, True, True]
# starting_node = 0

# RECURSIVE 1
# current_node = 0
# parent = -1
# neighbor = 1

# RECURSIVE 2
# current_node = 1
# parent = 0
# neighbor = 2

# RECURSIVE 3
# current_node = 2
# parent = 1
# neighbor = 0
