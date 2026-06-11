from __future__ import annotations
from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass
class TreeNode:
    data: int
    left: TreeNode | None = None
    right: TreeNode | None = None


class BinaryTree:

    def __init__(self):
        self.root: TreeNode | None = None

    def build_tree(self, nodes: list[int]):
        """
            Builds a complete binary tree from a list of nodes using recursive approach.
            Args:
                nodes: The list of nodes to build the tree from.
            Returns:
                TreeNode: The root of the tree.
            Time Complexity: O(n)
            Space Complexity: O(height) = O(log n) for balanced tree
        """
        if not nodes:
            return None
        
        def build(index: int) -> TreeNode | None:
            if index >= len(nodes):
                return None
            node = TreeNode(nodes[index])
            node.left = build(2 * index + 1)
            node.right = build(2 * index + 2)
            return node
        
        self.root = build(0)
        return self.root

    # https://takeuforward.org/plus/dsa/problems/inorder-traversal?subject=dsa&approach=graphs-and-types
    def in_order_traversal(self):  # LNR
        """
            Performs In Order Traversal (DFS) traversal. Explores the tree in the order: left, root, right.
            Returns:
                list[int]: The list of nodes in in order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = []
        def in_order_traversal_helper(node: TreeNode | None):
            if node is None:
                return
            in_order_traversal_helper(node.left)
            result.append(node.data)
            in_order_traversal_helper(node.right)

        in_order_traversal_helper(self.root)
        return result

    # https://takeuforward.org/plus/dsa/problems/preorder-traversal?subject=dsa&approach=graphs-and-types
    def pre_order_traversal(self):  # NLR
        """
            Performs Pre Order Traversal (DFS) traversal. Explores the tree in the order: root, left, right.
            Returns:
                list[int]: The list of nodes in pre order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = []
        def pre_order_traversal_helper(node: TreeNode | None):
            if node is None:
                return
            result.append(node.data)
            pre_order_traversal_helper(node.left)
            pre_order_traversal_helper(node.right)

        pre_order_traversal_helper(self.root)
        return result

    # https://takeuforward.org/plus/dsa/problems/postorder-traversal?subject=dsa&approach=graphs-and-types
    def post_order_traversal(self):  # LRN
        """
            Performs Post Order Traversal (DFS) traversal. Explores the tree in the order: left, right, root.
            Returns:
                list[int]: The list of nodes in post order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = []
        def post_order_traversal_helper(node: TreeNode | None):
            if node is None:
                return
            post_order_traversal_helper(node.left)
            post_order_traversal_helper(node.right)
            result.append(node.data)

        post_order_traversal_helper(self.root)
        return result

    # https://takeuforward.org/plus/dsa/problems/level-order-traversal?subject=dsa&approach=graphs-and-types
    def level_order_traversal(self):
        """
            Performs Level Order Traversal (BFS) traversal. Explores the tree level by level from left to right.
            Returns:
                list[int]: The list of nodes in level order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if self.root is None:
            return []
        queue = deque([self.root])
        result: list[int] = []
        while queue:
            current_node = queue.popleft()
            result.append(current_node.data)
            if current_node.left is not None:
                queue.append(current_node.left)
            if current_node.right is not None:
                queue.append(current_node.right)
        return result

    def level_order_traversal_v2(self):
        """
            Performs Level Order Traversal (BFS) with level separators.
            Returns:
                list[list[int]]: List of levels, where each level is a list of node values.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if self.root is None:
            return []
        
        result: list[list[int]] = []
        queue = deque([self.root])
        
        while queue:
            level_size = len(queue)
            current_level: list[int] = []
            
            for _ in range(level_size):
                current_node = queue.popleft()
                current_level.append(current_node.data)
                
                if current_node.left is not None:
                    queue.append(current_node.left)
                if current_node.right is not None:
                    queue.append(current_node.right)
            
            result.append(current_level)
        
        return result

    # https://takeuforward.org/plus/dsa/problems/pre,-post,-inorder-in-one-traversal?subject=dsa&approach=graphs-and-types
    def pre_in_post_in_single_go(self) -> list[list[int], list[int], list[int]]:
        pre, in_order, post = [], [], []

        if not self.root:
            return [pre, in_order, post]
        
        stack = [(self.root, 1)]

        while stack:
            node, state = stack.pop()
            
            if state == 1:
                pre.append(node.data)
                stack.append((node, 2))
                if node.left:
                    stack.append((node, 1))
            elif state == 2:
                in_order.append(node.data)
                stack.append((node, 3))
                if node.right: 
                    stack.append((node, 1))
            else:
                post.append(node.data)
        
        return [pre, in_order, post]

    # https://takeuforward.org/plus/dsa/problems/maximum-depth-in-bt?subject=dsa&approach=graphs-and-types
    def height_of_tree(self):  # max depth of the tree
        """
            Calculates the height of the tree.
            Returns:
                int: The height of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def height_of_tree_helper(node: TreeNode | None):
            if node is None:
                return 0
            left_height = height_of_tree_helper(node.left)
            right_height = height_of_tree_helper(node.right)
            return max(left_height, right_height) + 1

        return height_of_tree_helper(self.root)

    # https://takeuforward.org/plus/dsa/problems/check-for-balanced-binary-tree?subject=dsa&approach=graphs-and-types
    def is_tree_balanced(self):
        """
            Checks if the tree is balanced (height difference between left and right subtrees <= 1).
            Returns:
                bool: True if the tree is balanced, False otherwise.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def check_balanced(node: TreeNode | None) -> tuple[bool, int]:
            if node is None:
                return True, 0
            
            left_balanced, left_height = check_balanced(node.left)
            right_balanced, right_height = check_balanced(node.right)
            
            is_balanced = (left_balanced and right_balanced and 
                          abs(left_height - right_height) <= 1)
            height = max(left_height, right_height) + 1
            
            return is_balanced, height
        
        is_balanced, _ = check_balanced(self.root)
        return is_balanced
    def check_balaced_tree_or_not(self):
        def height(node: TreeNode) -> int:
            if node is None:
                return 0
            return 1 + max(height(node.left), height(node.right))
        
        def balance_helper(node: TreeNode) -> bool:
            if root is None:
                return True
            
            if abs(height(node.left) - height(node.right)) > 1:
                return False
            
            return balance_helper(node.left) and balance_helper(node.right)
        
        return balance_helper(self.root)

    # https://takeuforward.org/plus/dsa/problems/diameter-of-binary-tree?subject=dsa&approach=graphs-and-types
    def diameter_of_tree(self):
        """
            Calculates the diameter of the tree (longest path between any two nodes).
            Returns:
                int: The diameter of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        max_diameter = 0
        
        def height_and_diameter(node: TreeNode | None) -> int:
            nonlocal max_diameter
            if node is None:
                return 0
            
            left_height = height_and_diameter(node.left)
            right_height = height_and_diameter(node.right)
            
            # Diameter passing through current node
            current_diameter = left_height + right_height
            max_diameter = max(max_diameter, current_diameter)
            
            # Return height of current subtree
            return max(left_height, right_height) + 1
        
        height_and_diameter(self.root)
        return max_diameter

    # # https://takeuforward.org/plus/dsa/problems/maximum-path-sum-?subject=dsa&approach=graphs-and-types
    def path_sum(self, target_sum: int):
        """
            Finds all root-to-leaf paths that sum to the target sum.
            Args:
                target_sum: The target sum to find.
            Returns:
                list[list[int]]: List of all paths (each path is a list of node values).
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        curr_path: list[int] = []
        result: list[list[int]] = []

        def helper(node: TreeNode | None, remaining_sum: int):
            if node is None:
                return
            
            curr_path.append(node.data)
            remaining_sum -= node.data
            
            # Check if leaf node and path sum matches
            if node.left is None and node.right is None:
                if remaining_sum == 0:
                    result.append(curr_path.copy())
            
            # Recurse on children
            helper(node.left, remaining_sum)
            helper(node.right, remaining_sum)
            
            # Backtrack
            curr_path.pop()

        helper(self.root, target_sum)
        return result

    # https://takeuforward.org/plus/dsa/problems/check-for-symmetrical-bts?subject=dsa&approach=graphs-and-types
    def is_symmetrical(self):
        if self.root is None:
            return True
        
        def helper(left: TreeNode, right: TreeNode) -> bool:
            if left is None and right is None:
                return True
            
            if left is not None or right is not None:
                return False
            
            if left.data =! right.data:
                return False
            
            return helper(left.left, right.right) and helper(left.right, right.left)

        return helper(root.left, root.right)
    
    # https://takeuforward.org/plus/dsa/problems/zig-zag-or-spiral-traversal?subject=dsa&approach=graphs-and-types
    from collections import deque
    def zig_zag_level_order_traversal(self):
        result = []
        if self.root is None:
            return result
        
        dq = deque([self.root])
        left_to_right = True

        while dq:
            size = len(dq)
            last_index = size - 1
            level = [None] * size

            for i in range(size):

                curr = dq.popleft()

                index = i if left_to_right else last_index - i
                level[index] = curr.data

                if curr.left:
                    dq.append(curr.left)
                if curr.right:
                    dq.append(curr.right)
                
                left_to_right = not left_to_right
        
        result.append(level)

        return result

    # https://takeuforward.org/plus/dsa/problems/top-view-of-bt?subject=dsa&approach=graphs-and-types
    def top_view(self):
        """
            Constructs the top view of the tree (leftmost node at each horizontal distance).
            Returns:
                list[int]: The top view of the tree, ordered from left to right.
            Time Complexity: O(n)
            Space Complexity: O(height) for recursion + O(n) for result = O(n)
        """
        if self.root is None:
            return []
        
        result: dict[int, int] = {}
        
        def helper(node: TreeNode | None, hd: int):
            if node is None:
                return
            if hd not in result:
                result[hd] = node.data
            helper(node.left, hd - 1)
            helper(node.right, hd + 1)
        
        helper(self.root, 0)
        return [result[hd] for hd in sorted(result.keys())]

    # https://takeuforward.org/plus/dsa/problems/bottom-view-of-bt?subject=dsa&approach=graphs-and-types
    def bottom_view(self):
        """
            Constructs the bottom view of the tree (bottommost node at each horizontal distance).
            Returns:
                list[int]: The bottom view of the tree, ordered from left to right.
            Time Complexity: O(n)
            Space Complexity: O(height) for recursion + O(n) for result = O(n)
        """
        if self.root is None:
            return []
        
        result: dict[int, int] = {}
        
        def helper(node: TreeNode | None, hd: int):
            if node is None:
                return
            result[hd] = node.data  # Always update to get bottommost
            helper(node.left, hd - 1)
            helper(node.right, hd + 1)
        
        helper(self.root, 0)
        return [result[hd] for hd in sorted(result.keys())]

    # https://takeuforward.org/plus/dsa/problems/right-left-view-of-bt?subject=dsa&approach=graphs-and-types
    def left_view(self):
        """
            Constructs the left view of the tree.
            Returns:
                list[int]: The left view of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        ans: list[int] = []
        def helper(node: TreeNode | None, level: int):
            if node is None:
                return
            if level == len(ans):
                ans.append(node.data)
            helper(node.left, level + 1)
            helper(node.right, level + 1)
        
        helper(self.root, 0)
        return ans

    # https://takeuforward.org/plus/dsa/problems/right-left-view-of-bt?subject=dsa&approach=graphs-and-types
    def right_view(self):
        """
            Constructs the right view of the tree.
            Returns:
                list[int]: The right view of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        ans: list[int] = []
        def helper(node: TreeNode | None, level: int):
            if node is None:
                return
            if level == len(ans):
                ans.append(node.data)
            helper(node.right, level + 1)
            helper(node.left, level + 1)
        
        helper(self.root, 0)
        return ans

    # https://takeuforward.org/plus/dsa/problems/boundary-traversal?subject=dsa&approach=graphs-and-types
    def boundary_traversal(self):
        """
            Constructs the boundary traversal of the tree (left boundary, leaves, right boundary).
            Returns:
                list[int]: The boundary traversal of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if self.root is None:
            return []
        
        # Single node case
        if self.root.left is None and self.root.right is None:
            return [self.root.data]
        
        result: list[int] = [self.root.data]
        
        def left_boundary(node: TreeNode | None):
            """Add left boundary (excluding root and leaves)"""
            if node is None or (node.left is None and node.right is None):
                return
            result.append(node.data)
            if node.left is not None:
                left_boundary(node.left)
            else:
                left_boundary(node.right)
        
        def leaf_boundary(node: TreeNode | None):
            """Add all leaf nodes"""
            if node is None:
                return
            if node.left is None and node.right is None:
                result.append(node.data)
            leaf_boundary(node.left)
            leaf_boundary(node.right)
        
        def right_boundary(node: TreeNode | None):
            """Add right boundary (excluding root and leaves)"""
            if node is None or (node.left is None and node.right is None):
                return
            if node.right is not None:
                right_boundary(node.right)
            else:
                right_boundary(node.left)
            result.append(node.data)
        
        # Traverse boundaries
        if self.root.left is not None:
            left_boundary(self.root.left)
        leaf_boundary(self.root)
        if self.root.right is not None:
            right_boundary(self.root.right)
        
        return result

    # https://takeuforward.org/plus/dsa/problems/vertical-order-traversal?subject=dsa&approach=optimal
    from collections import defaultdict, deque
    def vertical_traversal(self):
        if self.root is None:
            return []
        
        result: list[list[int]] = []
        ver_lev_key_maps: defaultdict[int, defaultdict[int, list]] = defaultdict(lambda: defaultdict(list))
        dq: deque[tuple[TreeNode, int, int]] = deque([(self.root, 0, 0)])

        ver_lev_key_maps: defaultdict[int, defaultdict[int, list]] = defaultdict(lambda: defaultdict(list))

        while dq:
            node, ver, lev = dq.popleft()
            ver_lev_key_maps[ver][lev] = node.data

            if node.left:
                dq.append((node.left, ver-1, lev+1))
            
            if node.right:
                dq.append((node.right, ver+1, lev+1))
            
        for ver in sorted(ver_lev_key_maps):
            col = []
            for lev in sorted(ver_lev_key_maps[ver]):
                col.extend(sorted(ver_lev_key_maps[ver][lev]))
            result.append(col)
        
        return result

    # https://takeuforward.org/plus/dsa/problems/print-root-to-note-path-in-bt?subject=dsa&approach=graphs-and-types
    def all_root_to_leaf_paths(self) -> list[list[int]]:
        all_paths: list[list[int]] = []
        if self.root is None:
            return all_paths
        
        def dfs(node: TreeNode, path: list[]):
            if node.left is None and node.right is None:
                path.append[node.data]
            else:
                if node.left:
                    dfs(node.left, path)
                if node.right:
                    dfs(node.right, path)
            path.pop()

        dfs(self.root, [])

        return all_paths
    
    # https://takeuforward.org/plus/dsa/problems/lca-in-bt?subject=dsa&approach=graphs-and-types
    def lowest_common_ancestor(self, value1: int, value2: int):
        """
            Finds the lowest common ancestor of two nodes.
            Returns:
                TreeNode: The lowest common ancestor of the two nodes.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def lowest_common_ancestor_helper(node: TreeNode | None, value1: int, value2: int):
            if node is None:
                return None
            if node.data == value1 or node.data == value2:
                return node
            left_ancestor = lowest_common_ancestor_helper(node.left, value1, value2)
            right_ancestor = lowest_common_ancestor_helper(node.right, value1, value2)
            if left_ancestor is not None and right_ancestor is not None:
                return node
            return left_ancestor if left_ancestor is not None else right_ancestor
        
        return lowest_common_ancestor_helper(self.root, value1, value2)

    # https://takeuforward.org/plus/dsa/problems/maximum-width-of-bt?subject=dsa&approach=graphs-and-types
    from collections import deque
    def maximum_width(self):
        ans = 0
        if self.root is None:
            return ans
        
        dq = deque([self.root, 0])

        while dq:

            size = len(dq)
            lev_min_id = dq[0][1]
            left, right = 0, 0

            for i in range(size):
                node, ind = dq[i]
                dq.popleft()

                cur_ind = ind - lev_mon_id
                
                if i == 0:
                    first = cur_ind
                
                if i == (size - 1):
                    last = cur_ind
                
                if node.left:
                    dq.append([node.left, (cur_ind << 1) + 1])
                
                if node.right:
                    dq.append([node.right, (curr_ind << 1) + 2])
                
            ans = max(ans, (right - left + 1))
        
        return ans

    # https://takeuforward.org/plus/dsa/problems/print-all-nodes-at-a-distance-of-k-in-bt?subject=dsa&approach=graphs-and-types
    def all_nodes_at_distance_k(self, node, k):
        pass

    # https://takeuforward.org/plus/dsa/problems/minimum-time-taken-to-burn-the-bt-from-a-given-node?subject=dsa&approach=graphs-and-types
    def minimum_time_to_burn_from(self, node):
        pass

    # https://takeuforward.org/plus/dsa/problems/count-total-nodes-in-a-complete-bt?subject=dsa&approach=graphs-and-types
    def count_nodes_in_cbt(self):
              
        def helper(node: TreeNode):
            if node is None:
                return 0
            
            lh = height(node, True)
            rh = height(node, False)

            if lh == rh:
                return (1 << (lh+1)) - 1
            
            return 1 + helper(node.left) + helper(node.right)
        
        def height(node: TreeNode, left_dir: bool):
            h = -1

            while node:
                node = node.left if left_dir else node.right
                h += 1
            
            return h
        
        return helper(self.root)

    # https://takeuforward.org/plus/dsa/problems/requirements-needed-to-construct-a-unique-bt?subject=dsa&approach=graphs-and-types
    def unique_bianry_tree(self):
        pass

    # https://takeuforward.org/plus/dsa/problems/construct-a-bt-from-preorder-and-inorder?subject=dsa&approach=graphs-and-types
    def build_tree_from_pre_order_and_in_order(self, pre_order: list[int], in_order: list[int]):
        """
            Constructs a tree from a pre-order and in-order traversal.
            Args:
                pre_order: Pre-order traversal list.
                in_order: In-order traversal list.
            Returns:
                TreeNode: The root of the constructed tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if not pre_order or not in_order or len(pre_order) != len(in_order):
            return None
        
        # Build hash map for O(1) lookup
        in_order_map = {val: idx for idx, val in enumerate(in_order)}
        pre_index = 0
        
        def helper(in_start: int, in_end: int) -> TreeNode | None:
            nonlocal pre_index
            if in_start > in_end or pre_index >= len(pre_order):
                return None
            
            root_value = pre_order[pre_index]
            root = TreeNode(root_value)
            pre_index += 1
            
            root_index = in_order_map[root_value]
            
            root.left = helper(in_start, root_index - 1)
            root.right = helper(root_index + 1, in_end)
            
            return root
        
        return helper(0, len(in_order) - 1)

    # https://takeuforward.org/plus/dsa/problems/construct-a-bt-from-postorder-and-inorder?subject=dsa&approach=graphs-and-types
    def build_tree_from_post_order_and_in_order(self, post_order: list[int], in_order: list[int]):
        """
            Constructs a tree from a post-order and in-order traversal.
            Args:
                post_order: Post-order traversal list.
                in_order: In-order traversal list.
            Returns:
                TreeNode: The root of the constructed tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        if not post_order or not in_order or len(post_order) != len(in_order):
            return None
        
        # Build hash map for O(1) lookup
        in_order_map = {val: idx for idx, val in enumerate(in_order)}
        post_index = len(post_order) - 1
        
        def helper(in_start: int, in_end: int) -> TreeNode | None:
            nonlocal post_index
            if in_start > in_end or post_index < 0:
                return None
            
            root_value = post_order[post_index]
            root = TreeNode(root_value)
            post_index -= 1
            
            root_index = in_order_map[root_value]
            
            # Build right subtree first (post-order: left, right, root)
            root.right = helper(root_index + 1, in_end)
            root.left = helper(in_start, root_index - 1)
            
            return root
        
        return helper(0, len(in_order) - 1) 

    # https://takeuforward.org/plus/dsa/problems/serialize-and-de-serialize-bt?subject=dsa&approach=graphs-and-types
    def serialize_deserialize_bt(self):
        pass

    # https://takeuforward.org/plus/dsa/problems/morris-inorder-traversal-?subject=dsa&approach=graphs-and-types
    def morris_inorder_traversal(self):
        pass

    # https://takeuforward.org/plus/dsa/problems/morris-preorder-traversal-?subject=dsa&approach=graphs-and-types
    def morris_preorder_traversal(self):
        pass

    # https://www.geeksforgeeks.org/problems/transform-to-sum-tree/1
    def convert_into_sum_tree(self):
        """
            Converts the tree into a sum tree where each node contains sum of its children.
            Returns:
                TreeNode: The root of the tree.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def convert_helper(node: TreeNode | None) -> int:
            if node is None:
                return 0
            
            # Store original value
            old_value = node.data
            
            # Get sum of children
            left_sum = convert_helper(node.left)
            right_sum = convert_helper(node.right)
            
            # Update node to sum of children
            node.data = left_sum + right_sum
            
            # Return sum including original value
            return old_value + left_sum + right_sum
        
        convert_helper(self.root)
        return self.root

    def kth_ancestor(self, k: int, value: int):
        """
            Finds the kth ancestor of a node with given value.
            Args:
                k: The ancestor level (1 = parent, 2 = grandparent, etc.)
                value: The value of the target node.
            Returns:
                int | None: The kth ancestor value, or None if not found.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        ans: int | None = None
        
        def helper(node: TreeNode | None, target_value: int) -> bool:
            nonlocal ans, k
            if node is None:
                return False
            
            if node.data == target_value:
                return True
            
            # Search in left and right subtrees
            found = helper(node.left, target_value) or helper(node.right, target_value)
            
            if found:
                k -= 1
                if k == 0:
                    ans = node.data
                return True
            
            return False
        
        helper(self.root, value)
        return ans
    
    # https://www.geeksforgeeks.org/dsa/diagonal-traversal-of-binary-tree/
    def diagonal_traversal(self):
        pass
    
    # https://leetcode.com/problems/flatten-binary-tree-to-linked-list/description/
    def flatten_bt_into_linked_list(self):
        pass

    # https://www.geeksforgeeks.org/dsa/maximum-sum-nodes-binary-tree-no-two-adjacent/
    def maximum_sum_of_non_adjacent_nodes(self):
        pass

    # https://www.geeksforgeeks.org/problems/sum-of-the-longest-bloodline-of-a-tree/1
    def sum_of_longest_root_to_leaf_path(self):
        pass

# https://takeuforward.org/plus/dsa/problems/check-if-two-trees-are-identical-or-not?subject=dsa&approach=graphs-and-types
def is_same_tree(p: TreeNode | None, q: TreeNode | None) -> bool:
    if p is None and q is None:
        return True
    
    if p is None or q is None:
        return False
    
    if p.data != q.data:
        return False
    
    return is_same_tree(p.left, q.left) and is_same_tree(p.right, q.right)
