from __future__ import annotations
from collections import deque
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
