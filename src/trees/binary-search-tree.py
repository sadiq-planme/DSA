from __future__ import annotations
from dataclasses import dataclass


@dataclass
class TreeNode:
    data: int
    left: TreeNode | None = None
    right: TreeNode | None = None


@dataclass
class TreeNodeMetadata:
    min: float
    max: float
    size: int
    is_bst: bool


class BinarySearchTree:

    def __init__(self):
        self.root: TreeNode | None = None

    def build_bst(self, nodes: list[int]):
        """
            Builds a BST from a list of nodes.
            Args:
                nodes: The list of nodes to build the BST from.
            Returns:
                TreeNode: The root of the BST.
            Time Complexity: O(n log n)
            Space Complexity: O(n)
        """
        def insert(node: TreeNode | None, data: int):
            if node is None:
                return TreeNode(data)
            if data < node.data:
                node.left = insert(node.left, data)
            else:
                node.right = insert(node.right, data)
            return node
        
        for node in nodes:
            self.root = insert(self.root, node)
        
        return self.root

    def search(self, data: int) -> bool:
        """
            Searches for a node in the BST.
            Args:
                data: The data of the node to search for.
            Returns:
                bool: True if the node exists, False otherwise.
            Time Complexity: O(h) where h is height, O(log n) average, O(n) worst case
            Space Complexity: O(h) for recursion, O(1) for iterative version
        """
        def helper(node: TreeNode | None, data: int):
            if node is None:
                return False
            if node.data == data:
                return True
            if data < node.data:
                return helper(node.left, data)
            else:
                return helper(node.right, data)
        
        return helper(self.root, data)

    def min(self):
        """
            Finds the minimum node in the BST.
            Returns:
                int: The minimum value in the BST.
            Raises:
                ValueError: If the tree is empty.
            Time Complexity: O(log n)
            Space Complexity: O(1)
        """
        if self.root is None:
            raise ValueError("Cannot find minimum in empty tree")
        temp = self.root
        while temp.left is not None:
            temp = temp.left
        return temp.data

    def max(self) -> int:
        """
            Finds the maximum value in the BST.
            Returns:
                int: The maximum value in the BST.
            Raises:
                ValueError: If the tree is empty.
            Time Complexity: O(h) where h is height, O(log n) average, O(n) worst case
            Space Complexity: O(1)
        """
        if self.root is None:
            raise ValueError("Cannot find maximum in empty tree")
        temp = self.root
        while temp.right is not None:
            temp = temp.right
        return temp.data

    def delete(self, data: int):
        """
            Deletes a node from the BST.
            Args:
                data: The data of the node to delete.
            Returns:
                TreeNode: The root of the BST.
            Time Complexity: O(log n)
            Space Complexity: O(log n)
        """
        def find_successor(node: TreeNode) -> int:
            """Finds the in-order successor (minimum in right subtree)."""
            temp = node
            while temp.left is not None:
                temp = temp.left
            return temp.data

        def helper(node: TreeNode | None, data: int):
            # base case
            if node is None:
                return None
            # recursive case
            if data < node.data:
                node.left = helper(node.left, data)
            elif data > node.data:
                node.right = helper(node.right, data)
            else:
                # Handling leaf node
                if (node.left is None) and (node.right is None):
                    return None
                # Handling internal node with one child
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left
                # Handling internal node with two children
                successor = find_successor(node.right)
                node.data = successor
                node.right = helper(node.right, successor)
            return node

        self.root = helper(self.root, data)
        return self.root

    def is_bst(self):
        """
            Checks if the BST is valid.
            Returns:
                bool: True if the BST is valid, False otherwise.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def helper(node: TreeNode | None, min_val: int, max_val: int):
            if node is None:
                return True
            if node.data <= min_val or node.data >= max_val:
                return False
            left_valid = helper(node.left, min_val, node.data)
            right_valid = helper(node.right, node.data, max_val)
            return left_valid and right_valid
        
        return helper(self.root, float('-inf'), float('inf'))

    def lca(self, value1: int, value2: int):
        """
            Finds the lowest common ancestor of two nodes.
            Args:
                value1: First node value.
                value2: Second node value.
            Returns:
                TreeNode: The lowest common ancestor node, None if not found.
            Time Complexity: O(h) where h is height, O(n) worst case
            Space Complexity: O(h)
        """
        def helper(node: TreeNode | None, value1: int, value2: int) -> TreeNode | None:
            if node is None:
                return None
            if value1 < node.data and value2 < node.data:
                return helper(node.left, value1, value2)
            if value1 > node.data and value2 > node.data:
                return helper(node.right, value1, value2)
            return node
        
        return helper(self.root, value1, value2)

    def kth_smallest(self, k: int):
        """
            Finds the kth smallest node in the BST.
            Args:
                k: The position (1-indexed) of the smallest element to find.
            Returns:
                int: The kth smallest value in the BST, None if not found.
            Time Complexity: O(h + k) where h is height, O(n) worst case
            Space Complexity: O(h)
        """
        if k <= 0:
            return None
        
        count = [0]  # Use list to allow modification in nested function
        
        def helper(node: TreeNode | None) -> int | None:
            if node is None:
                return None
            
            # Traverse left subtree
            result = helper(node.left)
            if result is not None:
                return result
            
            # Process current node
            count[0] += 1
            if count[0] == k:
                return node.data
            
            # Traverse right subtree
            return helper(node.right)
        
        return helper(self.root)

    def kth_largest(self, k: int):
        """
            Finds the kth largest node in the BST.
            Args:
                k: The position (1-indexed) of the largest element to find.
            Returns:
                int: The kth largest value in the BST, None if not found.
            Time Complexity: O(h + k) where h is height, O(n) worst case
            Space Complexity: O(h)
        """
        if k <= 0:
            return None
        
        count = [0]  # Use list to allow modification in nested function
        
        def helper(node: TreeNode | None) -> int | None:
            if node is None:
                return None
            
            # Traverse right subtree first (reverse in-order)
            result = helper(node.right)
            if result is not None:
                return result
            
            # Process current node
            count[0] += 1
            if count[0] == k:
                return node.data
            
            # Traverse left subtree
            return helper(node.left)
        
        return helper(self.root)

    def build_balanced_bst_from_in_order(self, in_order: list[int]):
        """
            Builds a balanced BST from a sorted in-order traversal array.
            Args:
                in_order: Sorted list of integers (in-order traversal).
            Returns:
                TreeNode: The root of the balanced BST, None if in_order is empty.
            Time Complexity: O(n)
            Space Complexity: O(log n)
        """
        if not in_order:
            self.root = None
            return None
        
        def helper(start: int, end: int) -> TreeNode | None:
            if start > end:
                return None
            mid = (start + end) // 2
            parent = TreeNode(in_order[mid])
            parent.left = helper(start, mid - 1)
            parent.right = helper(mid + 1, end)
            return parent
        
        self.root = helper(0, len(in_order) - 1)
        return self.root

    # TODO: HW: convert BST into a balanced BST

    def inorder_traversal(self):
        """
            Performs In Order Traversal (DFS) traversal. Explores the tree in the order: left, root, right.
            Returns:
                list[int]: The list of nodes in in order traversal.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = []
        def helper(node: TreeNode | None):
            if node is None:
                return
            helper(node.left)
            result.append(node.data)
            helper(node.right)
        
        helper(self.root)
        return result

    def two_sum(self, target: int):
        """
            Finds two nodes in the BST that sum to the target.
            Returns:
                tuple[int, int]: The two nodes that sum to the target, None if not found.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        result: list[int] = self.inorder_traversal()
        left = 0
        right = len(result) - 1
        while left < right:
            if result[left] + result[right] == target:
                return (result[left], result[right])
            elif result[left] + result[right] < target:
                left += 1
            else:
                right -= 1
        return None

    def convert_bst_into_sorted_dll(self):
        """
            Converts the BST into a sorted doubly linked list (DLL).
            Uses reverse in-order traversal to build the DLL.
            Returns:
                TreeNode: The head (leftmost node) of the DLL, None if tree is empty.
            Time Complexity: O(n)
            Space Complexity: O(h) where h is height
        """
        head: TreeNode | None = None
        
        def helper(node: TreeNode | None):
            nonlocal head
            if node is None:
                return
            # Reverse in-order: right, root, left
            helper(node.right)
            # Link current node to head
            node.right = head  # right => next
            if head is not None:
                head.left = node  # left => prev
            head = node
            helper(node.left)
        
        helper(self.root)
        return head

    def convert_sorted_dll_into_bst(self, head: TreeNode | None, size: int):
        """
            Converts the sorted DLL into a BST.
            Returns:
                TreeNode: The root of the BST.
            Time Complexity: O(n)
            Space Complexity: O(n)
        """
        def helper(n: int):
            nonlocal head
            if n <= 0 or head is None:
                return None
            # create the left subtree
            left_subtree = helper(n // 2)
            # create the root node
            root = head
            root.left = left_subtree
            head = head.right # move head to the next node
            # create the right subtree
            root.right = helper(n - n // 2 - 1)
            return root

        self.root = helper(size)
        return self.root

    def largest_bst_subtree_size_in_bt(self):
        """
            Finds the size of the largest BST subtree in the binary tree.
            Returns:
                int: The size of the largest BST subtree.
            Time Complexity: O(n)
            Space Complexity: O(h) where h is height
        """
        size_of_largest_bst: int = 0
        
        def helper(node: TreeNode | None) -> TreeNodeMetadata:
            nonlocal size_of_largest_bst
            if node is None:
                return TreeNodeMetadata(
                    min=float('inf'),
                    max=float('-inf'),
                    size=0,
                    is_bst=True
                )
            
            left_metadata = helper(node.left)
            right_metadata = helper(node.right)
            
            # Check if current subtree is a valid BST
            if (
                left_metadata.is_bst and
                right_metadata.is_bst and
                node.data > left_metadata.max and
                node.data < right_metadata.min
            ):
                current_size = left_metadata.size + right_metadata.size + 1
                size_of_largest_bst = max(size_of_largest_bst, current_size)
                return TreeNodeMetadata(
                    min=min(node.data, left_metadata.min) if left_metadata.size > 0 else node.data,
                    max=max(node.data, right_metadata.max) if right_metadata.size > 0 else node.data,
                    size=current_size,
                    is_bst=True
                )
            
            # Current subtree is not a BST, but track largest BST found in subtrees
            size_of_largest_bst = max(
                size_of_largest_bst,
                left_metadata.size,
                right_metadata.size
            )
            return TreeNodeMetadata(
                min=float('inf'),
                max=float('-inf'),
                size=0,
                is_bst=False
            )
        
        helper(self.root)
        return size_of_largest_bst
