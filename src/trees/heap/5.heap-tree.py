
from dataclasses import dataclass
from collections import deque


# RARELY ASKED 1 QUESTION SOLVED HERE: MaxHeap with Binary Tree
@dataclass
class HeapTreeNode:
    val: int
    parent: HeapTreeNode | None = None
    left: HeapTreeNode | None = None
    right: HeapTreeNode | None = None


class MaxHeapWithBinaryTree:

    def __init__(self):
        self.root: HeapTreeNode | None = None

    # ********* Primary Methods *********
    def push(self, val: int):  # O(n + log n)
        """Add an element to the max heap using binary tree structure"""
        if self.root is None:
            self.root = HeapTreeNode(val)
        else:
            # Find the first available position (level-order insertion)
            queue = deque([self.root])
            while queue:
                current_node = queue.popleft()
                # Try to insert left
                if current_node.left is None:
                    current_node.left = HeapTreeNode(val, parent=current_node)
                    self._heapify_up(current_node.left)
                    break
                # Try to insert right
                if current_node.right is None:
                    current_node.right = HeapTreeNode(val, parent=current_node)
                    self._heapify_up(current_node.right)
                    break
                # Continue to next level
                queue.append(current_node.left)
                queue.append(current_node.right)

    def _heapify_up(self, node: HeapTreeNode):  # O(log n)
        """Bubble up the node to maintain max heap property"""
        while node.parent is not None and node.val > node.parent.val:
            node.val, node.parent.val = node.parent.val, node.val
            node = node.parent

    def pop(self):  # O(n + log n)
        """Remove and return the maximum element from the heap"""
        if self.root is None:
            return None
        
        max_val = self.root.val
        
        # Find the last node in the tree (Level Order Traversal)
        queue = deque([self.root])
        last_node = None
        while queue:
            last_node = queue.popleft()
            if last_node.left is not None:
                queue.append(last_node.left)
            if last_node.right is not None:
                queue.append(last_node.right)
        
        # Edge Case: Only one node (root only)
        if last_node == self.root:
            self.root = None
            return max_val
        
        # Move last node's value to root
        self.root.val = last_node.val

        # Remove the last node
        if last_node.parent.left == last_node:
            last_node.parent.left = None
        else:
            last_node.parent.right = None
        
        # Restore heap property downwards
        self._heapify(self.root)
        return max_val

    def _heapify(self, node: HeapTreeNode | None):  # O(log n)
        """Bubble down the node to maintain max heap property"""
        while node and node.left is not None:
            largest = node
            # Compare with left child
            if node.left.val > largest.val:
                largest = node.left
            # Compare with right child
            if node.right is not None and node.right.val > largest.val:
                largest = node.right
            # Swap if child is larger
            if largest != node:
                node.val, largest.val = largest.val, node.val
                node = largest
            else:
                break

    def build(self, vals: list[int]):  # O(n^2)
        """Build a max heap from a list of values"""
        for val in vals:
            self.push(val)

    # ********* Problems *********
    def is_heap(self):  # O(n)
        """
            Checks if the binary tree maintains max heap property.
            Returns:
                bool: True if valid max heap, False otherwise
            Time Complexity: O(n)
            Space Complexity: O(log n)
        """
        def helper(node: HeapTreeNode | None):
            if node is None:
                return True
            # Check left child
            if node.left is not None:
                if node.left.val > node.val or not helper(node.left):
                    return False
            # Check right child
            if node.right is not None:
                if node.right.val > node.val or not helper(node.right):
                    return False
            return True
        
        return helper(self.root)
