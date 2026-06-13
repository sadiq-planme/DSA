from dataclasses import dataclass


@dataclass
class TreeNode:
    key: int
    left: TreeNode | None = None
    right: TreeNode | None = None


class BST:

    def __init__(self):
        self.root: TreeNode | None = None

    # TC: O(h) where h => log(n) in ave and h => n in worst case SC: O(1)
    def insert(self, key: int):
        if self.root is None:
            self.root = TreeNode(key)
            return
        
        curr = self.root
        while True:
            if key == curr.key:
                return
            elif key < curr.key:
                if curr.left is None:
                    curr.left = TreeNode(key)
                    return
                curr = curr.left
            else:
                if curr.right is None:
                    curr.right = TreeNode(key)
                    return
                curr = curr.right

    # TC: O(nh) where h => log(n) in ave case(O(n log n)) and h => n in worst case(O(n^2)) SC: O(n)
    def build(self, keys: list[int]):
        for key in keys:
            self.insert(key)

    # TC: O(h) where h => log(n) in ave and h => n in worst case SC: O(1)
    def search(self, key: int):
        if self.root:
            curr = self.root

            while curr:
                if key == curr.key:
                    return curr
                elif key < curr.key:
                    curr = curr.left
                else:
                    curr = curr.right
        
        return None

    # TC: O(h) where h => log(n) in ave and h => n in worst case SC: O(1)
    def min(self, node: TreeNode):
        curr = node
        while curr.left:
            curr = curr.left
        return curr.key

    # TC: O(h) where h => log(n) in ave and h => n in worst case SC: O(1)
    def max(self, node: TreeNode):
        curr = node
        while curr.right:
            curr = curr.right
        return curr.key

    # TC: O(n) | SC: O(n) due to result list and O(h) recursion stack
    def inorder(self):
        result: list[int] = []
        def helper(node: TreeNode | None):
            if node is None:
                return
            helper(node.left)
            result.append(node.key)
            helper(node.right)

        if self.root:
            curr = self.root
            helper(curr)

        return result

    # SC & TC: O(h) where h => log(n) in ave and h => n in worst case
    def delete(self, key: int):
        def helper(node: TreeNode | None, key: int):
            if node is None:
                return None
            elif key < node.key:
                node.left = helper(node.left, key)
            elif key > node.key:
                node.right = helper(node.right, key)
            else:
                # leaf node or single child case
                if node.left is None:
                    return node.right
                if node.right is None:
                    return node.left
                
                # 2 child case
                succ = self.min(node.right)
                node.key = succ
                node.right = helper(node.right, succ)
            return node
        
        self.root = helper(self.root, key)
