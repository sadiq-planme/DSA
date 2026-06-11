from __future__ import annotations
from dataclasses import dataclass


@dataclass
class AVLNode:
    key: int
    left: AVLNode | None = None
    right: AVLNode | None = None
    height: int = 1


class AVLTree: 
    def __init__(self) -> None:
        self.root: AVLNode | None = None

    # ---------------------------- Insertion ----------------------------
    def insert(self, key: int) -> AVLNode | None:
        self.root = self._insert(self.root, key)
        return self.root

    def _insert(self, node: AVLNode | None, key: int) -> AVLNode:
        if node is None:
            return AVLNode(key)

        if key < node.key:
            node.left = self._insert(node.left, key)
        else:
            # duplicates go right
            node.right = self._insert(node.right, key)

        return self._rebalance(node)

    def _rebalance(self, node: AVLNode) -> AVLNode:
        """
            Rebalances a node after insert/delete.

            This compact decision tree covers:
            - Insertion: LL/RR/LR/RL
            - Deletion: L1/L0/L-1 and R1/R0/R-1 via child balance factor checks
        """
        self._update_height(node)
        bf = self._balance_factor(node)

        # Left heavy
        if bf == 2:
            if self._balance_factor(node.left) == -1:  # L-1 -> LR Imbalance Case
                node.left = self._rotate_left(node.left)  
            return self._rotate_right(node)  # L1/L0 -> LL Imbalance Case

        # Right heavy
        if bf == -2:
            if self._balance_factor(node.right) == 1:  # R1 -> RL Imbalance Case
                node.right = self._rotate_right(node.right)  
            return self._rotate_left(node)  # R-1/R0 -> RR Imbalance Case

        return node

    def _update_height(self, node: AVLNode) -> None:
        height_of_left_subtree = self._height(node.left)
        height_of_right_subtree = self._height(node.right)
        node.height = 1 + max(height_of_left_subtree, height_of_right_subtree)

    def _balance_factor(self, node: AVLNode | None) -> int:
        if node is None:
            return 0
        height_of_left_subtree = self._height(node.left)
        height_of_right_subtree = self._height(node.right)
        balance_factor = height_of_left_subtree - height_of_right_subtree
        return balance_factor

    # Handle LL Imbalance Case, node p as pivot
    def _rotate_right(self, p: AVLNode) -> AVLNode:
        pl = p.left
        plr = pl.right

        # Perform right rotation
        p.left = plr
        pl.right = p

        # Update heights bottom-up
        self._update_height(p)
        self._update_height(pl)
        
        return pl

    # Handle RR Imbalance Case, node p as pivot
    def _rotate_left(self, p: AVLNode) -> AVLNode:
        pr = p.right
        prl = pr.left

        # Perform left rotation
        pr.left = p
        p.right = prl

        # Update heights bottom-up
        self._update_height(p)
        self._update_height(pr)
        
        return pr

    @staticmethod
    def _height(node: AVLNode | None) -> int:
        return node.height if node else 0

    # ---------------------------- Deletion ----------------------------
    def delete(self, key: int) -> AVLNode | None:
        self.root = self._delete(self.root, key)
        return self.root

    def _min_node(self, node: AVLNode) -> AVLNode:
        curr = node
        while curr.left is not None:
            curr = curr.left
        return curr

    def _delete(self, node: AVLNode | None, key: int) -> AVLNode | None:
        if node is None:
            return None

        if key < node.key:
            node.left = self._delete(node.left, key)
        elif key > node.key:
            node.right = self._delete(node.right, key)
        else:
            # Found node to delete
            if node.left is None:
                return node.right
            if node.right is None:
                return node.left

            # Two children: replace with inorder successor (min in right subtree)
            succ = self._min_node(node.right)
            node.key = succ.key
            node.right = self._delete(node.right, succ.key)

        # If we actually deleted the last node in this subtree
        if node is None:
            return None

        return self._rebalance(node)
