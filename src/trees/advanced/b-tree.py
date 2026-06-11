from __future__ import annotations
from dataclasses import dataclass, field
from collections import deque

@dataclass
class BTreeNode:
    keys: list[int] = field(default_factory=list)
    children: list[BTreeNode] = field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children


class BTree:
    """
        t <= degree of each node <= 2t
        t-1 <= keys count of each node <= 2t-1
        Exceptions:
            1. Min degree of root node is 2. Hence, min keys count of root node is 1.
            2. The degree of leaf node is 0. But it can have keys, as per the keys count constraint above.
    """
    def __init__(self, t: int = 2) -> None:
        if t < 2:
            raise ValueError(f"Minimum degree t must be >= 2, got {t}.")
        self.t = t
        self.root = BTreeNode()

    # ---------------------------- Insertion ----------------------------
    def insert(self, key: int) -> None:
        # split root if it is full
        if self._is_full(self.root):
            old_root = self.root
            self.root = BTreeNode()
            self.root.children.append(old_root)
            # old_root will become the left child for the new root
            self._split_child(self.root, 0)      
        self._insert_non_full(self.root, key)

    # ---------------------------- Helpers ----------------------------
    def _is_full(self, node: BTreeNode) -> bool:
        return len(node.keys) == 2 * self.t - 1

    def _split_child(self, parent: BTreeNode, child_idx: int) -> None:
        child = parent.children[child_idx]
        mid_idx = self.t - 1
        mid_key = child.keys[mid_idx]

        right = BTreeNode()
        right.keys = child.keys[mid_idx + 1:]
        child.keys = child.keys[:mid_idx]
        
        if not child.is_leaf:
            right.children = child.children[mid_idx + 1:]
            child.children = child.children[:mid_idx + 1]

        parent.keys.insert(child_idx, mid_key)
        parent.children.insert(child_idx + 1, right)

    def _insert_non_full(self, node: BTreeNode, key: int) -> None:
        key_idx = self._find_index(node.keys, key)

        if key_idx < len(node.keys) and node.keys[key_idx] == key:
            raise ValueError(f"Key {key} already exists in the B-tree.")

        if node.is_leaf:
            node.keys.insert(key_idx, key)
        else:
            key_idx = self._fix_full_child(node, key_idx, key)
            self._insert_non_full(node.children[key_idx], key)

    @staticmethod
    def _find_index(keys: list[int], key: int) -> int:
        lo, hi = 0, len(keys)-1
        while lo <= hi:
            mid = (lo + hi) >> 1
            if key > keys[mid]:
                lo = mid + 1
            elif key < keys[mid]:
                hi = mid - 1
            else:
                return mid
        return lo

    def _fix_full_child(self, parent: BTreeNode, child_idx: int, key: int) -> int:
        child = parent.children[child_idx]
        if self._is_full(child):
            self._split_child(parent, child_idx)
            # After split, the promoted key sits at parent.keys[child_idx].
            # Decide which of the two halves the new key belongs to.
            if key > parent.keys[child_idx]:
                child_idx += 1
        return child_idx

    # ---------------------------- Deletion ----------------------------
    def delete(self, key: int) -> None:
        if not self.root.keys:
            return
        self._delete(self.root, key)
        # If the root was completely drained by a merge, shrink the tree height.
        if not self.root.keys and not self.root.is_leaf:
            self.root = self.root.children[0]

    # ---------------------------- Helpers ----------------------------
    def _delete(self, node: BTreeNode, key: int) -> None:
        key_idx = self._find_index(node.keys, key)

        if key_idx < len(node.keys) and node.keys[key_idx] == key:
            if node.is_leaf:
                node.keys.pop(key_idx)
            else:
                self._delete_from_internal(node, key_idx, key)
        else:
            if node.is_leaf:
                return   # key absent → no-op
            
            key_idx = self._fix_deficient_child(node, key_idx)
            self._delete(node.children[key_idx], key)

    def _delete_from_internal(self, node: BTreeNode, idx: int, key: int) -> None:
        left = node.children[idx]
        right = node.children[idx + 1]

        if self._can_donate(left):
            pred = self._get_max(left)
            node.keys[idx] = pred
            self._delete(left, pred)

        elif self._can_donate(right):
            succ = self._get_min(right)
            node.keys[idx] = succ
            self._delete(right, succ)

        else:
            # Merge the Right subtree root node of key into the Left subtree root node of key.
            merged = self._merge(node, idx)  # key is now inside merged
            self._delete(merged, key)

    def _fix_deficient_child(self, parent: BTreeNode, key_idx: int) -> int:
        child = parent.children[key_idx]
        if len(child.keys) >= self.t:
            return key_idx           # already has enough keys

        left_idx = key_idx - 1
        right_idx = key_idx + 1

        if left_idx >= 0 and self._can_donate(parent.children[left_idx]):
            self._borrow_from_left_sibling(parent, key_idx)

        elif (right_idx < len(parent.children) and self._can_donate(parent.children[right_idx])):
            self._borrow_from_right_sibling(parent, key_idx)

        elif left_idx >= 0:
            # Merge child into its left sibling and descend into merged node.
            self._merge(parent, left_idx)
            key_idx = left_idx

        else:
            # No left sibling - merge right sibling into child.
            self._merge(parent, key_idx)
            # key_idx unchanged; right was absorbed into child.

        return key_idx

    # Check if the node has enough keys to donate.
    def _can_donate(self, node: BTreeNode) -> bool:
        return len(node.keys) >= self.t

    # Borrow the left sibling's largest key through the parent.
    def _borrow_from_left_sibling(self, parent: BTreeNode, child_idx: int) -> None:
        child = parent.children[child_idx]
        left_sib = parent.children[child_idx - 1]

        child.keys.insert(0, parent.keys[child_idx - 1])
        parent.keys[child_idx - 1] = left_sib.keys.pop()
        if not left_sib.is_leaf:
            child.children.insert(0, left_sib.children.pop())

    # Borrow the right sibling's smallest key through the parent.
    def _borrow_from_right_sibling(self, parent: BTreeNode, child_idx: int) -> None:
        child = parent.children[child_idx]
        right_sib = parent.children[child_idx + 1]

        child.keys.append(parent.keys[child_idx])
        parent.keys[child_idx] = right_sib.keys.pop(0)
        if not right_sib.is_leaf:
            child.children.append(right_sib.children.pop(0))

    # Merge child at 1+left_idx index into child at left_idx index.
    def _merge(self, parent: BTreeNode, left_idx: int) -> BTreeNode:
        left = parent.children[left_idx]
        right = parent.children[left_idx + 1]
        sep = parent.keys.pop(left_idx)
        parent.children.pop(left_idx + 1)

        left.keys.append(sep)
        left.keys.extend(right.keys)
        if not right.is_leaf:
            left.children.extend(right.children)
        return left

    @staticmethod
    def _get_max(node: BTreeNode) -> int:
        while not node.is_leaf:
            node = node.children[-1]
        return node.keys[-1]

    @staticmethod
    def _get_min(node: BTreeNode) -> int:
        while not node.is_leaf:
            node = node.children[0]
        return node.keys[0]

    def level_order_traversal(self) -> None:
        if self.root is None:
            return None
        result: list[list[int]] = []
        queue = deque([self.root])
        while queue:
            level_size = len(queue)
            current_level: list[list[int]] = []
            for _ in range(level_size):
                current_node = queue.popleft()
                current_level.append(current_node.keys)
                for child in current_node.children:
                    queue.append(child)
            result.append(current_level)

        for index, level in enumerate(result):
            end = 0
            if index > 0:
                for last_level_keys in result[index - 1]:
                    length = len(last_level_keys)
                    print(f"    {level[end:end + length + 1]}    ", end="")
                    end += length + 1                
            else:
                print(f"    {level[0]}    ", end="")
            print()


if __name__ == "__main__":
    tree = BTree(t=2)
    for key in range(1, 9):
        tree.insert(key)
    tree.level_order_traversal()
