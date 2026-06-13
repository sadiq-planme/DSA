
from dataclasses import dataclass, field
from collections import deque


@dataclass
class BPlusNode:
    keys: list[int] = field(default_factory=list)
    children: list[BPlusNode] = field(default_factory=list)
    next: BPlusNode | None = None
    
    @property
    def is_leaf(self) -> bool:
        return not self.children


class BPlusTree:
    """
        t <= degree of each node <= 2t
        t-1 <= keys count of each node <= 2t-1
        Exceptions:
            1. Min degree of root node is 2. Hence, min keys count of root node is 1.
            2. The degree of leaf node is 0. But it can have keys, as per the keys count constraint above.
    """
    def __init__(self, t: int = 2) -> None:
        if t < 2:
            raise ValueError(f"Minimum degree t must be ≥ 2, got {t}.")
        self.t = t
        self.root = BPlusNode()

    # ---------------------------- Insertion ----------------------------
    def insert(self, key: int) -> None:
        # split root if it is full
        if self._is_full(self.root):
            old_root  = self.root
            self.root = BPlusNode()
            self.root.children.append(old_root)
            # old_root will become the left child for the new root
            self._split_child(self.root, 0)
        self._insert_non_full(self.root, key)

    # ---------------------------- Helpers ----------------------------
    def _is_full(self, node: BPlusNode) -> bool:
        return len(node.keys) == 2 * self.t - 1

    def _split_child(self, parent: BPlusNode, child_idx: int) -> None:
        if parent.children[child_idx].is_leaf:
            self._split_leaf(parent, child_idx)
        else:
            self._split_internal(parent, child_idx)

    def _insert_non_full(self, node: BPlusNode, key: int) -> None:
        if node.is_leaf:
            
            key_idx = self._find_left_most_occurrence_index(node.keys, key)
            
            if key_idx < len(node.keys) and node.keys[key_idx] == key:
                raise ValueError(f"Key {key} already exists in the tree.")
            
            node.keys.insert(key_idx, key)
        else:
            # bisect_right: key == separator belongs in right child
            key_idx = self._find_right_most_occurrence_index(node.keys, key)
            
            key_idx = self._fix_full_child(node, key_idx, key)
            self._insert_non_full(node.children[key_idx], key)

    def _split_leaf(self, parent: BPlusNode, leaf_idx: int) -> None:
        leaf = parent.children[leaf_idx]
        mid_idx = self.t - 1 # left keeps first t-1 keys; right gets the remaining t

        right = BPlusNode()
        right.keys = leaf.keys[mid_idx:]
        leaf.keys = leaf.keys[:mid_idx]
        
        # Stitch linked list
        right.next = leaf.next           
        leaf.next = right

        # Copy (not move) the first key of right up as separator.
        parent.keys.insert(leaf_idx, right.keys[0])
        parent.children.insert(leaf_idx + 1, right)

    def _split_internal(self, parent: BPlusNode, child_idx: int) -> None:
        child = parent.children[child_idx]
        mid_idx = self.t - 1
        mid_key = child.keys[mid_idx]

        right = BPlusNode()
        right.keys = child.keys[mid_idx + 1:]
        child.keys = child.keys[:mid_idx]

        right.children = child.children[mid_idx + 1:]
        child.children = child.children[:mid_idx + 1]

        parent.keys.insert(child_idx, mid_key)   # M moves up
        parent.children.insert(child_idx + 1, right)
    
    # it acts like bisect_left
    @staticmethod
    def _find_left_most_occurrence_index(keys: list[int], key: int) -> int:
        lo, hi = 0, len(keys)-1
        while lo <= hi:
            mid = (lo + hi) >> 1
            if key > keys[mid]:
                lo = mid + 1
            # Dont add '=' block here, because we want to find the left most occurrence of the key, not just the occurrence of the key.
            else:
                hi = mid - 1
        return lo

    # it acts like bisect_right
    @staticmethod
    def _find_right_most_occurrence_index(keys: list[int], key: int) -> int:
        lo, hi = 0, len(keys)-1
        while lo <= hi:
            mid = (lo + hi) >> 1
            # Don't remove '=' from the condition in this if block, because we want to find the right most occurrence of the key, not just the occurrence of the key.
            if key >= keys[mid]:
                lo = mid + 1
            else:
                hi = mid - 1
        return lo

    def _fix_full_child(self, parent: BPlusNode, child_idx: int, key: int) -> int:
        child = parent.children[child_idx]
        if self._is_full(child):
            self._split_child(parent, child_idx)
            # After split, the promoted key sits at parent.keys[child_idx].
            # Decide which of the two halves the new key belongs to.
            if key >= parent.keys[child_idx]:
                child_idx += 1
        return child_idx

    # ---------------------------- Deletion ----------------------------
    def delete(self, key: int) -> None:
        if not self.root.keys:
            return
        self._delete(self.root, key)
        # Root drained by a merge → shrink tree height by one level.
        if not self.root.keys and not self.root.is_leaf:
            self.root = self.root.children[0]

    # ---------------------------- Helpers ----------------------------
    def _delete(self, node: BPlusNode, key: int) -> None:
        if node.is_leaf:
            key_idx = self._find_left_most_occurrence_index(node.keys, key)
            
            if key_idx < len(node.keys) and node.keys[key_idx] == key:
                node.keys.pop(key_idx)
            
            return   # key absent → no-op

        # Navigate: bisect_right because separator == min of RIGHT child
        child_idx = self._find_right_most_occurrence_index(node.keys, key)
        child_idx = self._fix_deficient_child(node, child_idx)
        self._delete(node.children[child_idx], key)

    def _fix_deficient_child(self, parent: BPlusNode, child_idx: int) -> int:
        child = parent.children[child_idx]
        if len(child.keys) >= self.t:
            return child_idx    # already healthy

        left_idx  = child_idx - 1
        right_idx = child_idx + 1

        # Borrow from left sibling if it has enough keys
        if left_idx >= 0 and self._can_donate(parent.children[left_idx]):
            if child.is_leaf: 
                self._borrow_from_left_leaf(parent, child_idx)
            else:
                self._borrow_from_left_internal(parent, child_idx)
        
        # Borrow from right sibling if it has enough keys
        elif right_idx < len(parent.children) and self._can_donate(parent.children[right_idx]):
            if child.is_leaf: 
                self._borrow_from_right_leaf(parent, child_idx)
            else: 
                self._borrow_from_right_internal(parent, child_idx)
        
        # Merge the child into its left sibling
        elif left_idx >= 0:
            if child.is_leaf: 
                self._merge_leaves(parent, left_idx)
            else: 
                self._merge_internal(parent, left_idx)
            child_idx = left_idx # merged node sits at left_idx
        
        # No left sibling — merge right sibling into child
        else:
            if child.is_leaf: 
                self._merge_leaves(parent, child_idx)
            else: 
                self._merge_internal(parent, child_idx)

        # Returns the index of the child with enough keys >= self.t
        return child_idx

    def _can_donate(self, node: BPlusNode) -> bool:
        return len(node.keys) >= self.t

    def _borrow_from_left_leaf(self, parent: BPlusNode, child_idx: int) -> None:
        child = parent.children[child_idx]
        left_sib = parent.children[child_idx - 1]

        child.keys.insert(0, left_sib.keys.pop())
        parent.keys[child_idx - 1] = child.keys[0]   # new min of child

    def _borrow_from_right_leaf(self, parent: BPlusNode, child_idx: int) -> None:
        child = parent.children[child_idx]
        right_sib = parent.children[child_idx + 1]
        
        child.keys.append(right_sib.keys.pop(0))
        parent.keys[child_idx] = right_sib.keys[0]   # right_sib's new min

    # Merges right leaf into left leaf
    def _merge_leaves(self, parent: BPlusNode, left_idx: int) -> None:
        left = parent.children[left_idx]
        right = parent.children[left_idx + 1]

        left.keys.extend(right.keys)
        left.next = right.next          # stitch linked list, skip right

        parent.keys.pop(left_idx)       # remove separator
        parent.children.pop(left_idx + 1)

    # same as _borrow_from_left_sibling in B-tree
    def _borrow_from_left_internal(self, parent: BPlusNode, child_idx: int) -> None:
        child = parent.children[child_idx]
        left_sib = parent.children[child_idx - 1]

        child.keys.insert(0, parent.keys[child_idx - 1])
        parent.keys[child_idx - 1] = left_sib.keys.pop()
        child.children.insert(0, left_sib.children.pop())

    # same as _borrow_from_right_sibling in B-tree
    def _borrow_from_right_internal(self, parent: BPlusNode, child_idx: int) -> None:
        child = parent.children[child_idx]
        right_sib = parent.children[child_idx + 1]

        child.keys.append(parent.keys[child_idx])
        parent.keys[child_idx] = right_sib.keys.pop(0)
        child.children.append(right_sib.children.pop(0))

    # same as _merge in B-tree
    def _merge_internal(self, parent: BPlusNode, left_idx: int) -> None:
        left = parent.children[left_idx]
        right = parent.children[left_idx + 1]
        sep = parent.keys.pop(left_idx)
        parent.children.pop(left_idx + 1)

        left.keys.append(sep)
        left.keys.extend(right.keys)
        left.children.extend(right.children)

    def level_order_traversal(self) -> None:
        if not self.root or not self.root.keys:
            print("Tree is empty")
            return

        queue = deque([self.root])
        
        while queue:
            level_size = len(queue)
            level_strings = []
            
            for _ in range(level_size):
                node = queue.popleft()
                # Format node keys as [key1, key2]
                level_strings.append(str(node.keys))
                
                # Queue up children for the next level
                for child in node.children:
                    queue.append(child)
            
            # Print the entire level separated by spaces
            print("   ".join(level_strings))


if __name__ == "__main__":
    tree = BPlusTree(t=2)
    for key in range(1, 9):
        tree.insert(key)
    tree.level_order_traversal()
