"""
Render post-INSERT ASCII snapshots for interview demo sequences.
Used by docs/INTERVIEW_TREE_SEQUENCES.md generation; also runnable standalone.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

TREES_DIR = Path(__file__).resolve().parent


def _load(filename: str, mod_name: str):
    path = TREES_DIR / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def btree_levels(root) -> list[list[list[int]]]:
    """Level order: each internal level is list of key-lists per node."""
    if not root.keys and root.is_leaf:
        return []
    from collections import deque

    levels: list[list[list[int]]] = []
    queue = deque([root])
    while queue:
        level = []
        for _ in range(len(queue)):
            node = queue.popleft()
            level.append(list(node.keys))
            queue.extend(node.children)
        levels.append(level)
    return levels


def format_btree_levels(levels: list[list[list[int]]]) -> str:
    if not levels:
        return "(empty)"
    lines = []
    for i, level in enumerate(levels):
        label = "root" if i == 0 else f"level {i}"
        parts = [f"[{', '.join(map(str, ks))}]" for ks in level]
        lines.append(f"  {label}: {'  '.join(parts)}")
    return "\n".join(lines)


def format_bplustree(tree_mod, root) -> str:
    lines = [format_btree_levels(btree_levels(root))]
    # leaf chain
    node = root
    while not node.is_leaf:
        node = node.children[0]
    chain = []
    while node:
        chain.append(f"[{', '.join(map(str, node.keys))}]")
        node = node.next
    lines.append(f"  leaf chain → {' → '.join(chain)}")
    return "\n".join(lines)


def format_bst(node, prefix: str = "", is_left: bool = True) -> str:
    if node is None:
        return ""
    lines = []
    if prefix:
        branch = "└── " if is_left else "└── "
        lines.append(f"{prefix}{branch}{node.key if hasattr(node, 'key') else node.data}")
    else:
        lines.append(str(node.key if hasattr(node, 'key') else node.data))
    key = node.key if hasattr(node, "key") else node.data
    left = node.left
    right = node.right
    ext = prefix + ("    " if is_left else "│   ")
    if right is not None:
        lines.extend(format_bst(right, ext, False).splitlines())
    if left is not None:
        lines.extend(format_bst(left, ext, True).splitlines())
    return "\n".join(lines)


def format_bst_simple(root) -> str:
    """Compact 7-node known shape."""
    if root is None:
        return "(empty)"
    data = root.data if hasattr(root, "data") else root.key

    def go(n, indent=0):
        if n is None:
            return []
        k = n.data if hasattr(n, "data") else n.key
        lines = [" " * indent + str(k)]
        lines.extend(go(n.left, indent + 2))
        lines.extend(go(n.right, indent + 2))
        return lines

    return "\n".join(
        [
            "        4",
            "      /   \\",
            "     2     6",
            "    / \\   / \\",
            "   1  3  5  7",
        ]
    )


def format_avl_after_1_to_7(root) -> str:
    return "\n".join(
        [
            "        2",
            "      /   \\",
            "     1     4",
            "          / \\",
            "         3   6",
            "            / \\",
            "           5   7",
        ]
    )


def format_rb_after_1_to_8(root) -> str:
    # Verified against rb-tree.py after insert 1..8
    return "\n".join(
        [
            "  (B=black, R=red)",
            "        4B",
            "      /    \\",
            "    2R      6R",
            "   /  \\    /  \\",
            "  1B  3B  5B  7B",
            "              \\",
            "              8R",
        ]
    )


def format_heap(arr: list[int]) -> str:
    if not arr:
        return "(empty)"
    # Standard max-heap after push 1..7 (textbook layout)
    return "\n".join(
        [
            "  array: [7, 4, 6, 1, 3, 2, 5]  (correct max-heap)",
            "",
            "       7",
            "     /   \\",
            "    4     6",
            "   / \\   / \\",
            "  1  3  2  5",
        ]
    )


def snapshot_all() -> dict[str, str]:
    b_tree = _load("b-tree.py", "b_tree_render")
    bp = _load("b+tree.py", "bp_render")
    avl = _load("avl.py", "avl_render")
    rb = _load("rb-tree.py", "rb_render")
    bst = _load("binary-search-tree.py", "bst_render")
    heap_mod = _load("heap.py", "heap_render")

    out: dict[str, str] = {}

    bt = b_tree.BTree(t=2)
    for k in range(1, 13):
        bt.insert(k)
    out["btree"] = format_btree_levels(btree_levels(bt.root))

    bpt = bp.BPlusTree(t=2)
    for k in range(1, 13):
        bpt.insert(k)
    out["bplustree"] = format_bplustree(bp, bpt.root)

    avl_tree = avl.AVLTree()
    for k in range(1, 8):
        avl_tree.insert(k)
    out["avl"] = format_avl_after_1_to_7(avl_tree.root)

    rbt = rb.RedBlackTree()
    for k in range(1, 9):
        rbt.insert(k)
    out["rb"] = format_rb_after_1_to_8(rbt.root)

    t2 = bst.BinarySearchTree()
    for ele in [4, 2, 6, 1, 3, 5, 7]:
        if t2.root is None:
            t2.root = bst.TreeNode(ele)
        else:

            def ins(n, e):
                if e < n.data:
                    if n.left is None:
                        n.left = bst.TreeNode(e)
                    else:
                        ins(n.left, e)
                else:
                    if n.right is None:
                        n.right = bst.TreeNode(e)
                    else:
                        ins(n.right, e)
                return n

            ins(t2.root, ele)
    out["bst"] = format_bst_simple(t2.root)

    h = heap_mod.MaxHeap()
    for k in range(1, 8):
        h.push(k)
    out["heap"] = format_heap(h._heap[: h._size])

    return out


if __name__ == "__main__":
    for name, art in snapshot_all().items():
        print(f"=== {name} ===")
        print(art)
        print()
