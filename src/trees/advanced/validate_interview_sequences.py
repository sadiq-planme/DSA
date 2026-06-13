"""
Verify interview INSERT/DELETE sequences hit expected B-tree / B+ tree edge cases (t=2).
Run: python src/trees/validate_interview_sequences.py
"""


import importlib.util
import sys
from pathlib import Path
from typing import Callable

TREES_DIR = Path(__file__).resolve().parent

INSERT_SEQ = list(range(1, 13))
# Interleave 10 before 3 so a deficient leaf borrows from the left sibling.
DELETE_SEQ = [1, 2, 10, 3, 11, 12, 6, 4, 5, 7, 8, 9]


def _load_module(filename: str, mod_name: str):
    path = TREES_DIR / filename
    spec = importlib.util.spec_from_file_location(mod_name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    spec.loader.exec_module(mod)
    return mod


def _wrap(tree, method: str, counter: dict[str, int], label: str) -> None:
    original: Callable = getattr(tree, method)

    def wrapper(*args, **kwargs):
        counter[label] = counter.get(label, 0) + 1
        return original(*args, **kwargs)

    setattr(tree, method, wrapper)


def _instrument_btree(tree, counter: dict[str, int]) -> None:
    _wrap(tree, "_split_child", counter, "split_child")
    _wrap(tree, "_borrow_from_left_sibling", counter, "borrow_left")
    _wrap(tree, "_borrow_from_right_sibling", counter, "borrow_right")
    _wrap(tree, "_merge", counter, "merge")
    _wrap(tree, "_get_max", counter, "predecessor")
    _wrap(tree, "_get_min", counter, "successor")


def _instrument_bplustree(tree, counter: dict[str, int]) -> None:
    _wrap(tree, "_split_leaf", counter, "split_leaf")
    _wrap(tree, "_split_internal", counter, "split_internal")
    _wrap(tree, "_borrow_from_left_leaf", counter, "borrow_left_leaf")
    _wrap(tree, "_borrow_from_right_leaf", counter, "borrow_right_leaf")
    _wrap(tree, "_borrow_from_left_internal", counter, "borrow_left_internal")
    _wrap(tree, "_borrow_from_right_internal", counter, "borrow_right_internal")
    _wrap(tree, "_merge_leaves", counter, "merge_leaf")
    _wrap(tree, "_merge_internal", counter, "merge_internal")


def _validate_btree() -> None:
    b_tree = _load_module("b-tree.py", "b_tree")
    tree = b_tree.BTree(t=2)
    counter: dict[str, int] = {}
    _instrument_btree(tree, counter)

    for k in INSERT_SEQ:
        tree.insert(k)
    assert counter.get("split_child", 0) >= 1, "insert should trigger at least one split"

    insert_splits = counter.get("split_child", 0)

    for k in DELETE_SEQ:
        tree.delete(k)

    required = {
        "borrow_right": 1,
        "borrow_left": 1,
        "merge": 1,
        "predecessor": 1,
    }
    missing = [name for name, min_count in required.items() if counter.get(name, 0) < min_count]
    if missing:
        raise AssertionError(f"B-tree missing operations: {missing}; counts={counter}")

    # Root should shrink at least once during deletes (merge chain)
    print("B-tree OK:", {**counter, "insert_splits": insert_splits})


def _validate_bplustree() -> None:
    bp = _load_module("b+tree.py", "b_plus_tree")
    tree = bp.BPlusTree(t=2)
    counter: dict[str, int] = {}
    _instrument_bplustree(tree, counter)

    for k in INSERT_SEQ:
        tree.insert(k)

    assert counter.get("split_leaf", 0) >= 1, "B+ insert should split at least one leaf"

    for k in DELETE_SEQ:
        tree.delete(k)

    borrow = (
        counter.get("borrow_left_leaf", 0)
        + counter.get("borrow_right_leaf", 0)
        + counter.get("borrow_left_internal", 0)
        + counter.get("borrow_right_internal", 0)
    )
    merge = counter.get("merge_leaf", 0) + counter.get("merge_internal", 0)

    if borrow < 1:
        raise AssertionError(f"B+ tree: no borrow; counts={counter}")
    if merge < 1:
        raise AssertionError(f"B+ tree: no merge; counts={counter}")

    print("B+ tree OK:", counter)


def main() -> None:
    _validate_btree()
    _validate_bplustree()
    print("All interview sequence checks passed.")


if __name__ == "__main__":
    main()
