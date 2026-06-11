# Interview Tree Demo Sequences

Short, memorizable **INSERT → DELETE** sequences for whiteboard + live coding (`t=2`, max 3 keys per B-tree node).  
Verified by `python src/trees/validate_interview_sequences.py`.

---

## Cheat sheet

| # | Structure | INSERT | DELETE |
|---|-----------|--------|--------|
| 1 | B-tree | `1–12` | `1,2,10,3,11,12,6,4,5,7,8,9` |
| 2 | B+ tree | `1–12` | same as B-tree |
| 3 | AVL | `1–7` (+ corner `3,2,1` / `3,1,2` / `1,3,2`) | `3,7,5,1,6,4,2` |
| 4 | Red-black | `1–8` | `6,1,4,3,8,7,2,5` |
| 5 | BST | `4,2,6,1,3,5,7` | `3,6,4,2,7,5,1` |
| 6 | Max-heap | push `1–7` | `pop` ×7 |

**Delete mnemonic (B-tree / B+):** “1-2, **10 before 3**, 11-12, middle six.”  
(Deleting `10` before `3` triggers **borrow from left**; `1,2,3` triggers **borrow from right**.)

---

## 1. B-tree (`t=2`)

### INSERT: `1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12`

| Step | Edge case |
|------|-----------|
| `1` | empty → single leaf root |
| `2–3` | root leaf not full |
| `4` | **root split** (height +1) |
| `5–12` | **child splits** on path; stays 2 levels |

### After INSERT (level order)

```
root:     [4]
level 1:  [2]  [6, 8, 10]
level 2:  [1] [3] [5] [7] [9] [11, 12]
```

### DELETE: `1, 2, 10, 3, 11, 12, 6, 4, 5, 7, 8, 9`

| Delete | Edge case | Code hook |
|--------|-----------|-----------|
| `1,2,3` | borrow from **right** sibling | `_borrow_from_right_sibling` |
| `10` (before `3`) | borrow from **left** sibling | `_borrow_from_left_sibling` |
| `11,12` | more borrow / merge | `_fix_deficient_child` |
| `6` | **internal** key → **predecessor** | `_get_max` + `_delete` |
| `4,5,7,8,9` | **merge**, **root shrink** | `_merge` |

---

## 2. B+ tree (`t=2`)

### INSERT / DELETE

Same numbers as B-tree. On the board, also draw:

- **Leaf split:** copy `right.keys[0]` into parent (not mid-key promotion like internal B-tree split).
- **Leaf chain:** `next` pointers after each leaf split.

### After INSERT

```
root: [5]
       /    \
    [3]      [7]
   /  \      /  \
 ... leaves ...
leaf chain: [1] → [2] → [3] → [4] → [5] → [6] → [7] → [8] → [9] → [10, 11, 12]
```

B+ delete is **leaf-only** (no predecessor/successor in internal nodes); internal borrow/merge only while descending.

---

## 3. AVL tree

### INSERT: `1, 2, 3, 4, 5, 6, 7`

| Keys | Case |
|------|------|
| `1,2,3` | **RR** |
| `4–7` | mixed rebalance on right spine |

### After INSERT

```
        2
      /   \
     1     4
          / \
         3   6
            / \
           5   7
```

### Rotation corner (fresh tree, ~10s)

| INSERT | Case |
|--------|------|
| `3, 2, 1` | **LL** |
| `3, 1, 2` | **LR** |
| `1, 3, 2` | **RL** |
| (`1,2,3` = **RR** in main tree) | |

### DELETE: `3, 7, 5, 1, 6, 4, 2`

| Delete | Case |
|--------|------|
| `3`, `7` | leaf + rebalance |
| `5` | two children → successor |
| `1,6,4,2` | L0 / L-1 / R1 via `_rebalance` |

---

## 4. Red-black tree

### INSERT: `1, 2, 3, 4, 5, 6, 7, 8`

| Keys | Fixup |
|------|-------|
| `4` | uncle **RED** → recolor |
| `5–8` | **LL / RR / LR / RL** rotations |

### After INSERT

```
        4B
      /    \
    2R      6R
   /  \    /  \
  1B  3B  5B  7B
              \
              8R
```

### DELETE: `6, 1, 4, 3, 8, 7, 2, 5`

| Delete | Case |
|--------|------|
| `6` | **RED** node → simple remove |
| `1`, `4` | **BLACK** + **RED** child → recolor child |
| `3,8,7,2,5` | double-black fixup (sibling RED / nephews) |

---

## 5. BST (unbalanced)

### INSERT: `4, 2, 6, 1, 3, 5, 7`

```
        4
      /   \
     2     6
    / \   / \
   1  3  5  7
```

### DELETE: `3, 6, 4, 2, 7, 5, 1`

| Delete | Case |
|--------|------|
| `3`, `6` | **leaf** |
| `4` | **two children** → successor `5` |
| `2` | **one child** (right) |
| `7,5,1` | cleanup |

Optional: INSERT `1,2,3,4,5` on a second sketch → degenerate spine (why AVL/RB exist).

---

## 6. Max-heap

### PUSH: `1, 2, 3, 4, 5, 6, 7`

```
array: [7, 4, 6, 1, 3, 2, 5]

       7
     /   \
    4     6
   / \   / \
  1  3  2  5
```

### POP ×7

Each `pop()`: swap root/last, `_heapify_down` (pick larger child; last pops handle single-node tree).

> **Note:** `src/trees/heap.py` `_heapify_up` skips swap when `parent_index == 0`; use the diagram above for interviews. Fix: use `parent_index >= 0` and loop while `parent_index >= 0`.

---

## Interview workflow

1. Write the **INSERT** line → build once; mark splits/rotations.
2. Circle 2–3 nodes your code will touch.
3. Write **DELETE** → update the same diagram.
4. Point each step to one function in your repo.

## Tooling

```bash
python src/trees/validate_interview_sequences.py   # B-tree / B+ edge-case counters
python src/trees/render_interview_trees.py         # print post-INSERT ASCII
pytest src/trees/tests/test_interview_sequences.py -q
```
