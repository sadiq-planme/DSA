
from dataclasses import dataclass

RED = "RED"
BLACK = "BLACK"

@dataclass
class RBNode:
    key: int
    color: str = RED
    left: RBNode | None = None
    right: RBNode | None = None
    parent: RBNode | None = None


class RedBlackTree:
    def __init__(self) -> None:
        # NIL sentinel stands in for every "null" leaf — always BLACK,
        # so color checks on NIL nodes never need a special None-guard.
        self.NIL = RBNode(key=0, color=BLACK)
        self.NIL.left = self.NIL.right = self.NIL.parent = self.NIL
        self.root: RBNode = self.NIL

    # ── Insert ─────────────────────────────────────────────────────────────────
    def insert(self, key: int) -> None:
        # Rule 2: every new node enters as RED — we fix violations afterward
        new_node = RBNode(
            key=key, 
            color=RED,
            left=self.NIL, 
            right=self.NIL, 
            parent=self.NIL
        )

        parent = self.NIL
        curr = self.root
        # standard BST traversal
        while curr is not self.NIL:              
            parent = curr
            curr = curr.left if key < curr.key else curr.right

        new_node.parent = parent
        # Rule 1: tree was empty → becomes root
        if parent is self.NIL:
            self.root = new_node
            # Rule 1: root is always BLACK
            self.root.color = BLACK
            return
        # insert as left child
        elif key < parent.key:
            parent.left = new_node
        # insert as right child
        else:
            parent.right = new_node

        self._insert_fixup(new_node)

    def _insert_fixup(self, new_node: RBNode) -> None:
        # Rule 2a: parent is BLACK → no RED-RED conflict, nothing to fix
        # Rule 2b: parent is RED  → RED-RED conflict, enter the loop
        while new_node.parent.color == RED:
            parent = new_node.parent
            grandparent = parent.parent

            if parent is grandparent.left:       # ── LEFT subtree cases ──
                uncle = grandparent.right

                if uncle.color == RED:
                    # Rule 2b-ii-c  (Uncle RED)
                    # Recolor parent & uncle BLACK, grandparent RED, move up to recheck
                    parent.color = BLACK
                    uncle.color = BLACK
                    grandparent.color = RED
                    new_node = grandparent       # grandparent might now conflict with ITS parent

                else:                            # Uncle is BLACK or NIL
                    if new_node is parent.right:
                        # Rule 2b-ii-b  (LR imbalance)
                        # Step 1: left-rotate parent → converts LR into LL
                        new_node = parent
                        self._left_rotate(new_node)
                        parent = new_node.parent
                        grandparent = parent.parent

                    # Rule 2b-ii-a  (LL imbalance — also the result of LR step 1 above)
                    # Single right-rotation on grandparent + recolor
                    parent.color = BLACK
                    grandparent.color = RED
                    self._right_rotate(grandparent)

            else:                                # ── RIGHT subtree cases (mirror) ──
                uncle = grandparent.left

                if uncle.color == RED:
                    # Rule 2b-ii-c  (Uncle RED) — same logic, mirrored
                    parent.color = BLACK
                    uncle.color = BLACK
                    grandparent.color = RED
                    new_node = grandparent

                else:                            # Uncle is BLACK or NIL
                    if new_node is parent.left:
                        # Rule 2b-ii-b  (RL imbalance)
                        # Step 1: right-rotate parent → converts RL into RR
                        new_node = parent
                        self._right_rotate(new_node)
                        parent = new_node.parent
                        grandparent = parent.parent

                    # Rule 2b-ii-a  (RR imbalance — also the result of RL step 1 above)
                    # Single left-rotation on grandparent + recolor
                    parent.color = BLACK
                    grandparent.color = RED
                    self._left_rotate(grandparent)

        # Rule 1: root is always BLACK
        self.root.color = BLACK
        self.root.parent = self.NIL

    def _left_rotate(self, p: RBNode) -> None:
        pp = p.parent # 10
        pr = p.right # 30
        prl = pr.left # 25

        if pp is self.NIL:
            self.root = pr
        elif p is pp.left:
            pp.left = pr
        else:
            pp.right = pr
        pr.parent = pp
        
        pr.left = p
        p.parent = pr

        p.right = prl
        if prl is not self.NIL:
            prl.parent = p

    def _right_rotate(self, p: RBNode) -> None:
        pp = p.parent
        pl = p.left
        plr = pl.right 

        if pp is self.NIL:
            self.root = pl
        elif p is pp.left:
            pp.left = pl
        else:
            pp.right = pl
        pl.parent = pp

        pl.right = p
        p.parent = pl

        p.left = plr
        if plr is not self.NIL:
            plr.parent = p 

    # ── Delete ─────────────────────────────────────────────────────────────────
    def delete(self, key: int) -> bool:
        target = self._search_node(key)
        if target is self.NIL:
            return False

        # Internal node: swap with inorder successor (minimum of right subtree).
        # The successor has at most one child, so we always end up deleting a
        # node with at most one child — that's the case our fixup logic handles.
        if target.left is not self.NIL and target.right is not self.NIL:
            successor = self._minimum(target.right)
            # Color swap is not necessary here.
            target.key, successor.key = successor.key, target.key
            target = successor                   # now target has at most one child

        # The single surviving child (or NIL) that will replace target
        replacement = target.left if target.left is not self.NIL else target.right

        # Deletion Rule 1: target is RED → simple removal, no black-height change
        # Here replacement is always a self.NIL node.
        if target.color == RED:
            self._transplant(target, replacement)
            return True

        # Target is BLACK with a RED child → replace and recolor child BLACK.
        # This absorbs the "extra black" that target carried, restoring balance.
        # No branch for a non-NIL BLACK replacement: a BLACK node with exactly one real child cannot have that child be BLACK (black-height would beat the opposite NIL side). So here replacement is RED or NIL — never a lone BLACK child.
        if replacement.color == RED:
            self._transplant(target, replacement)
            replacement.color = BLACK  # RED child moves up; paint it BLACK so this path does not lose a black node (simplest delete fixup).
            return True

        # Target is BLACK with no RED child (replacement is NIL or BLACK).
        # Removing target creates a double-black deficit on `replacement`.
        self._transplant(target, replacement)
        self._delete_fixup(replacement)          # resolve double-black
        return True

    def _delete_fixup(self, double_black: RBNode) -> None:
        # Push the double-black up the tree until it hits the root or a RED node.
        while double_black is not self.root and double_black.color == BLACK:

            if double_black is double_black.parent.left:   # ── double-black is LEFT child ──
                sibling = double_black.parent.right

                # Rule 4: Sibling is RED
                # Swap parent ↔ sibling colors, rotate parent toward double-black (left).
                # After rotation double-black has a new BLACK sibling → fall into cases 3/5/6.
                if sibling.color == RED:
                    # sibling.color, double_black.parent.color = double_black.parent.color, sibling.color
                    sibling.color = BLACK
                    double_black.parent.color = RED
                    self._left_rotate(double_black.parent)
                    sibling = double_black.parent.right    # new sibling is now BLACK

                # Rule 3: Sibling BLACK, both sibling-children BLACK
                # Remove double-black, make sibling RED, push the extra black to parent.
                if sibling.color == BLACK and sibling.left.color == BLACK and sibling.right.color == BLACK:
                    sibling.color = RED
                    parent = double_black.parent
                    if parent.color == RED:
                        # Extra black is absorbed by recoloring RED parent → BLACK; done.
                        parent.color = BLACK
                        double_black = self.root # marks loop exit: problem solved. double black removed.
                    else:
                        double_black = parent  # BLACK parent inherits the double-black

                # Rule 5: Sibling BLACK, far-child BLACK, near-child RED
                # Swap sibling ↔ near-child colors, rotate sibling away from double-black (left).
                # This converts Rule 5 → Rule 6.
                case5Flag = False
                if sibling.color == BLACK and sibling.left.color == RED and sibling.right.color == BLACK:       # near child is red and far child (right) is BLACK
                    sibling.left.color = BLACK         # near child (left) was RED
                    sibling.color = RED
                    self._right_rotate(sibling)
                    sibling = double_black.parent.right
                    case5Flag = True

                # If Rule 5 is ran then Rule 6 has to be run.
                # Rule 6: Sibling BLACK, far-child RED
                # Swap parent ↔ sibling colors, rotate parent toward double-black (left),
                # recolor far child BLACK → double-black is resolved.
                # sibling.color, double_black.parent.color = double_black.parent.color, sibling.color
                if case5Flag or (sibling.color == BLACK and sibling.right.color == RED): # Rule 6
                    sibling.color = double_black.parent.color
                    double_black.parent.color = BLACK
                    self._left_rotate(double_black.parent)
                    double_black = self.root               # marks loop exit: problem solved
                    sibling.right.color = BLACK            # far (right) child → BLACK

            else:                                          # ── double-black is RIGHT child (mirror) ──
                sibling = double_black.parent.left

                # Rule 4 (mirror): Sibling RED → rotate parent rightward
                if sibling.color == RED:
                    sibling.color = BLACK
                    double_black.parent.color = RED
                    self._right_rotate(double_black.parent)
                    sibling = double_black.parent.left

                # Rule 3 (mirror): Both sibling-children BLACK → push black up
                if sibling.left.color == BLACK and sibling.right.color == BLACK:
                    sibling.color = RED
                    parent = double_black.parent
                    if parent.color == RED:
                        # Extra black is absorbed by recoloring RED parent → BLACK; done.
                        parent.color = BLACK
                        double_black = self.root
                    else:
                        double_black = parent

                else:
                    # Rule 5 (mirror): far-child BLACK, near-child RED → convert to Rule 6
                    if sibling.left.color == BLACK:        # far child (left) is BLACK
                        sibling.right.color = BLACK        # near child (right) was RED
                        sibling.color = RED
                        self._left_rotate(sibling)
                        sibling = double_black.parent.left

                    # Rule 6 (mirror): Sibling BLACK, far-child RED → resolve
                    sibling.color = double_black.parent.color
                    double_black.parent.color = BLACK
                    sibling.left.color = BLACK             # far (left) child → BLACK
                    self._right_rotate(double_black.parent)
                    double_black = self.root

        # Rule 2: double-black reached the root → simply strip the extra black
        double_black.color = BLACK

    def _search_node(self, key: int) -> RBNode:
        curr = self.root
        while curr is not self.NIL and curr.key != key:
            curr = curr.left if key < curr.key else curr.right
        return curr

    def _minimum(self, node: RBNode) -> RBNode:
        while node.left is not self.NIL:
            node = node.left
        return node

    def _transplant(self, removed: RBNode, replacement: RBNode) -> None:
        p = removed.parent
        if p is self.NIL:
            self.root = replacement
        elif removed is p.left:
            p.left = replacement
        else:
            p.right = replacement
        replacement.parent = p
