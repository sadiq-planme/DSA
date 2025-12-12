from dataclasses import dataclass


@dataclass
class Node:
    info: int | str
    prev: "Node" | None = None
    next: "Node" | None = None


class CircularDoublyLinkedList:
    """
        PROS:
            1. No Memory wastage, unlike Arrays.
            2. Dynamically we can insert, delete Nodes.
            3. Bidirectional traversal (forward and backward).
        CONS:
            1. Non contiguous Memory Allocation. Index based access is not possible.
            2. Extra memory overhead for previous pointer.
    """

    def __init__(self):
        self._size: int = 0
        self._head: Node | None = None
        self._tail: Node | None = None

    # -------------------------- Public methods --------------------------
    # tail <->> head   ===>   tail <->> new_node <->> head
    # after this function call, head will be the new node
    def insert_as_head(self, info: int | str):
        new_node = Node(info, self._tail, self._head)
        if self.is_empty():
            self._initialize_single_node(new_node)
        else:
            self._head.prev = new_node
            self._head = new_node
            self._make_circular()
        self._size += 1
        return True

    # tail <->> head   ===>   previous_tail <->> new_node <->> head
    # after this function call, tail will be the new node
    def insert_as_tail(self, info: int | str):
        new_node = Node(info, self._tail, self._head)
        if self.is_empty():
            self._initialize_single_node(new_node)
        else:
            self._tail.next = new_node
            self._head.prev = new_node
            self._tail = new_node
            self._make_circular()
        self._size += 1
        return True

    def insert_node_at_position(self, info: int | str, posi: int):
        """
            Insert a node at the given position (1-indexed).
            Negative positions count from the end.
        """
        # if posi = -1   ===>   tail <->> head   will become   new_node <->> tail <->> head
        if posi < 1:
            posi = self._size + posi + 1
        
        # tail <->> head   ===>   tail <->> new_node <->> head
        # after this, head will be the new node
        if posi <= 1 or self.is_empty():
            return self.insert_as_head(info)
        # tail <->> head   ===>   tail <->> new_node <->> head
        # after this, tail will be the new node
        if posi > self._size:
            return self.insert_as_tail(info)
        
        # prev_node <->> curr_node   ===>   prev_node <->> new_node <->> curr_node
        curr_node = self._head
        for _ in range(posi - 1):
            curr_node = curr_node.next
        prev_node = curr_node.prev

        new_node = Node(info, prev_node, curr_node)
        prev_node.next = new_node
        curr_node.prev = new_node
        self._size += 1
        return True

    def find(self, info: int | str):
        """
            Find a node with the given info value.
            Returns the target node or None if not found.
        """
        if self.is_empty():
            return None
        
        curr_node = self._head
        for _ in range(self._size):
            if curr_node.info == info:
                return curr_node
            curr_node = curr_node.next
        
        # if the node is not found, return None
        return None

    # target_node <->> next_node   ===>   target_node <->> new_node <->> next_node
    # after this function call, new_node will be inserted after target_node
    def insert_after_target(self, target: int | str, info: int | str):
        """
            Insert a new node after the first occurrence of target value.
        """
        target_node = self.find(target)
        if target_node is None:
            return False
        
        if target_node == self._tail:
            return self.insert_as_tail(info)
        
        # NODE INSERTION LOGIC: inserting the new node after the target node
        # target_node <->> new_node <->> next_node
        next_node = target_node.next
        new_node = Node(info, target_node, next_node)
        target_node.next = new_node
        next_node.prev = new_node
        
        self._size += 1
        return True

    # tail <->> head <->> head.next   ===>   tail <->> head.next
    # after this function call, head will be the next node
    def remove_head(self):
        """
            Remove the first node from the list.
            Returns the removed node or None if list is empty.
        """
        if self.is_empty():
            return None
        
        curr_node = self._head
        if self._size == 1:
            self._head = self._tail = None
        else:
            # NODE REMOVAL LOGIC: moving the head pointer to the next node
            # tail <->> head <->> head.next
            self._head = self._head.next
            self._make_circular()
        
        curr_node.next = curr_node.prev = None
        self._size -= 1
        return curr_node

    # previous_node_to_tail <->> tail <->> head   ===>   previous_node_to_tail <->> head
    # after this function call, tail will be the previous node of the tail node
    def remove_tail(self):
        """
            Remove the last node from the list.
            Returns the removed node or None if list is empty.
        """
        if self.is_empty():
            return None
        
        curr_node = self._tail
        if self._size == 1:
            self._head = self._tail = None
        else:
            # NODE REMOVAL LOGIC: moving the tail pointer to the previous node
            # previous_node_to_tail <->> tail <->> head   ===>   previous_node_to_tail <->> head
            self._tail = self._tail.prev
            self._make_circular()
        
        curr_node.next = curr_node.prev = None
        self._size -= 1
        return curr_node

    # prev_node <->> curr_node <->> next_node   ===>   prev_node <->> next_node
    # after this function call, curr_node will be removed from the list
    def remove(self, info: int | str):
        """
            Remove the first occurrence of a node with the given info value.
            Returns the removed node or None if not found.
        """
        curr_node = self.find(info)
        if curr_node is None:
            return None
        
        if curr_node == self._head:
            return self.remove_head()
        if curr_node == self._tail:
            return self.remove_tail()
        
        # NODE REMOVAL LOGIC: removing the curr_node node from the list
        # prev_node <->> curr_node <->> next_node
        next_node = curr_node.next
        prev_node = curr_node.prev
        prev_node.next = next_node
        next_node.prev = prev_node
        
        # GARBAGE COLLECTION LOGIC: clearing pointers to prevent memory leaks
        curr_node.next = curr_node.prev = None
        self._size -= 1
        return curr_node

    # prev_node <->> curr_node <->> next_node   ===>   prev_node <->> next_node
    # after this function call, curr_node will be removed from the list
    def remove_node_at_position(self, posi: int):
        """
            Remove a node at the given position (1-indexed).
            Negative positions count from the end.
            Returns the removed node or None if position is invalid.
        """
        if self.is_empty():
            return None
        
        if posi < 1:
            posi = self._size + posi + 1
        
        if posi == 1:
            return self.remove_head()
        elif posi == self._size:
            return self.remove_tail()
        elif posi > self._size or posi < 1:
            return None
        else:
            curr_node = self._head
            # at the end of this loop, curr_node will be on the node user wants to remove
            for _ in range(posi - 1):
                curr_node = curr_node.next
            
            # NODE REMOVAL LOGIC: finding and removing the node at given position
            # prev_node <->> curr_node <->> next_node
            prev_node = curr_node.prev
            next_node = curr_node.next
            prev_node.next = next_node
            next_node.prev = prev_node
            
            # GARBAGE COLLECTION LOGIC: setting the next pointer of curr_node to None before returning it to the user
            # As long as the user holds that one removed node, the entire linked list cannot be garbage collected, creating a massive memory leak.
            curr_node.next = curr_node.prev = None
            self._size -= 1
            return curr_node

    # current list: 1 <->> 2 <->> 3 <->> 4 <->> 5 (circular: 5 <->> 1)
    # new list    : 1 <<-> 2 <<-> 3 <<-> 4 <<-> 5 (circular: 5 <<-> 1)
    # after this function call, the list will be reversed
    # FOR SINGLE ITERATION: 
    # prev_node <->> curr_node <->> next_node   ===>   prev_node <<->> curr_node <-> next_node
    # OBSERVE THERE IS NO POINTER BETWEEN CURRENT NODE & NEXT NODE
    def reverse_in_place(self):
        if self._size > 1:
            # initializing prev with tail because of the circular nature of the list
            prev_node, curr_node, next_node = self._tail, self._head, self._head.next
            
            # at the end of this loop, curr_node will be on the head node of the new list
            for _ in range(self._size):
                # REVERSAL LOGIC: swap next and prev pointers for curr_node node
                curr_node.next, curr_node.prev = prev_node, next_node
                # moving the pointers forward
                prev_node, curr_node, next_node = curr_node, next_node, next_node.next
            
            # Update head and tail references
            self._head, self._tail = self._tail, self._head

    def is_empty(self):
        return self._size == 0

    # -------------------------- Properties --------------------------
    @property
    def size(self):
        return self._size

    # ------------------------ Protected methods -------------------------
    def _make_circular(self):
        if self._tail and self._head:
            self._tail.next = self._head
            self._head.prev = self._tail

    def _initialize_single_node(self, node: Node):
        self._head = self._tail = node
        self._make_circular()

    # -------------------------- Dunder methods --------------------------
    def __str__(self):
        if self.is_empty():
            return "[]"
        
        nodes = []
        curr_node = self._head
        for _ in range(self._size):
            nodes.append(str(curr_node.info))
            curr_node = curr_node.next
        return " <->> ".join(nodes)
