from __future__ import annotations
from dataclasses import dataclass


@dataclass
class Node:
    info: int | str
    next: Node | None = None


class CircularSinglyLinkedList:
    """
        PROS:
            1. No Memory wastage, unlike Arrays.
            2. Dynamically we can insert, delete Nodes.
        CONS:
            1. Non contiguous Memory Allocation. Index based access is not possible.
    """

    def __init__(self):
        self._size: int = 0
        self._head: Node = None
        self._tail: Node = None

    # -------------------------- Public methods --------------------------
    # tail ->> head   ===>   tail ->> new_node ->> head
    # after this function call, head will be the new node
    def insert_as_head(self, info: int | str):
        new_node = Node(info, self._head)
        if self.is_empty():
            self._initialize_single_node(new_node)
        else:
            self._head = new_node
            self._make_circular()
        self._size += 1

    # tail ->> head   ===>   previous_tail ->> new_node ->> head
    # after this function call, tail will be the new node
    def insert_as_tail(self, info: int | str):
        new_node = Node(info, self._head)
        if self.is_empty():
            self._initialize_single_node(new_node)
        else:
            self._tail.next = new_node
            self._tail = new_node
            self._make_circular()
        self._size += 1

    def insert_node_at_position(self, info: int | str, posi: int) -> bool:
        """
            Insert a node at the given position (1-indexed).
            Negative positions count from the end.
            Position 0 is treated as position 1 (head).
        """
        # there is no node at posi == 0. Input is position not index.
        if  posi == 0 or self._size == 0 or abs(posi) > self._size:
            return False
        
        if posi < 0:
            posi = self._size + posi + 1
        
        if posi == 1:
            self.insert_as_head(info)
            return True
        
        # prev_node ->> curr_node   ===>   prev_node ->> new_node ->> curr_node
        prev_node, curr_node = self._tail, self._head
        for _ in range(posi - 1):
            prev_node = curr_node
            curr_node = curr_node.next
        
        new_node = Node(info, curr_node)
        prev_node.next = new_node
        self._size += 1
        return True

    def find(self, info: int | str) -> tuple[Node | None, Node | None]:
        """
            Find a node with the given info value.
            Returns (previous_node, target_node) or (None, None) if not found.
        """
        if self.is_empty():
            return None, None
        
        prev_node, curr_node = self._tail, self._head
        for _ in range(self._size):
            if curr_node.info == info:
                return prev_node, curr_node
            prev_node = curr_node
            curr_node = curr_node.next
        
        # if the node is not found, return None, None
        return None, None

    # target_node ->> next_node   ===>   target_node ->> new_node ->> next_node
    # after this function call, target_node will point to the new node
    def insert_after_target(self, target: int | str, info: int | str):
        """
            Insert a new node after the first occurrence of target value.
        """
        if target == self._tail:
            self.insert_as_tail(info)
            return True
        
        _, target_node = self.find(target)
        if target_node is None:
            return False
        
        # NODE INSERTION LOGIC: inserting the new node after the target node
        # target_node ->> new_node ->> next_node
        next_node = target_node.next
        new_node = Node(info, next_node)
        target_node.next = new_node

        self._size += 1
        return True

    # tail ->> head ->> head.next   ===>   tail ->> head.next
    # after this function call, head will be the next node
    def remove_head(self):
        """
            Remove the first node from the list.
            Returns the removed node or None if list is empty.
        """
        if self.is_empty():
            return None
        
        if self._size == 1:
            curr_node = self._head
            self._head = self._tail = None
            
            self._size = 0
            curr_node.next = None
            return curr_node
        else:
            # NODE REMOVAL LOGIC: moving the head pointer to the next node
            # tail ->> head ->> head.next
            curr_node = self._head
            self._head = self._head.next
            self._make_circular()
            
            self._size -= 1
            curr_node.next = None
            return curr_node

    # previous_node_to_tail ->> tail ->> head   ===>   previous_node_to_tail ->> head
    # after this function call, new tail will be the previous node of the tail node
    def remove_tail(self):
        """
            Remove the last node from the list.
            Returns the removed node or None if list is empty.
        """
        if self.is_empty():
            return None
        
        if self._size == 1:
            # saving the tail node before setting it to None
            curr_node = self._tail
            self._head = self._tail = None
            curr_node.next = None
            self._size = 0
            return curr_node
        else:
            curr_node = self._head
            # at the end of this loop, curr_node will be on the previous node to the tail node
            while curr_node.next != self._tail:
                curr_node = curr_node.next
            
            # NODE REMOVAL LOGIC: removing the tail node from the list
            # saving the actual tail node before updating pointers
            tail_node = self._tail
            # curr_node ->> tail ->> head
            self._tail = curr_node
            self._make_circular()
            # returning the actual tail node that was removed
            tail_node.next = None
            self._size -= 1
            return tail_node

    # prev_node ->> curr_node ->> next_node   ===>   prev_node ->> next_node
    # after this function call, curr_node will be removed from the list
    def remove(self, info: int | str):
        """
            Remove the first occurrence of a node with the given info value.
            Returns the removed node or None if not found.
        """
        prev_node, curr_node = self.find(info)
        if curr_node is None:
            return None
        
        if curr_node == self._head:
            return self.remove_head()
        if curr_node == self._tail:
            return self.remove_tail()
        
        # NODE REMOVAL LOGIC: removing the curr_node node from the list
        # prev_node ->> curr_node ->> next_node
        next_node = curr_node.next
        prev_node.next = next_node

        # GARBAGE COLLECTION LOGIC: clearing pointer to prevent memory leaks
        curr_node.next = None
        self._size -= 1
        return curr_node

    # prev_node ->> curr_node ->> next_node   ===>   prev_node ->> next_node
    # after this function call, curr_node will be removed from the list
    def remove_node_at_position(self, posi: int):
        """
            Remove a node at the given position (1-indexed).
            Negative positions count from the end.
            Returns the removed node or None if position is invalid.
        """
        if self.is_empty() or posi == 0 or abs(posi) > self._size:
            return False
        
        if posi < 0:
            posi = self._size + posi + 1
        
        if posi == 1:
            self.remove_head()    
        elif posi == self._size:
            self.remove_tail()
        else:
            prev_node, curr_node = self._tail, self._head
            # at the end of this loop, curr_node will be on the node user wants to remove
            for _ in range(posi - 1):
                prev_node = curr_node
                curr_node = curr_node.next
            
            # NODE REMOVAL LOGIC: removing the curr_node node from the list
            # prev_node ->> curr_node ->> next_node
            next_node = curr_node.next
            prev_node.next = next_node

            # GARBAGE COLLECTION LOGIC: setting the next pointer of curr_node to None before returning it to the user
            # As long as the user holds that one removed node, the entire linked list cannot be garbage collected, creating a massive memory leak.
            curr_node.next = None
            self._size -= 1
            return True
        return False

    # current list: 1 ->> 2 ->> 3 ->> 4 ->> 5 (since circular, 5 ->> 1)   
    # new list    : 1 <<- 2 <<- 3 <<- 4 <<- 5 (since circular, 5 <<- 1) 
    # after this function call, the list will be reversed
    # FOR SINGLE ITERATION: prev ->> curr_node ->> next   ===>   prev <<- curr_node  next
    # OBSERVE THERE IS NO POINTER BETWEEN CURRENT NODE & NEXT NODE
    def reverse_in_place(self):
        if self._size <= 1:
            return
        if self._size > 2:
            # initializing prev with tail because of the circular nature of the list
            prev_node, curr_node, next_node = self._tail, self._head, self._head.next
            
            # at the end of this loop, curr_node will be on the head node of the new list
            for _ in range(self._size - 1):
                # REVERSAL LOGIC: curr_node will point to the previous node instead of the next node
                curr_node.next = prev_node
                # moving the pointers forward
                prev_node, curr_node, next_node = curr_node, next_node, next_node.next
            
        # updating the head and tail for self._size >= 2
        self._head, self._tail = self._tail, self._head
        self._make_circular()

    def is_empty(self):
        return self._size == 0

    # -------------------------- Properties --------------------------
    @property
    def size(self):
        return self._size

    # ------------------------ Protected methods ------------------------
    def _make_circular(self):
        if self._tail:
            self._tail.next = self._head

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
        return " ->> ".join(nodes)
