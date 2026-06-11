class Heap:

    # def __init__(self, heap: list[int] = []):  MISTAKE: Objects would only share the same list reference
    def __init__(self, max_priority: bool = True):
        self._heap: list[int] = []
        self._size: int = 0
        self._max_priority: bool = max_priority

    # ********* HELPER METHODS *********
    def _left(self, parent_index: int) -> int:
        return (parent_index << 1) + 1  # (parent_index * 2) + 1

    def _parent(self, child_index: int) -> int:
        return (child_index - 1) >> 1  # (child_index - 1) // 2

    def _bubble_up_criteria(self, child_index: int, parent_index: int) -> bool:
        if parent_index < 0:
            return False
        if self._max_priority:
            return self._heap[child_index] > self._heap[parent_index]
        return self._heap[child_index] < self._heap[parent_index]

    def _heapify(self, parent_index: int) -> None:
        """TC: O(log n), SC: O(log n) (due to recursion)"""
        left_child_index = self._left(parent_index)
        right_child_index = left_child_index + 1
        largest_index = parent_index
        if self._heapify_criteria(left_child_index, largest_index):
            largest_index = left_child_index
        if self._heapify_criteria(right_child_index, largest_index):
            largest_index = right_child_index
        if largest_index != parent_index:
            self._heap[parent_index], self._heap[largest_index] = self._heap[largest_index], self._heap[parent_index]
            self._heapify(largest_index)

    def _heapify_criteria(self, child_index: int, largest_index: int) -> bool:
        if child_index < 0 or child_index >= self._size:
            return False
        if self._max_priority:
            return self._heap[child_index] > self._heap[largest_index]
        return self._heap[child_index] < self._heap[largest_index]

    # ********* PRIMARY METHODS *********
    @property
    def peek(self) -> int | None: 
        if self._size == 0:
            return None
        return self._heap[0]

    @property
    def size(self) -> int:
        return self._size

    @property
    def empty(self) -> bool:
        return self._size == 0

    def push(self, k: int) -> None:
        """TC: O(log n), SC: O(1)"""
        self._heap.append(k)
        self._size += 1
        if self._size > 1:
            child_index = self._size - 1
            parent_index = self._parent(child_index)
            while self._bubble_up_criteria(child_index, parent_index):
                self._heap[child_index], self._heap[parent_index] = self._heap[parent_index], self._heap[child_index]
                child_index = parent_index
                parent_index = self._parent(child_index)

    def pop(self) -> int | None:
        """TC: O(log n), SC: O(log n) (due to recursion in _heapify)"""
        if self._size == 0:
            return None
        self._heap[0], self._heap[-1] = self._heap[-1], self._heap[0]
        self._size -= 1
        self._heapify(0)
        return self._heap.pop()

    def build(self, data: list[int]):
        """TC: O(n), SC: O(n + log n) (due to copying the list and recursion in _heapify)"""
        self._heap = data.copy()
        self._size = len(data)
        # range((self._size - 2) // 2, -1, -1)
        for internal_node_index in range((self._size - 2) >> 1, -1, -1):
            self._heapify(internal_node_index)

    def sort(self) -> list[int]:
        """
        This is a destructive operation that sorts in place and returns heap reference itself not a copy
        TC: O(n log n), SC: O(log n) (due to recursion in _heapify)
        """
        while self._size > 0:
            last_index = self._size - 1  # MISTAKE: self._heap[-1] != self._heap[self._size - 1]
            self._heap[0], self._heap[last_index] = self._heap[last_index], self._heap[0]
            # size decrease in necessary before heapifying
            self._size -= 1
            self._heapify(0)
        return self._heap
