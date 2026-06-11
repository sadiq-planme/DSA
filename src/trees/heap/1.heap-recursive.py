class MaxHeap:

    def __init__(self):
        self._h: list[int] = []
        self._n: int = 0

    # ********* HELPER METHODS *********
    def _left(self, p: int) -> int:
        return (p << 1) + 1

    def _parent(self, c: int) -> int:
        return (c - 1) >> 1

    def _heapify(self, p: int) -> None:
        """TC: O(log n), SC: O(log n) (due to recursion)"""
        l = self._left(p)
        r = l + 1
        largest = p
        if l < self._n and self._h[l] > self._h[largest]:
            largest = l
        if r < self._n and self._h[r] > self._h[largest]:
            largest = r
        if largest != p:
            self._h[p], self._h[largest] = self._h[largest], self._h[p]
            self._heapify(largest)

    # ********* PRIMARY METHODS *********
    @property
    def peek(self) -> int | None:
        if self._n == 0:
            return None
        return self._h[0]

    @property
    def size(self) -> int:
        return self._n

    @property
    def empty(self) -> bool:
        return self._n == 0

    def push(self, k: int) -> None:
        """TC: O(log n), SC: O(1)"""
        self._h.append(k)
        self._n += 1
        if self._n > 1:
            c = self._n - 1
            p = self._parent(c)
            # MISTAKE: while p > 0 prevents swapping with the root when the new element is larger than the root
            while p >= 0 and self._h[c] > self._h[p]:
                self._h[c], self._h[p] = self._h[p], self._h[c]
                c = p
                p = self._parent(c)

    def pop(self) -> int | None:
        """TC: O(log n), SC: O(log n) (due to recursion in _heapify)"""
        if self._n == 0:
            return None
        self._h[0], self._h[-1] = self._h[-1], self._h[0]
        self._n -= 1
        self._heapify(0)
        return self._h.pop()

    def build(self, data: list[int]):
        """TC: O(n), SC: O(n + log n) (due to copying the list and recursion in _heapify)"""
        self._h = data.copy()
        self._n = len(data)
        for i in range((self._n - 2) >> 1, -1, -1):
            self._heapify(i)

    def sort(self) -> list[int]:
        """
        This is a destructive operation that sorts in place and returns heap reference itself not a copy
        TC: O(n log n), SC: O(log n) (due to recursion in _heapify)
        """
        while self._n > 0:
            last_index = self._n - 1  # MISTAKE: self._h[-1] != self._h[self._n - 1]
            self._h[0], self._h[last_index] = self._h[last_index], self._h[0]
            # size decrease in necessary before heapifying
            self._n -= 1
            self._heapify(0)
        return self._h
