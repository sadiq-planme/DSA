"""
    COMPLEXITY: Time O(n log n) avg, O(n²) worst | Space O(log n) avg, O(n) worst | Stable: No | In-place: Yes

    ALGORITHM: Choose random pivot → Partition (≤ pivot left, > pivot right) → Recursively sort subarrays

    WHEN TO USE:
    ✅ General-purpose | In-place required | Large datasets
    ❌ Stability needed | Worst-case guarantee | Nearly sorted (without randomization)

    KEY INTERVIEW POINTS:
    1. Why O(n log n) avg? → Balanced partitions → log n levels
    2. Why O(n²) worst? → Unbalanced (pivot always min/max) → n levels
    3. Random pivot? → Prevents O(n²) on sorted arrays
    4. Why unstable? → Non-adjacent swaps disrupt equal element order
    5. Optimize? → Median-of-three, insertion for small subarrays, 3-way partition
"""

import random

class QuickSort:

    def sort(self, arr: list[int]):
        if len(arr) > 1:
            self._arr = arr
            self._quick_sort(0, len(self._arr) - 1)

    def _quick_sort(self, left: int, right: int):
        if left < right:
            pivot_pos = self._partition(left, right)
            self._quick_sort(left, pivot_pos - 1)
            self._quick_sort(pivot_pos + 1, right)

    def _partition(self, left: int, right: int):
        pivot_idx = random.randint(left, right)
        self._arr[pivot_idx], self._arr[right] = self._arr[right], self._arr[pivot_idx]
        
        left_boundary = left
        for current in range(left, right):
            if self._arr[current] <= self._arr[right]:
                self._arr[left_boundary], self._arr[current] = self._arr[current], self._arr[left_boundary]
                left_boundary += 1
        
        self._arr[left_boundary], self._arr[right] = self._arr[right], self._arr[left_boundary]
        return left_boundary
