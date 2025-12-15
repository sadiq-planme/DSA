"""
    Merge Sort - FANG Interview Essentials

    COMPLEXITY: Time O(n log n) all cases | Space O(n) | Stable: Yes | In-place: No

    ALGORITHM: Divide → Split array in half | Conquer → Recursively sort halves | Combine → Merge sorted halves
    Key: Sentinel values (float('inf')) eliminate boundary checks in merge

    WHEN TO USE:
    ✅ Stability required | External sorting | Guaranteed O(n log n) | Linked lists
    ❌ Memory-constrained | Small arrays

    KEY INTERVIEW POINTS:
    1. Why O(n log n)? → log n levels × O(n) merge per level
    2. Why stable? → Merge uses ≤ (not <), preserves equal element order
    3. Space? → O(n) temp arrays + O(log n) stack = O(n)
    4. Optimize? → Insertion sort for small subarrays, bottom-up iterative
"""


class MergeSort:

    def __init__(self, arr: list[int]):
        self._arr = arr

    def sort(self):
        """Sorts array in-place. Handles empty/single element edge cases."""
        if not self._arr or len(self._arr) <= 1:
            return
        self._merge_sort(0, len(self._arr) - 1)

    def _merge_sort(self, left: int, right: int):
        """Divide: split until base case. Conquer: merge sorted halves."""
        if left < right:
            mid = (left + right) // 2
            self._merge_sort(left, mid)
            self._merge_sort(mid + 1, right)
            self._merge(left, mid, right)

    def _merge(self, left: int, mid: int, right: int):
        """Merges [left...mid] and [mid+1...right]. Sentinel values avoid boundary checks."""
        left_arr = self._arr[left:mid + 1] + [float('inf')]
        right_arr = self._arr[mid + 1:right + 1] + [float('inf')]
        
        i = j = 0
        for k in range(left, right + 1):
            if left_arr[i] <= right_arr[j]:
                self._arr[k] = left_arr[i]
                i += 1
            else:
                self._arr[k] = right_arr[j]
                j += 1
