"""
    COMPLEXITY: Time O(n) best [sorted], O(n²) avg/worst | Space O(1) | Stable: Yes | Adaptive: Yes | Online: Yes

    ALGORITHM: Build sorted array incrementally. Insert each element into correct position in sorted portion.
    Intuition: Like sorting playing cards in hand.

    WHEN TO USE:
    ✅ Small arrays (<50) | Nearly sorted | Memory-constrained | Online sorting | Stability required
    ❌ Large arrays (>1000) | Random data | Need guaranteed O(n log n)

    KEY INTERVIEW POINTS:
    1. Why stable? → Uses > (not >=), equal elements don't swap
    2. Why adaptive? → O(n) for nearly sorted data
    3. Why online? → Can sort as elements arrive (streaming)
    4. vs Selection: More swaps but fewer comparisons, better for nearly sorted
    5. vs Bubble: Faster, adaptive, fewer swaps
    6. Real-world: Timsort uses insertion for small subarrays (<64 elements)
    7. Optimize? → Binary search reduces comparisons to O(n log n) but shifts still O(n²)
"""


class InsertionSort:
    def __init__(self, arr: list[int]):
        self._arr = arr
    
    def sort(self, start_idx: int = 0):
        if not self._arr or len(self._arr) <= 1:
            return
        
        for i in range(start_idx + 1, len(self._arr)):  #  unsorted arr head id
            key = self._arr[i]  #  element to be sorted
            j = i - 1  #  sorted arr tail id. using it to traverse the sorted array in reverse order.
            
            while j >= 0 and self._arr[j] > key:  #  using >= instead of > makes sorting unstable
                self._arr[j + 1] = self._arr[j]  #  shift the element to the right
                j -= 1
            
            self._arr[j + 1] = key  #  insert the element at the correct position
    
    def online_sort(self, element: int) -> None:
        """Adds a new element and maintains sorted order incrementally."""
        self._arr.append(element)
        self.sort(start_idx=len(self._arr) - 2)
