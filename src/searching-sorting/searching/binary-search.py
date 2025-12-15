"""
    WHEN TO USE:
        - When you need to search for a target in a sorted array.
        - When you need to find the index of a target in a sorted array.
        - When you need to find the first occurrence of a target in a sorted array.
        - When you need to find the last occurrence of a target in a sorted array.
    KEY INTERVIEW POINTS:
    1. Why O(log n)? → Each iteration halves the search space
"""

class BinarySearch:

    def __init__(self, arr: list[int]):
        self._arr = arr

    def search(self, target: int) -> int:
        """
            Searches for a target in a sorted array using iterative approach.
            Args:
                target: The target value to search for.
            Returns:
                int: The index of the target value, -1 if not found.
            Time Complexity: O(log n)
            Space Complexity: O(1)
            Edge Cases:
                - Target not found (returns -1)
                - Empty array (returns -1)
                - Single element array (returns 0 if target is the only element)
                - Multiple occurrences of target (returns first occurrence)
                - Target at start/end of array
                - Target not in array
        """
        left, right = 0, len(self._arr) - 1
        while left <= right:
            mid = (left + right) >> 1
            if self._arr[mid] == target:
                return mid
            elif self._arr[mid] < target:
                left = mid + 1
            else:
                right = mid - 1
        return -1
    
    def search_recursive(self, target: int) -> int:
        """
            Searches for a target in a sorted array using recursive approach.
            Args:
                target: The target value to search for.
            Returns:
                int: The index of the target value, -1 if not found.
            Time Complexity: O(log n)
            Space Complexity: O(log n) for recursion
            Edge Cases:
                - Target not found (returns -1)
                - Empty array (returns -1)
                - Single element array (returns 0 if target is the only element)
                - Multiple occurrences of target (returns first occurrence)
                - Target at start/end of array
                - Target not in array
        """
        def helper(left: int, right: int) -> int:
            if left > right:
                return -1
            mid = (left + right) >> 1
            if self._arr[mid] == target:
                return mid
            elif self._arr[mid] < target:
                return helper(mid + 1, right)
            else:
                return helper(left, mid - 1)
        
        return helper(0, len(self._arr) - 1)

if __name__ == "__main__":
    arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    binary_search = BinarySearch(arr)
    print(binary_search.search(5))
    print(binary_search.search_recursive(5))
