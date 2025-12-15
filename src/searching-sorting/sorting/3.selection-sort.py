"""
    COMPLEXITY: Time O(n²) all cases | Space O(1) | Stable: No | Adaptive: No
    Comparisons: O(n²) | Swaps: O(n) worst, O(1) best

    ALGORITHM: Find minimum in unsorted portion, swap with first unsorted element.
    n-1 passes, each scans n-i elements → n(n-1)/2 comparisons total.

    WHEN TO USE:
    ✅ Small arrays (<20) | Expensive writes (flash memory) | Need minimal swaps
    ❌ Large data | Need stability | Nearly sorted data

    KEY INTERVIEW POINTS:
    1. Why O(n²)? → n-1 passes × (n-i) comparisons = n(n-1)/2
    2. Why not stable? → Swapping can move equal elements out of order
    3. Advantage → Minimal swaps (max n-1 swaps)
    4. vs Insertion → Selection: O(n) swaps | Insertion: O(n²) swaps but adaptive
    5. Swaps: Min 0 (sorted) | Max n-1 (reverse sorted)
"""


def selection_sort(array):
    n = len(array)
    for i in range(n-1):  #  number of passes = n - 1
        min_RSA_ele_index, un_sorted_RSA_traverse_index = i, i+1
        while un_sorted_RSA_traverse_index < n:
            if array[un_sorted_RSA_traverse_index] < array[min_RSA_ele_index]:
                min_RSA_ele_index = un_sorted_RSA_traverse_index
            un_sorted_RSA_traverse_index += 1
        # ASCENDING: swap the minimum element with the first element of the unsorted array
        array[i], array[min_RSA_ele_index] = array[min_RSA_ele_index], array[i]

if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]
    selection_sort(arr)
    print(arr)
