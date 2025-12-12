"""
    Bubble Sort - FANG Interview Quick Reference

    COMPLEXITY: Time O(n²) worst/avg, O(n) best (early termination) | Space O(1) | Stable: Yes | Adaptive: Yes

    ALGORITHM: Compare adjacent elements, swap if out of order. Largest "bubbles up" each pass.
    Early termination: Stop if no swaps (array sorted).

    WHEN TO USE:
    ✅ Small datasets (<10) | Nearly sorted | Memory-constrained
    ❌ Large datasets | Production code

    KEY INTERVIEW POINTS:
    1. Why stable? → Only swaps adjacent elements, equal elements never cross
    2. Why O(n²)? → Each element may bubble through n positions
    3. Common mistake: Forgetting "-1" in inner loop → index out of bounds
    4. vs Selection: Both O(n²), but bubble stable, selection not
    5. vs Insertion: Both stable, insertion has better constants
"""


def bubble_sort(arr: list[int]):
    n = len(arr)
    if n <= 1:
        return
    
    # we have to do n-1 passes to sort the array of size n. In each pass 1 element is sorted.
    for num_of_sorted_ele in range(n):
        # If no swapping is done in a single pass, the array is already sorted
        swapped = False
        # IMPORTANCE OF "-1" IN BELOW FOR LOOP
        # If you remove "-1" => when num_of_sorted_ele == 0 => iter_to_comp can take values = [0, n-1] => but when iter_to_comp == n-1 => iter_to_comp + 1 will be out of index
        for iter_to_comp in range(n - 1 - num_of_sorted_ele):
            # sorts data in ascending order
            # to sort data in descending order, just change the > to <
            if arr[iter_to_comp] > arr[iter_to_comp + 1]:
                arr[iter_to_comp], arr[iter_to_comp + 1] = arr[iter_to_comp + 1], arr[iter_to_comp]
                swapped = True
        # EARLY STOPPING CRITERIA
        if swapped == False:
            return



if __name__ == "__main__":
    arr = [64, 34, 25, 12, 22, 11, 90]
    bubble_sort(arr)
    print(arr)
