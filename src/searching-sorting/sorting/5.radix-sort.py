"""
    COMPLEXITY: Time O(d * n) where d = max digits, k = 10 (base) | Space O(n + k) | Stable: Yes | In-place: No
    Non-comparative → Uses digit-by-digit distribution | For integers: O(d * n) time, O(n) space

    ALGORITHM: Sort by each digit position (LSD: right-to-left, MSD: left-to-right)
    Uses stable counting sort as subroutine for each digit position

    WHEN TO USE:
    ✅ Integers with bounded digits (phone numbers, IDs) | Strings with fixed length
    ✅ When O(n log n) comparison sorts too slow for large datasets
    ❌ Variable length strings | Negative numbers (without modification)

    KEY INTERVIEW POINTS:
    1. Why O(d * n)? → d passes × O(n) counting sort per pass
    2. LSD vs MSD? → LSD: right-to-left, stable, used here | MSD: left-to-right, can be unstable
    3. Why stable? → Uses stable counting sort as subroutine
    4. Common Q: "Sort 1 million phone numbers" → Use radix sort
    5. Negative numbers? → Split into positive/negative, sort separately, combine
"""


class RadixSort:
    def __init__(self, arr: list[int]):
        self.arr = arr if arr else []

    def sort(self):
        if self.arr:
            max_ele = max(self.arr)
            exp = 1  # Represents 1s, 10s, 100s place

            while max_ele // exp > 0:
                self._stable_count_sort(exp)
                exp *= 10

    def _stable_count_sort(self, exp: int):
        n = len(self.arr)
        output = [0] * n
        count = [0] * 10

        # 1. Store count of occurrences in count[]
        for i in range(n):
            index = (self.arr[i] // exp) % 10
            count[index] += 1

        # 2. Change count[i] so that it contains actual
        #    position of this digit in output[]
        for i in range(1, 10):
            count[i] += count[i - 1]

        # 3. Build the output array (Stable Sort: Iterate in reverse)
        for i in range(n - 1, -1, -1):
            index = (self.arr[i] // exp) % 10
            output[count[index] - 1] = self.arr[i]
            count[index] -= 1

        # 4. Copy the output array to arr, so that arr now
        #    contains sorted numbers according to current digit
        self.arr[:] = output
