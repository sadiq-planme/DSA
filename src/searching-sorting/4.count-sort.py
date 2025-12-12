"""
    Counting Sort - FANG Interview Essentials

    COMPLEXITY: Time O(n + k) where k = range (max - min + 1) | Space O(n + k) | Stable: Yes | In-place: No
    Non-comparison based → Breaks Ω(n log n) lower bound | Linear O(n) when k = O(n)

    ALGORITHM: Count frequencies → Prefix sum (cumulative counts) → Build output in REVERSE for stability
    Key: Iterate right-to-left when placing elements to maintain stability

    WHEN TO USE:
    ✅ Small integer range (k ≤ n) | Stability required | Used as subroutine in Radix Sort
    ❌ Large range (k >> n) | Negative numbers (shift by min first)

    KEY INTERVIEW POINTS:
    1. Why stable? → Reverse iteration preserves relative order of equal elements
    2. Why O(n + k)? → Count array O(k) + prefix sum O(k) + output build O(n)
    3. Negative numbers? → Shift all by min value, sort, shift back
    4. vs Bucket/Radix? → Counting sort is subroutine of radix sort
    5. Common Q: "Sort integers [0, k] in linear time" → Use counting sort
"""

class CountSort:
    def __init__(self, arr: list[int]):
        self.arr = arr 
        self.max_ele = max(arr) if arr else 0
        self.cum_sum_of_freq_of_num = [0] * (1 + self.max_ele) if arr else [0]

    def sort(self):
        n = len(self.arr)
        if self.arr:
            # count frequency of each element in the input sequence
            for num in self.arr: 
                self.cum_sum_of_freq_of_num[num] += 1
            
            # calculate cumulative sum of frequency of each element
            for num in range(1, 1 + self.max_ele):
                self.cum_sum_of_freq_of_num[num] += self.cum_sum_of_freq_of_num[num - 1]
            
            # build the output array STABLE
            output_arr = [0] * n
            for i in range(n - 1, -1, -1):
                num = self.arr[i]
                output_arr[self.cum_sum_of_freq_of_num[num] - 1] = num
                self.cum_sum_of_freq_of_num[num] -= 1

            # update the input array
            self.arr[:] = output_arr


if __name__ == "__main__":
    arr = [4, 2, 2, 8, 3, 3, 1]
    count_sort = CountSort(arr)
    count_sort.sort()
    print(arr)
