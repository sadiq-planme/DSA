from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq
import math


class MaxHeap:
    
    def __init__(self):
        self._h: list[int] = []
        self._n: int = 0
    
    def _parent(self, c: int) -> int:
        return (c - 1) >> 1

    def push(self, val: int) -> None:
        """TC: O(log n) SC: O(1)"""
        self._h.append(val)
        self._n += 1
        if self._n > 1:
            c = self._n - 1
            # Bubble up the newly inserted element to make it part of the heap
            self._bubble_up(c, p)

    def _bubble_up(self, c: int):
        p = self._parent(c)
        while p >= 0 and self._h[c] > self._h[p]:
            self._h[c], self._h[p] = self._h[p], self._h[c]
            c = p
            p = self._parent(c)

    def peak(self) -> int | None:
        if self._n == 0:
            return None
        return self._h[0]

    def pop(self) -> int | None:
        """TC: O(log n) SC: O(1)"""
        if self._n == 0:
            return None
        if self._n == 1:
            self._n -= 1
            return self._h.pop()
        self._h[0], self._h[-1] = self._h[-1], self._h[0]
        self._n -= 1
        self._max_heapify(0)
        return self._h.pop()

    def _left(self, p: int) -> int:
        return (p << 1) + 1

    def _max_heapify(self, p: int) -> None:
        while True:
            l = self._left(p)
            r = l + 1
            largest = p
            if l < self._n and self._h[l] > self._[largest]:
                largest = l
            if r < self._n and self._h[r] > self._h[largest]:
                largest = r
            if largest != p:
                self._h[largest], self._h[p] = self._h[p], self._h[largest]
                p = largest
            else:
                break
    
    def build(self, data: list[int]) -> None:
        """TC: O(n) SC(n)"""
        if not data:
            return
        self._h = data.copy()
        self._n = len(data)
        last_int = (self._n - 2) >> 1
        for i in range(last_int, -1, -1):
            self._max_heapify(i)

    def sort(self) -> list[int] | None:
        """TC: O(n log n) SC: O(1)"""
        while self._n > 1:
            l = self._n - 1
            self._h[0], self._h[l] = self._h[l], self._h[0]
            # size decrease in necessary before heapifying
            self._n -= 1
            self._max_heapify(0)
        return self._h

    def initialize_heap(self) -> None:
        self._n = 0
        self._h.clear()
    
    def change_key(self, i: int, v: int) -> None:
        if self._h[i] < v:
            self._decrease_key(i, v)
        else:
            self._increase_key(i, v)
    
    def _decrease_key(self, i: int, v: int) -> None:
        self._h[i] = v
        self._max_heapify(i)

    def _increase_key(self, i: int, v: int) -> None:
        self._h[i] = v
        self._bubble_up(i)

    def delete_key(self, i: int) -> bool:
        # If the heap is Min Heap, we should be using the "_decrease_key" method with
        # key "-float('inf')"
        self._increase_key(i, float('inf'))
        self.pop()
    
    @property
    def size(self) -> int:
        return self._n

    @property
    def empty(self) -> bool:
        return self._n == 0

# --------- PROBLEMS ---------
def is_max_heap(data: list[int]) -> bool:
    n = len(data)
    last_int_id = (n - 2) >> 1
    
    for p in range(last_int_id, -1, -1):
        lc = (p << 1) + 1
        rc = lc + 1
        
        # If left child is greater than the parent
        if lc < n and data[lc] > data[p]:
            return False
        
        # If right child is greater than the parent
        if rc < n and data[rc] > data[p]:
            return False
    
    return True

def convert_into_min_heap(data: list[int]) -> list[int]:
    n = len(data)
    last_int_id = (n - 2) >> 1

    for p in range(last_int_id, -1, -1):
        while True:
            lc = (p << 1) + 1
            rc = lc + 1

            smallest = p

            if lc < n and data[lc] < data[smallest]:
                smallest = lc
            if rc < n and data[rc] < data[smallest]:
                smallest = rc

            if smallest != p:
                data[smallest], data[p] = data[p], data[smallest]
                p = smallest
            else:
                break
    
    return data

def k_th_largest(data: list[int], k: int) -> int:
    if not data: 
        return -1
    
    import heapq
    pq = []

    for i in range(k):
        heapq.heappush(pq, data[i])

    for i in range(k, len(data)):
        if data[i] > pq[0]:
            heapq.heappop(pq)
            heapq.heappush(pq, data[i])
    
    return pq[0]

# K th Largest Number in a Running stream of numbers
import heapq
class Soution:

    def __init__(self, data: list[int], k: int):
        self.k: int = k
        self.pq: list[int] = []

        for num in data:
            if len(self.pq) < self.k:
                heapq.heappush(self.pq, num)
            elif num > self.pq[0]:
                heapq.heappop(self.pq)
                heapq.heappush(self.pq, num)

    def k_th_largest(self, key: int):
        if len(self.pq) < self.k:
            heapq.heappush(self.pq, key)
            return self.pq[0]
        
        if key > self.pq[0]:
            heapq.heappop(self.pq)
            heapq.heappush(self.pq, key)
        
        return self.pq[0]


class MaxHeap:
    # ********* Problems *********
    def k_th_smallest(self, k: int, arr: list[int]):
        """
            Finds the kth smallest element in the array using max heap.
            Args:
                k: The kth position (1-indexed)
                arr: Input array
            Returns:
                int: The kth smallest element, or None if invalid input
            Time Complexity: O(n log k)
            Space Complexity: O(k)
        """
        if not arr or k <= 0 or k > len(arr):
            return None
        self.heap = []
        self.build(arr[:k])
        for i in range(k, len(arr)):
            if arr[i] < self.top():
                self.pop()
                self.push(arr[i])
        return self.top()

    def min_stone_sum(self, stones: list[int], k: int):
        """
            Finds the minimum sum of stones after k operations.
            Each operation removes the largest stone and adds floor(largest/2) back.
            Args:
                stones: List of stone weights
                k: Number of operations
            Returns:
                int: The minimum sum of stones after k operations
            Time Complexity: O(n + k log n)
            Space Complexity: O(n)
        """
        # Use negative values to simulate max heap with min heap
        heap = [-stone for stone in stones]
        heapq.heapify(heap)
        while k > 0 and heap:
            max_val = -heapq.heappop(heap)
            new_val = max_val // 2
            if new_val > 0:
                heapq.heappush(heap, -new_val)
            k -= 1
        return sum(-val for val in heap)

    # https://leetcode.com/problems/reorganize-string/description/?ref=read.learnyard.com
    # Minimum cost to cut the ropes which uses Heaps not DP, also uses this approach
    def reorganize_string(self, s: str):
        """
            Reorganizes the string so no two same characters are adjacent.
            Args:
                s: Input string
            Returns:
                str: Reorganized string, or empty string if impossible
            Time Complexity: O(n log k) where k is unique characters
            Space Complexity: O(k)
        """
        if not s:
            return ""
        
        count: defaultdict[str, int] = defaultdict(int)
        for char in s:
            count[char] += 1
        
        max_heap: list[tuple[int, str]] = [(-count[char], char) for char in count]
        heapq.heapify(max_heap)
        
        ans: list[str] = []
        while len(max_heap) > 1:
            count1, char1 = heapq.heappop(max_heap)
            count2, char2 = heapq.heappop(max_heap)
            count1, count2 = -count1, -count2
            
            ans.append(char1)
            ans.append(char2)
            
            if count1 > 1:
                heapq.heappush(max_heap, (-(count1 - 1), char1))
            if count2 > 1:
                heapq.heappush(max_heap, (-(count2 - 1), char2))
        
        if len(max_heap) == 1:
            count1, char1 = heapq.heappop(max_heap)
            count1 = -count1
            if count1 > 1:
                return ''
            ans.append(char1)
        
        return ''.join(ans)

    # https://leetcode.com/problems/longest-happy-string/description/
    def longest_happy_string(self, a: int, b: int, c: int):
        """
            Finds the longest happy string (no 3 consecutive same characters).
            Args:
                a: Count of 'a'
                b: Count of 'b'
                c: Count of 'c'
            Returns:
                str: The longest happy string
            Time Complexity: O(n log k) where n is total characters
            Space Complexity: O(k)
        """
        max_heap: list[tuple[int, str]] = [
            (-count, char) for count, char in [(a, 'a'), (b, 'b'), (c, 'c')] if count > 0
        ]
        heapq.heapify(max_heap)

        ans: list[str] = []
        while len(max_heap) > 1:
            count1, char1 = heapq.heappop(max_heap)
            count2, char2 = heapq.heappop(max_heap)
            count1, count2 = -count1, -count2
            
            # Use up to 2 of the most frequent character
            use_count1 = min(2, count1)
            ans.append(char1 * use_count1)
            count1 -= use_count1
            if count1 > 0:
                heapq.heappush(max_heap, (-count1, char1))
            
            # Use up to 2 of the second most frequent character
            # Use 2 only if count2 >= count1 to balance
            use_count2 = min(2, count2) if count2 >= count1 else min(1, count2)
            ans.append(char2 * use_count2)
            count2 -= use_count2
            if count2 > 0:
                heapq.heappush(max_heap, (-count2, char2))
        
        # Handle remaining character
        if len(max_heap) == 1:
            count1, char1 = heapq.heappop(max_heap)
            count1 = -count1
            # Can only add if last char is different and count <= 2
            if ans and ans[-1] == char1:
                return ''.join(ans)  # Cannot add more
            use_count = min(2, count1)
            ans.append(char1 * use_count)
        
        return ''.join(ans)

    def median_of_stream(self, nums: list[int]):
        """
            Finds the median of a stream of numbers using two heaps.
            Uses max_heap for smaller half and min_heap for larger half.
            Args:
                nums: Stream of numbers
            Returns:
                list[float]: Median after each number is added
            Time Complexity: O(n log n)
            Space Complexity: O(n)
        """
        min_heap: list[int] = []
        max_heap: list[int] = []
        medians: list[float] = [float('-inf')]
        
        for num in nums:
            if len(max_heap) == len(min_heap):
                if num > medians[-1]:
                    heapq.heappush(min_heap, num)
                    medians.append(min_heap[0])
                else:
                    heapq.heappush(max_heap, -num)
                    medians.append(-max_heap[0])
            elif len(max_heap) > len(min_heap):
                if num > medians[-1]:
                    heapq.heappush(min_heap, num)
                    medians.append((min_heap[0] + -max_heap[0]) / 2)
                else:
                    heapq.heappush(min_heap, -heapq.heappop(max_heap))
                    heapq.heappush(max_heap, -num)
                    medians.append((min_heap[0] + -max_heap[0]) / 2)
            elif len(max_heap) < len(min_heap):
                if num > medians[-1]:
                    heapq.heappush(max_heap, -heapq.heappop(min_heap))
                    heapq.heappush(min_heap, num)
                    medians.append((min_heap[0] + -max_heap[0]) / 2)
                else:
                    heapq.heappush(max_heap, -num)
                    medians.append((min_heap[0] + -max_heap[0]) / 2)
        
        return medians[1:]

    # TODO: Merge 2 Heaps
    # TODO: Is it CBT?


@dataclass
class LinkedListNode:
    val: int
    next: LinkedListNode | None = None


class MinHeap:
    # ********* Problems *********
    def merge_k_sorted_arrays(self, arrays: list[list[int]]):
        """
            Merges k sorted arrays into a single sorted array.
            Args:
                arrays: List of k sorted arrays
            Returns:
                list[int]: The merged sorted array
            Time Complexity: O(n log k) where n is total elements
            Space Complexity: O(k)
        """
        if not arrays:
            return []
        
        # Initialize heap with first element from each non-empty array
        heap: list[tuple[int, int, int]] = []
        for i, arr in enumerate(arrays):
            if arr:  # Only add non-empty arrays
                heap.append((arr[0], i, 0))
        heapq.heapify(heap)

        # Merge the arrays
        result: list[int] = []
        while heap:
            val, i, j = heapq.heappop(heap)
            result.append(val)
            if j + 1 < len(arrays[i]):
                heapq.heappush(heap, (arrays[i][j + 1], i, j + 1))
        
        return result

    def merge_k_sorted_sll(self, heads: list[LinkedListNode]):
        """
            Merges k sorted singly linked lists into a single sorted linked list.
            Args:
                heads: List of head nodes of k sorted linked lists
            Returns:
                tuple: (head, tail) of the merged sorted linked list
            Time Complexity: O(n log k) where n is total nodes
            Space Complexity: O(k)
        """
        if not heads:
            return None, None
        
        heap: list[tuple[int, LinkedListNode]] = []
        for head in heads:
            if head:
                heap.append((head.val, head))
        heapq.heapify(heap)

        dummy = LinkedListNode(0)
        tail = dummy
        while heap:
            _, curr_node = heapq.heappop(heap)
            tail.next = curr_node
            tail = curr_node
            if curr_node.next:
                heapq.heappush(heap, (curr_node.next.val, curr_node.next))
        
        return dummy.next, tail

    def smallest_range(self, arrays: list[list[int]]):
        """
            Finds the smallest range that includes at least one element from each array.
            Args:
                arrays: List of k sorted arrays
            Returns:
                tuple[int, int]: The smallest range [start, end]
            Time Complexity: O(n log k) where n is total elements
            Space Complexity: O(k)
        """
        if not arrays or any(not arr for arr in arrays):
            return None, None
        
        k = len(arrays)
        heap: list[tuple[int, int, int]] = []  # (value, array_index, element_index)
        current_max = float('-inf')
        
        # Initialize heap with first element from each array
        for i in range(k):
            heap.append((arrays[i][0], i, 0))
            current_max = max(current_max, arrays[i][0])
        heapq.heapify(heap)

        min_range = math.inf
        min_range_start = min_range_end = None
        
        while True:
            val, i, j = heapq.heappop(heap)
            if current_max - val < min_range:
                min_range = current_max - val
                min_range_start = val
                min_range_end = current_max
            
            # Move to next element in the same array
            if j + 1 < len(arrays[i]):
                next_val = arrays[i][j + 1]
                heapq.heappush(heap, (next_val, i, j + 1))
                current_max = max(current_max, next_val)
            else:
                break  # One array exhausted
        
        return min_range_start, min_range_end
