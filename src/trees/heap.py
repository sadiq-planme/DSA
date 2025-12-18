from __future__ import annotations
from dataclasses import dataclass
from collections import defaultdict, deque
import heapq
import math


class MaxHeap:

    def __init__(self):
        self._heap: list[int] = []
        self._size: int = 0

    # ********* Primary Methods *********
    def push(self, data: int):
        self._heap.append(data)
        self._size += 1
        self._heapify_up(self._size - 1)

    def pop(self) -> int | None:
        if self._size == 0:
            return None
        self._heap[0], self._heap[-1] = self._heap[-1], self._heap[0]
        self._size -= 1
        self._heapify_down(0)
        return self._heap.pop()

    def top(self) -> int | None:
        if self._size == 0:
            return None
        return self._heap[0]

    def build(self, data: list[int]) -> None:
        if not data:
            return
        self._heap = data
        self._size = len(data)
        for internal_node_index in range((self._size - 2) >> 1, -1, -1):  # range((self._size - 2) // 2, -1, -1)
            self._heapify_down(internal_node_index)

    # ********* Helper Methods *********
    def _heapify_up(self, child_index: int):
        """Bubble up the element at child_index to maintain max heap property"""
        parent_index = (child_index - 1) >> 1  # (child_index - 1) // 2
        if (parent_index > 0) and (self._heap[child_index] > self._heap[parent_index]):
            self._heap[child_index], self._heap[parent_index] = self._heap[parent_index], self._heap[child_index]
            self._heapify_up(parent_index)

    def _heapify_down(self, parent_index: int):
        """Bubble down the element at parent_index to maintain max heap property"""
        left_child_index = (parent_index << 1) + 1
        right_child_index = (parent_index << 1) + 2
        largest = parent_index
        
        if (left_child_index < self._size) and (self._heap[left_child_index] > self._heap[largest]):
            largest = left_child_index
        if (right_child_index < self._size) and (self._heap[right_child_index] > self._heap[largest]):
            largest = right_child_index
        
        if largest != parent_index:
            self._heap[parent_index], self._heap[largest] = self._heap[largest], self._heap[parent_index]
            self._heapify_down(largest)

    # ********* Secondary Methods *********
    def sort(self):
        """
            Sorts the heap in ascending order (heap sort).
            Returns:
                list[int]: The sorted heap.
            Time Complexity: O(n log n)
            Space Complexity: O(1)
        """
        while self._size > 1:
            self._heap[0], self._heap[self._size - 1] = self._heap[self._size - 1], self._heap[0]  # MISTAKE: self._heap[self._size - 1]  != self._heap[-1]
            self._size -= 1
            self._heapify_down(0)
        return self._heap
    
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
    def k_th_largest(self, k: int, arr: list[int]):
        """
        Finds the kth largest element in the array using min heap.
        Args:
            k: The kth position (1-indexed)
            arr: Input array
        Returns:
            int: The kth largest element, or None if invalid input
        Time Complexity: O(n log k)
        Space Complexity: O(k)
        """
        if not arr or k <= 0 or k > len(arr):
            return None
        self.heap = []
        self.build(arr[:k])
        for i in range(k, len(arr)):
            if arr[i] > self.top():
                self.pop()
                self.push(arr[i])
        return self.top()

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


# RARELY ASKED 1 QUESTION SOLVED HERE: MaxHeap with Binary Tree
@dataclass
class HeapTreeNode:
    val: int
    parent: HeapTreeNode | None = None
    left: HeapTreeNode | None = None
    right: HeapTreeNode | None = None


class MaxHeapWithBinaryTree:

    def __init__(self):
        self.root: HeapTreeNode | None = None

    # ********* Primary Methods *********
    def push(self, val: int):  # O(n + log n)
        """Add an element to the max heap using binary tree structure"""
        if self.root is None:
            self.root = HeapTreeNode(val)
        else:
            # Find the first available position (level-order insertion)
            queue = deque([self.root])
            while queue:
                current_node = queue.popleft()
                # Try to insert left
                if current_node.left is None:
                    current_node.left = HeapTreeNode(val, parent=current_node)
                    self._heapify_up(current_node.left)
                    break
                # Try to insert right
                if current_node.right is None:
                    current_node.right = HeapTreeNode(val, parent=current_node)
                    self._heapify_up(current_node.right)
                    break
                # Continue to next level
                queue.append(current_node.left)
                queue.append(current_node.right)

    def _heapify_up(self, node: HeapTreeNode):  # O(log n)
        """Bubble up the node to maintain max heap property"""
        while node.parent is not None and node.val > node.parent.val:
            node.val, node.parent.val = node.parent.val, node.val
            node = node.parent

    def pop(self):  # O(n + log n)
        """Remove and return the maximum element from the heap"""
        if self.root is None:
            return None
        
        max_val = self.root.val
        
        # Find the last node in the tree (Level Order Traversal)
        queue = deque([self.root])
        last_node = None
        while queue:
            last_node = queue.popleft()
            if last_node.left is not None:
                queue.append(last_node.left)
            if last_node.right is not None:
                queue.append(last_node.right)
        
        # Edge Case: Only one node (root only)
        if last_node == self.root:
            self.root = None
            return max_val
        
        # Move last node's value to root
        self.root.val = last_node.val

        # Remove the last node
        if last_node.parent.left == last_node:
            last_node.parent.left = None
        else:
            last_node.parent.right = None
        
        # Restore heap property downwards
        self._heapify_down(self.root)
        return max_val

    def _heapify_down(self, node: HeapTreeNode | None):  # O(log n)
        """Bubble down the node to maintain max heap property"""
        while node and node.left is not None:
            largest = node
            # Compare with left child
            if node.left.val > largest.val:
                largest = node.left
            # Compare with right child
            if node.right is not None and node.right.val > largest.val:
                largest = node.right
            # Swap if child is larger
            if largest != node:
                node.val, largest.val = largest.val, node.val
                node = largest
            else:
                break

    def build(self, vals: list[int]):  # O(n^2)
        """Build a max heap from a list of values"""
        for val in vals:
            self.push(val)

    # ********* Problems *********
    def is_heap(self):  # O(n)
        """
            Checks if the binary tree maintains max heap property.
            Returns:
                bool: True if valid max heap, False otherwise
            Time Complexity: O(n)
            Space Complexity: O(log n)
        """
        def helper(node: HeapTreeNode | None):
            if node is None:
                return True
            # Check left child
            if node.left is not None:
                if node.left.val > node.val or not helper(node.left):
                    return False
            # Check right child
            if node.right is not None:
                if node.right.val > node.val or not helper(node.right):
                    return False
            return True
        
        return helper(self.root)
