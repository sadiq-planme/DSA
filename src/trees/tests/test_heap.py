import unittest
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from heap import MaxHeap, MinHeap, MaxHeapWithBinaryTree, LinkedListNode


class TestMaxHeap(unittest.TestCase):

    def test_push_and_top(self):
        """Test basic push and top operations"""
        heap = MaxHeap()
        heap.push(5)
        self.assertEqual(heap.top(), 5)
        heap.push(10)
        self.assertEqual(heap.top(), 10)
        heap.push(3)
        self.assertEqual(heap.top(), 10)
        heap.push(15)
        self.assertEqual(heap.top(), 15)

    def test_pop(self):
        """Test pop operation"""
        heap = MaxHeap()
        heap.push(10)
        heap.push(5)
        heap.push(15)
        heap.push(3)
        
        self.assertEqual(heap.pop(), 15)
        self.assertEqual(heap.pop(), 10)
        self.assertEqual(heap.pop(), 5)
        self.assertEqual(heap.pop(), 3)
        self.assertIsNone(heap.pop())  # Should return None when empty
        self.assertIsNone(heap.pop())  # Multiple pops on empty heap

    def test_empty(self):
        """Test empty check"""
        heap = MaxHeap()
        self.assertTrue(heap.empty())
        heap.push(5)
        self.assertFalse(heap.empty())
        heap.pop()
        self.assertTrue(heap.empty())

    def test_build(self):
        """Test building heap from array"""
        heap = MaxHeap()
        heap.build([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(heap.top(), 9)
        
        # Verify heap property by popping all
        popped = []
        while not heap.empty():
            popped.append(heap.pop())
        self.assertEqual(popped, [9, 6, 5, 4, 3, 2, 1, 1])

    def test_sort(self):
        """Test heap sort"""
        heap = MaxHeap()
        heap.build([3, 1, 4, 1, 5, 9, 2, 6])
        sorted_arr = heap.sort()
        self.assertEqual(sorted_arr, [1, 1, 2, 3, 4, 5, 6, 9])

    def test_k_th_smallest(self):
        """Test finding kth smallest element"""
        heap = MaxHeap()
        arr = [7, 10, 4, 3, 20, 15]
        
        self.assertEqual(heap.k_th_smallest(3, arr), 7)
        
        heap2 = MaxHeap()
        self.assertEqual(heap2.k_th_smallest(1, [1]), 1)
        
        heap3 = MaxHeap()
        self.assertEqual(heap3.k_th_smallest(2, [1, 2, 3]), 2)

    def test_min_stone_sum(self):
        """Test minimum stone sum after k operations"""
        heap = MaxHeap()
        # After removing max and adding floor(max/2) k times
        result = heap.min_stone_sum([5, 4, 9], 2)
        # This should use max heap, not min heap
        # Let's test the logic
        self.assertIsInstance(result, int)

    def test_reorganize_string(self):
        """Test string reorganization"""
        heap = MaxHeap()
        result = heap.reorganize_string("aab")
        self.assertEqual(result, "aba")  # Valid reorganization
        
        result2 = heap.reorganize_string("aaab")
        self.assertEqual(result2, "")  # Cannot reorganize
        
        result3 = heap.reorganize_string("aabb")
        self.assertNotEqual(result3, "")  # Should be reorganizable
        self.assertEqual(len(result3), 4)  # Should use all characters

    def test_longest_happy_string(self):
        """Test longest happy string"""
        heap = MaxHeap()
        result = heap.longest_happy_string(1, 1, 7)
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)

    def test_median_of_stream(self):
        """Test median of stream"""
        heap = MaxHeap()
        medians = heap.median_of_stream([5, 15, 1, 3])
        self.assertEqual(len(medians), 4)
        self.assertEqual(medians[0], 5.0)
        self.assertEqual(medians[1], 10.0)  # (5+15)/2
        self.assertEqual(medians[2], 5.0)
        self.assertEqual(medians[3], 4.0)  # (3+5)/2

    def test_len(self):
        """Test length method"""
        heap = MaxHeap()
        self.assertEqual(len(heap), 0)
        heap.push(5)
        self.assertEqual(len(heap), 1)
        heap.push(10)
        self.assertEqual(len(heap), 2)
        heap.pop()
        self.assertEqual(len(heap), 1)

    def test_edge_cases(self):
        """Test edge cases"""
        heap = MaxHeap()
        # Empty heap operations
        self.assertIsNone(heap.pop())
        self.assertIsNone(heap.top())
        self.assertTrue(heap.empty())
        
        # Single element
        heap.push(42)
        self.assertEqual(heap.top(), 42)
        self.assertEqual(heap.pop(), 42)
        self.assertTrue(heap.empty())
        
        # Duplicate elements
        heap.push(5)
        heap.push(5)
        heap.push(5)
        self.assertEqual(heap.pop(), 5)
        self.assertEqual(heap.pop(), 5)
        self.assertEqual(heap.pop(), 5)
        
        # k_th_smallest edge cases
        self.assertIsNone(heap.k_th_smallest(0, [1, 2, 3]))
        self.assertIsNone(heap.k_th_smallest(4, [1, 2, 3]))
        self.assertIsNone(heap.k_th_smallest(1, []))
        self.assertEqual(heap.k_th_smallest(1, [5]), 5)
        
        # Empty string reorganization
        self.assertEqual(heap.reorganize_string(""), "")
        
        # Single character
        self.assertEqual(heap.reorganize_string("a"), "a")
        
        # Median of empty stream
        self.assertEqual(heap.median_of_stream([]), [])


class TestMinHeap(unittest.TestCase):

    def test_push_and_top(self):
        """Test basic push and top operations"""
        heap = MinHeap()
        heap.push(10)
        self.assertEqual(heap.top(), 10)
        heap.push(5)
        self.assertEqual(heap.top(), 5)
        heap.push(15)
        self.assertEqual(heap.top(), 5)
        heap.push(3)
        self.assertEqual(heap.top(), 3)

    def test_pop(self):
        """Test pop operation"""
        heap = MinHeap()
        heap.push(10)
        heap.push(5)
        heap.push(15)
        heap.push(3)
        
        self.assertEqual(heap.pop(), 3)
        self.assertEqual(heap.pop(), 5)
        self.assertEqual(heap.pop(), 10)
        self.assertEqual(heap.pop(), 15)
        self.assertIsNone(heap.pop())  # Should return None when empty

    def test_build(self):
        """Test building heap from array"""
        heap = MinHeap()
        heap.build([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(heap.top(), 1)
        
        # Verify heap property by popping all
        popped = []
        while not heap.empty():
            popped.append(heap.pop())
        self.assertEqual(popped, [1, 1, 2, 3, 4, 5, 6, 9])

    def test_k_th_largest(self):
        """Test finding kth largest element"""
        heap = MinHeap()
        arr = [7, 10, 4, 3, 20, 15]
        self.assertEqual(heap.k_th_largest(3, arr), 10)
        
        heap2 = MinHeap()
        self.assertEqual(heap2.k_th_largest(1, [1]), 1)

    def test_merge_k_sorted_arrays(self):
        """Test merging k sorted arrays"""
        heap = MinHeap()
        arrays = [[1, 3, 5], [2, 4, 6], [0, 7, 8]]
        result = heap.merge_k_sorted_arrays(arrays)
        self.assertEqual(result, [0, 1, 2, 3, 4, 5, 6, 7, 8])
        
        # Edge case: single array
        result2 = heap.merge_k_sorted_arrays([[1, 2, 3]])
        self.assertEqual(result2, [1, 2, 3])
        
        # Edge case: empty arrays
        result3 = heap.merge_k_sorted_arrays([[], [1], [2, 3]])
        self.assertEqual(result3, [1, 2, 3])

    def test_merge_k_sorted_sll(self):
        """Test merging k sorted linked lists"""
        heap = MinHeap()
        
        # Create linked lists: [1,3,5], [2,4,6], [0,7]
        list1 = LinkedListNode(1)
        list1.next = LinkedListNode(3)
        list1.next.next = LinkedListNode(5)
        
        list2 = LinkedListNode(2)
        list2.next = LinkedListNode(4)
        list2.next.next = LinkedListNode(6)
        
        list3 = LinkedListNode(0)
        list3.next = LinkedListNode(7)
        
        head, tail = heap.merge_k_sorted_sll([list1, list2, list3])
        
        # Verify merged list
        result = []
        current = head
        while current:
            result.append(current.val)
            current = current.next
        self.assertEqual(result, [0, 1, 2, 3, 4, 5, 6, 7])

    def test_smallest_range(self):
        """Test finding smallest range"""
        heap = MinHeap()
        arrays = [[4, 10, 15, 24, 26], [0, 9, 12, 20], [5, 18, 22, 30]]
        start, end = heap.smallest_range(arrays)
        self.assertIsInstance(start, int)
        self.assertIsInstance(end, int)
        self.assertLessEqual(start, end)

    def test_edge_cases_min_heap(self):
        """Test edge cases for MinHeap"""
        heap = MinHeap()
        # Empty heap operations
        self.assertIsNone(heap.pop())
        self.assertIsNone(heap.top())
        self.assertTrue(heap.empty())
        
        # Single element
        heap.push(42)
        self.assertEqual(heap.top(), 42)
        self.assertEqual(heap.pop(), 42)
        
        # k_th_largest edge cases
        self.assertIsNone(heap.k_th_largest(0, [1, 2, 3]))
        self.assertIsNone(heap.k_th_largest(4, [1, 2, 3]))
        self.assertIsNone(heap.k_th_largest(1, []))
        
        # Merge empty arrays
        self.assertEqual(heap.merge_k_sorted_arrays([]), [])
        self.assertEqual(heap.merge_k_sorted_arrays([[], []]), [])
        
        # Merge single array
        self.assertEqual(heap.merge_k_sorted_arrays([[1, 2, 3]]), [1, 2, 3])
        
        # Smallest range edge cases
        start, end = heap.smallest_range([[1], [2], [3]])
        self.assertIsNotNone(start)
        self.assertIsNotNone(end)


class TestMaxHeapWithBinaryTree(unittest.TestCase):

    def test_push_and_top(self):
        """Test basic push and top operations"""
        heap = MaxHeapWithBinaryTree()
        heap.push(5)
        self.assertEqual(heap.top(), 5)
        heap.push(10)
        self.assertEqual(heap.top(), 10)
        heap.push(3)
        self.assertEqual(heap.top(), 10)
        heap.push(15)
        self.assertEqual(heap.top(), 15)

    def test_pop(self):
        """Test pop operation"""
        heap = MaxHeapWithBinaryTree()
        heap.push(10)
        heap.push(5)
        heap.push(15)
        heap.push(3)
        
        self.assertEqual(heap.pop(), 15)
        self.assertEqual(heap.pop(), 10)
        self.assertEqual(heap.pop(), 5)
        self.assertEqual(heap.pop(), 3)
        self.assertIsNone(heap.pop())  # Should return None when empty

    def test_build(self):
        """Test building heap from array"""
        heap = MaxHeapWithBinaryTree()
        heap.build([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertEqual(heap.top(), 9)

    def test_is_heap(self):
        """Test heap validation"""
        heap = MaxHeapWithBinaryTree()
        heap.build([3, 1, 4, 1, 5, 9, 2, 6])
        self.assertTrue(heap.is_heap())
        
        # Empty heap should be valid
        heap2 = MaxHeapWithBinaryTree()
        self.assertTrue(heap2.is_heap())
        
        # Single element should be valid
        heap3 = MaxHeapWithBinaryTree()
        heap3.push(5)
        self.assertTrue(heap3.is_heap())


if __name__ == '__main__':
    unittest.main()
