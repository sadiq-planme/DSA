"""
Comprehensive unit tests for CircularSinglyLinkedList implementation.
Tests all methods with edge cases and sample data.
"""

import unittest
import sys
import os
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sll import CircularSinglyLinkedList, Node


class TestCircularSinglyLinkedList(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.sll = CircularSinglyLinkedList()

    # ==================== Test insert_as_head ====================
    def test_insert_as_head_empty_list(self):
        """Test inserting head into empty list."""
        result = self.sll.insert_as_head(10)
        self.assertTrue(result)
        self.assertEqual(self.sll.size, 1)
        self.assertEqual(self.sll._head.info, 10)
        self.assertEqual(self.sll._tail.info, 10)
        self.assertEqual(self.sll._head.next, self.sll._head)

    def test_insert_as_head_multiple_elements(self):
        """Test inserting multiple heads."""
        self.sll.insert_as_head(3)
        self.sll.insert_as_head(2)
        self.sll.insert_as_head(1)
        self.assertEqual(self.sll.size, 3)
        self.assertEqual(self.sll._head.info, 1)
        self.assertEqual(self.sll._tail.info, 3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3")

    def test_insert_as_head_string_values(self):
        """Test inserting string values as head."""
        self.sll.insert_as_head("hello")
        self.sll.insert_as_head("world")
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(self.sll._head.info, "world")
        self.assertEqual(str(self.sll), "world ->> hello")

    # ==================== Test insert_as_tail ====================
    def test_insert_as_tail_empty_list(self):
        """Test inserting tail into empty list."""
        result = self.sll.insert_as_tail(10)
        self.assertTrue(result)
        self.assertEqual(self.sll.size, 1)
        self.assertEqual(self.sll._head.info, 10)
        self.assertEqual(self.sll._tail.info, 10)

    def test_insert_as_tail_multiple_elements(self):
        """Test inserting multiple tails."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        self.assertEqual(self.sll.size, 3)
        self.assertEqual(self.sll._head.info, 1)
        self.assertEqual(self.sll._tail.info, 3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3")

    # ==================== Test insert_node_at_position ====================
    def test_insert_at_position_empty_list(self):
        """Test inserting at position in empty list."""
        result = self.sll.insert_node_at_position(10, 1)
        self.assertTrue(result)
        self.assertEqual(self.sll.size, 1)
        self.assertEqual(self.sll._head.info, 10)

    def test_insert_at_position_beginning(self):
        """Test inserting at position 1."""
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        self.sll.insert_node_at_position(1, 1)
        self.assertEqual(self.sll.size, 3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3")

    def test_insert_at_position_middle(self):
        """Test inserting at middle position."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(3)
        self.sll.insert_node_at_position(2, 2)
        self.assertEqual(self.sll.size, 3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3")

    def test_insert_at_position_end(self):
        """Test inserting at end position."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_node_at_position(3, 3)
        self.assertEqual(self.sll.size, 3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3")

    def test_insert_at_position_negative(self):
        """Test inserting at negative position."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_node_at_position(3, -1)  # Should insert at position 2 (size + -1 + 1)
        self.assertEqual(self.sll.size, 3)
        self.assertEqual(str(self.sll), "1 ->> 3 ->> 2")

    def test_insert_at_position_zero(self):
        """Test inserting at position 0 (should insert at head)."""
        self.sll.insert_as_tail(2)
        self.sll.insert_node_at_position(1, 0)
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "1 ->> 2")

    def test_insert_at_position_beyond_size(self):
        """Test inserting at position beyond size (should insert at tail)."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_node_at_position(3, 100)
        self.assertEqual(self.sll.size, 3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3")

    # ==================== Test find ====================
    def test_find_empty_list(self):
        """Test finding in empty list."""
        prev, curr = self.sll.find(10)
        self.assertIsNone(prev)
        self.assertIsNone(curr)

    def test_find_existing_element(self):
        """Test finding existing element."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        prev, curr = self.sll.find(2)
        self.assertIsNotNone(curr)
        self.assertEqual(curr.info, 2)
        self.assertEqual(prev.info, 1)

    def test_find_head(self):
        """Test finding head element."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        prev, curr = self.sll.find(1)
        self.assertIsNotNone(curr)
        self.assertEqual(curr.info, 1)
        self.assertEqual(prev.info, 2)  # In circular list, prev of head is tail

    def test_find_tail(self):
        """Test finding tail element."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        prev, curr = self.sll.find(2)
        self.assertIsNotNone(curr)
        self.assertEqual(curr.info, 2)
        self.assertEqual(prev.info, 1)

    def test_find_nonexistent_element(self):
        """Test finding non-existent element."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        prev, curr = self.sll.find(99)
        self.assertIsNone(prev)
        self.assertIsNone(curr)

    # ==================== Test insert_after_target ====================
    def test_insert_after_target_empty_list(self):
        """Test inserting after target in empty list."""
        result = self.sll.insert_after_target(10, 20)
        self.assertFalse(result)
        self.assertEqual(self.sll.size, 0)

    def test_insert_after_target_existing(self):
        """Test inserting after existing target."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(3)
        result = self.sll.insert_after_target(1, 2)
        self.assertTrue(result)
        self.assertEqual(self.sll.size, 3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3")

    def test_insert_after_target_tail(self):
        """Test inserting after tail."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        result = self.sll.insert_after_target(2, 3)
        self.assertTrue(result)
        self.assertEqual(self.sll.size, 3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3")

    def test_insert_after_target_nonexistent(self):
        """Test inserting after non-existent target."""
        self.sll.insert_as_tail(1)
        result = self.sll.insert_after_target(99, 2)
        self.assertFalse(result)
        self.assertEqual(self.sll.size, 1)

    # ==================== Test remove_head ====================
    def test_remove_head_empty_list(self):
        """Test removing head from empty list."""
        result = self.sll.remove_head()
        self.assertIsNone(result)
        self.assertEqual(self.sll.size, 0)

    def test_remove_head_single_element(self):
        """Test removing head from single element list."""
        self.sll.insert_as_tail(1)
        result = self.sll.remove_head()
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 1)
        self.assertEqual(self.sll.size, 0)
        self.assertIsNone(self.sll._head)
        self.assertIsNone(self.sll._tail)

    def test_remove_head_multiple_elements(self):
        """Test removing head from multiple element list."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        result = self.sll.remove_head()
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 1)
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "2 ->> 3")

    def test_remove_head_all_elements(self):
        """Test removing all elements via head."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.remove_head()
        self.sll.remove_head()
        self.assertEqual(self.sll.size, 0)
        self.assertIsNone(self.sll._head)

    # ==================== Test remove_tail ====================
    def test_remove_tail_empty_list(self):
        """Test removing tail from empty list."""
        result = self.sll.remove_tail()
        self.assertIsNone(result)
        self.assertEqual(self.sll.size, 0)

    def test_remove_tail_single_element(self):
        """Test removing tail from single element list."""
        self.sll.insert_as_tail(1)
        result = self.sll.remove_tail()
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 1)
        self.assertEqual(self.sll.size, 0)
        self.assertIsNone(self.sll._head)
        self.assertIsNone(self.sll._tail)

    def test_remove_tail_multiple_elements(self):
        """Test removing tail from multiple element list."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        result = self.sll.remove_tail()
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 3)
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "1 ->> 2")

    def test_remove_tail_all_elements(self):
        """Test removing all elements via tail."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.remove_tail()
        self.sll.remove_tail()
        self.assertEqual(self.sll.size, 0)
        self.assertIsNone(self.sll._head)

    # ==================== Test remove ====================
    def test_remove_empty_list(self):
        """Test removing from empty list."""
        result = self.sll.remove(10)
        self.assertIsNone(result)
        self.assertEqual(self.sll.size, 0)

    def test_remove_head_element(self):
        """Test removing head element."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        result = self.sll.remove(1)
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 1)
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "2 ->> 3")

    def test_remove_tail_element(self):
        """Test removing tail element."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        result = self.sll.remove(3)
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 3)
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "1 ->> 2")

    def test_remove_middle_element(self):
        """Test removing middle element."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        result = self.sll.remove(2)
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 2)
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "1 ->> 3")

    def test_remove_single_element(self):
        """Test removing single element."""
        self.sll.insert_as_tail(1)
        result = self.sll.remove(1)
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 1)
        self.assertEqual(self.sll.size, 0)

    def test_remove_nonexistent_element(self):
        """Test removing non-existent element."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        result = self.sll.remove(99)
        self.assertIsNone(result)
        self.assertEqual(self.sll.size, 2)

    # ==================== Test remove_node_at_position ====================
    def test_remove_at_position_empty_list(self):
        """Test removing at position from empty list."""
        result = self.sll.remove_node_at_position(1)
        self.assertIsNone(result)
        self.assertEqual(self.sll.size, 0)

    def test_remove_at_position_beginning(self):
        """Test removing at position 1."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        result = self.sll.remove_node_at_position(1)
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 1)
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "2 ->> 3")

    def test_remove_at_position_middle(self):
        """Test removing at middle position."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        result = self.sll.remove_node_at_position(2)
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 2)
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "1 ->> 3")

    def test_remove_at_position_end(self):
        """Test removing at end position."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        result = self.sll.remove_node_at_position(3)
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 3)
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "1 ->> 2")

    def test_remove_at_position_negative(self):
        """Test removing at negative position."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        result = self.sll.remove_node_at_position(-1)  # Should remove last element
        self.assertIsNotNone(result)
        self.assertEqual(result.info, 3)
        self.assertEqual(self.sll.size, 2)

    def test_remove_at_position_invalid(self):
        """Test removing at invalid position."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        result = self.sll.remove_node_at_position(100)
        self.assertIsNone(result)
        self.assertEqual(self.sll.size, 2)

    def test_remove_at_position_zero(self):
        """Test removing at position 0 (invalid)."""
        self.sll.insert_as_tail(1)
        result = self.sll.remove_node_at_position(0)
        self.assertIsNone(result)
        self.assertEqual(self.sll.size, 1)

    # ==================== Test reverse_in_place ====================
    def test_reverse_empty_list(self):
        """Test reversing empty list."""
        self.sll.reverse_in_place()
        self.assertEqual(self.sll.size, 0)
        self.assertIsNone(self.sll._head)

    def test_reverse_single_element(self):
        """Test reversing single element list."""
        self.sll.insert_as_tail(1)
        self.sll.reverse_in_place()
        self.assertEqual(self.sll.size, 1)
        self.assertEqual(self.sll._head.info, 1)
        self.assertEqual(str(self.sll), "1")

    def test_reverse_two_elements(self):
        """Test reversing two element list."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.reverse_in_place()
        self.assertEqual(self.sll.size, 2)
        self.assertEqual(str(self.sll), "2 ->> 1")

    def test_reverse_multiple_elements(self):
        """Test reversing multiple element list."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        self.sll.insert_as_tail(4)
        self.sll.insert_as_tail(5)
        self.sll.reverse_in_place()
        self.assertEqual(self.sll.size, 5)
        self.assertEqual(str(self.sll), "5 ->> 4 ->> 3 ->> 2 ->> 1")

    def test_reverse_then_operations(self):
        """Test operations after reversing."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        self.sll.reverse_in_place()
        self.sll.insert_as_tail(4)
        self.assertEqual(str(self.sll), "3 ->> 2 ->> 1 ->> 4")

    # ==================== Test is_empty ====================
    def test_is_empty_empty_list(self):
        """Test is_empty on empty list."""
        self.assertTrue(self.sll.is_empty())

    def test_is_empty_non_empty_list(self):
        """Test is_empty on non-empty list."""
        self.sll.insert_as_tail(1)
        self.assertFalse(self.sll.is_empty())

    # ==================== Test size property ====================
    def test_size_property(self):
        """Test size property."""
        self.assertEqual(self.sll.size, 0)
        self.sll.insert_as_tail(1)
        self.assertEqual(self.sll.size, 1)
        self.sll.insert_as_tail(2)
        self.assertEqual(self.sll.size, 2)

    # ==================== Test __str__ ====================
    def test_str_empty_list(self):
        """Test string representation of empty list."""
        self.assertEqual(str(self.sll), "[]")

    def test_str_single_element(self):
        """Test string representation of single element."""
        self.sll.insert_as_tail(1)
        self.assertEqual(str(self.sll), "1")

    def test_str_multiple_elements(self):
        """Test string representation of multiple elements."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3")

    # ==================== Test circular property ====================
    def test_circular_property_single(self):
        """Test circular property with single node."""
        self.sll.insert_as_tail(1)
        self.assertEqual(self.sll._head.next, self.sll._head)
        self.assertEqual(self.sll._tail.next, self.sll._head)

    def test_circular_property_multiple(self):
        """Test circular property with multiple nodes."""
        self.sll.insert_as_tail(1)
        self.sll.insert_as_tail(2)
        self.sll.insert_as_tail(3)
        self.assertEqual(self.sll._tail.next, self.sll._head)

    # ==================== Test complex scenarios ====================
    def test_complex_operations(self):
        """Test complex sequence of operations."""
        # Insert elements
        self.sll.insert_as_head(3)
        self.sll.insert_as_head(1)
        self.sll.insert_as_tail(5)
        self.sll.insert_node_at_position(2, 2)
        self.sll.insert_node_at_position(4, 4)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 3 ->> 4 ->> 5")
        
        # Remove elements
        self.sll.remove(3)
        self.assertEqual(str(self.sll), "1 ->> 2 ->> 4 ->> 5")
        
        # Reverse
        self.sll.reverse_in_place()
        self.assertEqual(str(self.sll), "5 ->> 4 ->> 2 ->> 1")
        
        # More operations
        self.sll.remove_head()
        self.sll.remove_tail()
        self.assertEqual(str(self.sll), "4 ->> 2")


if __name__ == '__main__':
    unittest.main()











