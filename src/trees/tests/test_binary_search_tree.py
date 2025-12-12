"""
Comprehensive unit tests for BinarySearchTree implementation.
Tests all methods with edge cases and sample data.
"""

import unittest
import sys
import os
import importlib.util

# Import module with hyphenated name
module_path = os.path.join(os.path.dirname(__file__), "binary-search-tree.py")
spec = importlib.util.spec_from_file_location("binary_search_tree", module_path)
binary_search_tree = importlib.util.module_from_spec(spec)
sys.modules["binary_search_tree"] = binary_search_tree
spec.loader.exec_module(binary_search_tree)
BinarySearchTree = binary_search_tree.BinarySearchTree
TreeNode = binary_search_tree.TreeNode


class TestBinarySearchTree(unittest.TestCase):

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.bst = BinarySearchTree()

    def test_build_bst_empty_list(self):
        """Test building BST from empty list."""
        result = self.bst.build_bst([])
        self.assertIsNone(result)
        self.assertIsNone(self.bst.root)

    def test_build_bst_single_node(self):
        """Test building BST with single node."""
        result = self.bst.build_bst([5])
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 5)
        self.assertIsNone(result.left)
        self.assertIsNone(result.right)

    def test_build_bst_multiple_nodes(self):
        """Test building BST with multiple nodes."""
        nodes = [5, 3, 7, 2, 4, 6, 8]
        result = self.bst.build_bst(nodes)
        self.assertIsNotNone(result)
        # Verify structure
        self.assertEqual(result.data, 5)
        self.assertEqual(result.left.data, 3)
        self.assertEqual(result.right.data, 7)

    def test_build_bst_duplicate_values(self):
        """Test building BST with duplicate values."""
        nodes = [5, 3, 5, 7, 3]
        result = self.bst.build_bst(nodes)
        self.assertIsNotNone(result)
        # Duplicates should go to right subtree
        self.assertEqual(result.data, 5)
        self.assertEqual(result.right.data, 5)

    def test_search_empty_tree(self):
        """Test searching in empty tree."""
        self.assertFalse(self.bst.search(5))

    def test_search_existing_node(self):
        """Test searching for existing node."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.assertTrue(self.bst.search(5))
        self.assertTrue(self.bst.search(3))
        self.assertTrue(self.bst.search(7))
        self.assertTrue(self.bst.search(2))

    def test_search_non_existing_node(self):
        """Test searching for non-existing node."""
        self.bst.build_bst([5, 3, 7])
        self.assertFalse(self.bst.search(10))
        self.assertFalse(self.bst.search(1))

    def test_min_empty_tree(self):
        """Test finding minimum in empty tree."""
        with self.assertRaises(ValueError):
            self.bst.min()

    def test_min_single_node(self):
        """Test finding minimum with single node."""
        self.bst.build_bst([5])
        self.assertEqual(self.bst.min(), 5)

    def test_min_multiple_nodes(self):
        """Test finding minimum with multiple nodes."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.assertEqual(self.bst.min(), 2)

    def test_max_empty_tree(self):
        """Test finding maximum in empty tree."""
        with self.assertRaises(ValueError):
            self.bst.max()

    def test_max_single_node(self):
        """Test finding maximum with single node."""
        self.bst.build_bst([5])
        self.assertEqual(self.bst.max(), 5)

    def test_max_multiple_nodes(self):
        """Test finding maximum with multiple nodes."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.assertEqual(self.bst.max(), 8)

    def test_delete_empty_tree(self):
        """Test deleting from empty tree."""
        result = self.bst.delete(5)
        self.assertIsNone(result)

    def test_delete_leaf_node(self):
        """Test deleting a leaf node."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.bst.delete(2)
        self.assertFalse(self.bst.search(2))
        self.assertTrue(self.bst.search(5))
        self.assertTrue(self.bst.search(3))

    def test_delete_node_with_one_child(self):
        """Test deleting node with one child."""
        self.bst.build_bst([5, 3, 7, 2])
        self.bst.delete(3)
        self.assertFalse(self.bst.search(3))
        self.assertTrue(self.bst.search(2))
        self.assertTrue(self.bst.search(5))

    def test_delete_node_with_two_children(self):
        """Test deleting node with two children."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.bst.delete(5)
        self.assertFalse(self.bst.search(5))
        # Root should be replaced by successor (6)
        self.assertTrue(self.bst.search(6))
        self.assertTrue(self.bst.search(3))
        self.assertTrue(self.bst.search(7))

    def test_delete_root_single_node(self):
        """Test deleting root when it's the only node."""
        self.bst.build_bst([5])
        self.bst.delete(5)
        self.assertIsNone(self.bst.root)
        self.assertFalse(self.bst.search(5))

    def test_delete_non_existing_node(self):
        """Test deleting non-existing node."""
        self.bst.build_bst([5, 3, 7])
        result = self.bst.delete(10)
        self.assertIsNotNone(result)
        self.assertTrue(self.bst.search(5))

    def test_is_bst_empty_tree(self):
        """Test is_bst on empty tree."""
        self.assertTrue(self.bst.is_bst())

    def test_is_bst_valid_tree(self):
        """Test is_bst on valid BST."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.assertTrue(self.bst.is_bst())

    def test_is_bst_single_node(self):
        """Test is_bst with single node."""
        self.bst.build_bst([5])
        self.assertTrue(self.bst.is_bst())

    def test_lca_empty_tree(self):
        """Test LCA in empty tree."""
        result = self.bst.lca(3, 7)
        self.assertIsNone(result)

    def test_lca_both_values_exist(self):
        """Test LCA when both values exist."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        result = self.bst.lca(2, 4)
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 3)
        
        result = self.bst.lca(2, 8)
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 5)

    def test_lca_one_value_is_root(self):
        """Test LCA when one value is root."""
        self.bst.build_bst([5, 3, 7])
        result = self.bst.lca(5, 3)
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 5)

    def test_lca_same_value(self):
        """Test LCA with same value."""
        self.bst.build_bst([5, 3, 7])
        result = self.bst.lca(3, 3)
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 3)

    def test_kth_smallest_empty_tree(self):
        """Test kth_smallest on empty tree."""
        self.assertIsNone(self.bst.kth_smallest(1))

    def test_kth_smallest_valid_k(self):
        """Test kth_smallest with valid k."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.assertEqual(self.bst.kth_smallest(1), 2)
        self.assertEqual(self.bst.kth_smallest(2), 3)
        self.assertEqual(self.bst.kth_smallest(3), 4)
        self.assertEqual(self.bst.kth_smallest(4), 5)

    def test_kth_smallest_invalid_k(self):
        """Test kth_smallest with invalid k."""
        self.bst.build_bst([5, 3, 7])
        self.assertIsNone(self.bst.kth_smallest(10))
        self.assertIsNone(self.bst.kth_smallest(0))

    def test_kth_largest_empty_tree(self):
        """Test kth_largest on empty tree."""
        self.assertIsNone(self.bst.kth_largest(1))

    def test_kth_largest_valid_k(self):
        """Test kth_largest with valid k."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.assertEqual(self.bst.kth_largest(1), 8)
        self.assertEqual(self.bst.kth_largest(2), 7)
        self.assertEqual(self.bst.kth_largest(3), 6)
        self.assertEqual(self.bst.kth_largest(4), 5)

    def test_kth_largest_invalid_k(self):
        """Test kth_largest with invalid k."""
        self.bst.build_bst([5, 3, 7])
        self.assertIsNone(self.bst.kth_largest(10))
        self.assertIsNone(self.bst.kth_largest(0))

    def test_build_balanced_bst_from_in_order_empty(self):
        """Test building balanced BST from empty in-order list."""
        result = self.bst.build_balanced_bst_from_in_order([])
        self.assertIsNone(result)

    def test_build_balanced_bst_from_in_order_single(self):
        """Test building balanced BST from single element."""
        result = self.bst.build_balanced_bst_from_in_order([5])
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 5)

    def test_build_balanced_bst_from_in_order_multiple(self):
        """Test building balanced BST from multiple elements."""
        in_order = [1, 2, 3, 4, 5, 6, 7]
        result = self.bst.build_balanced_bst_from_in_order(in_order)
        self.assertIsNotNone(result)
        # Verify it's balanced and correct
        self.assertEqual(result.data, 4)
        self.assertEqual(result.left.data, 2)
        self.assertEqual(result.right.data, 6)

    def test_inorder_traversal_empty(self):
        """Test inorder traversal of empty tree."""
        result = self.bst.inorder_traversal()
        self.assertEqual(result, [])

    def test_inorder_traversal_single(self):
        """Test inorder traversal of single node."""
        self.bst.build_bst([5])
        result = self.bst.inorder_traversal()
        self.assertEqual(result, [5])

    def test_inorder_traversal_multiple(self):
        """Test inorder traversal of multiple nodes."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        result = self.bst.inorder_traversal()
        self.assertEqual(result, [2, 3, 4, 5, 6, 7, 8])

    def test_two_sum_empty_tree(self):
        """Test two_sum on empty tree."""
        result = self.bst.two_sum(10)
        self.assertIsNone(result)

    def test_two_sum_found(self):
        """Test two_sum when pair exists."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        result = self.bst.two_sum(10)
        self.assertIsNotNone(result)
        self.assertEqual(result[0] + result[1], 10)

    def test_two_sum_not_found(self):
        """Test two_sum when pair doesn't exist."""
        self.bst.build_bst([5, 3, 7])
        result = self.bst.two_sum(100)
        self.assertIsNone(result)

    def test_convert_bst_into_sorted_dll_empty(self):
        """Test converting empty BST to DLL."""
        result = self.bst.convert_bst_into_sorted_dll()
        self.assertIsNone(result)

    def test_convert_bst_into_sorted_dll_single(self):
        """Test converting single node BST to DLL."""
        self.bst.build_bst([5])
        result = self.bst.convert_bst_into_sorted_dll()
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 5)
        self.assertIsNone(result.left)
        self.assertIsNone(result.right)

    def test_convert_bst_into_sorted_dll_multiple(self):
        """Test converting multiple node BST to DLL."""
        self.bst.build_bst([5, 3, 7, 2, 4])
        result = self.bst.convert_bst_into_sorted_dll()
        self.assertIsNotNone(result)
        # Verify DLL structure (should be sorted: 2, 3, 4, 5, 7)
        self.assertEqual(result.data, 2)
        self.assertIsNotNone(result.right)
        self.assertEqual(result.right.data, 3)

    def test_convert_sorted_dll_into_bst_empty(self):
        """Test converting empty DLL to BST."""
        result = self.bst.convert_sorted_dll_into_bst(None, 0)
        self.assertIsNone(result)

    def test_convert_sorted_dll_into_bst_single(self):
        """Test converting single node DLL to BST."""
        head = TreeNode(5)
        result = self.bst.convert_sorted_dll_into_bst(head, 1)
        self.assertIsNotNone(result)
        self.assertEqual(result.data, 5)

    def test_largest_bst_subtree_size_in_bt_empty(self):
        """Test largest BST subtree size in empty tree."""
        result = self.bst.largest_bst_subtree_size_in_bt()
        self.assertEqual(result, 0)

    def test_largest_bst_subtree_size_in_bt_valid_bst(self):
        """Test largest BST subtree size when entire tree is BST."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        result = self.bst.largest_bst_subtree_size_in_bt()
        self.assertEqual(result, 7)

    def test_largest_bst_subtree_size_in_bt_single_node(self):
        """Test largest BST subtree size with single node."""
        self.bst.build_bst([5])
        result = self.bst.largest_bst_subtree_size_in_bt()
        self.assertEqual(result, 1)

    def test_kth_smallest_all_nodes(self):
        """Test kth_smallest for all nodes in tree."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.assertEqual(self.bst.kth_smallest(1), 2)
        self.assertEqual(self.bst.kth_smallest(2), 3)
        self.assertEqual(self.bst.kth_smallest(3), 4)
        self.assertEqual(self.bst.kth_smallest(4), 5)
        self.assertEqual(self.bst.kth_smallest(5), 6)
        self.assertEqual(self.bst.kth_smallest(6), 7)
        self.assertEqual(self.bst.kth_smallest(7), 8)
        self.assertIsNone(self.bst.kth_smallest(8))

    def test_kth_largest_all_nodes(self):
        """Test kth_largest for all nodes in tree."""
        self.bst.build_bst([5, 3, 7, 2, 4, 6, 8])
        self.assertEqual(self.bst.kth_largest(1), 8)
        self.assertEqual(self.bst.kth_largest(2), 7)
        self.assertEqual(self.bst.kth_largest(3), 6)
        self.assertEqual(self.bst.kth_largest(4), 5)
        self.assertEqual(self.bst.kth_largest(5), 4)
        self.assertEqual(self.bst.kth_largest(6), 3)
        self.assertEqual(self.bst.kth_largest(7), 2)
        self.assertIsNone(self.bst.kth_largest(8))

    def test_delete_all_nodes(self):
        """Test deleting all nodes from tree."""
        self.bst.build_bst([5, 3, 7])
        self.bst.delete(5)
        self.bst.delete(3)
        self.bst.delete(7)
        self.assertIsNone(self.bst.root)
        self.assertFalse(self.bst.search(5))

    def test_two_sum_edge_cases(self):
        """Test two_sum with various edge cases."""
        self.bst.build_bst([1, 2, 3, 4, 5])
        # Test with sum that exists
        result = self.bst.two_sum(3)
        self.assertIsNotNone(result)
        self.assertEqual(result[0] + result[1], 3)
        # Test with sum that doesn't exist
        self.assertIsNone(self.bst.two_sum(100))


if __name__ == '__main__':
    unittest.main()
