"""
    Trie (Prefix Tree) - FANG Interview Quick Reference
    • Time: Insert/Search/Delete O(m), Space O(ALPHABET_SIZE × N × M)
    • Use cases: Autocomplete, prefix matching, word search, IP routing
    • Assumes lowercase a-z (26 children array). For Unicode, use hashmap.
    • Key: is_terminal marks complete words, children array for next chars

    Tricky Questions:
    • Trie vs HashTable: Trie supports prefix search & lexicographic order; HashTable O(1) exact match
    • When to use: Need prefix matching → Trie; Only exact match → HashSet/HashMap
    • Deletion gotcha: Don't delete nodes shared by other words (check is_terminal & has children)
    • Search vs starts_with: search() requires is_terminal=True; starts_with() just checks path exists

    Linked Problems:
    • Word Search II: Build Trie from words, DFS on 2D grid with backtracking
    • Design Add/Search Words: Handle '.' wildcard with recursive DFS on all children
    • Longest Common Prefix: Insert all, traverse until node has >1 child or is_terminal
    • Maximum XOR: Bitwise Trie (binary tree) for XOR queries
    • Stream of Characters: Reverse Trie + sliding window for suffix matching
    • Replace Words: Build Trie from roots, find shortest prefix for each word

    Space Optimization:
    • Compressed Trie (Radix Tree): Merge single-child nodes to reduce space
    • Ternary Search Tree: 3 children per node (left/middle/right) for space-time tradeoff
"""

from __future__ import annotations
from dataclasses import dataclass, field


@dataclass
class TrieNode:
    is_terminal: bool = False
    children: list[TrieNode] = field(default_factory=lambda: [None] * 26)


class Trie:
    """
        Trie implementation for string operations.
        Assumes all operations use lowercase English letters (a-z).
    """
    def __init__(self):
        self.root = TrieNode()
    
    def _get_index(self, char: str) -> int:
        """Converts character to array index (a=0, b=1, ..., z=25)."""
        if not char or len(char) != 1 or not char.islower() or not char.isalpha():
            raise ValueError(f"Invalid character: '{char}'. Must be a lowercase letter (a-z).")
        idx = ord(char) - ord('a')
        if not (0 <= idx < 26):
            raise ValueError(f"Invalid character: '{char}'. Must be a lowercase letter (a-z).")
        return idx
    
    def _find_node(self, word: str) -> TrieNode | None:
        """Traverses the trie to find the node for given word/prefix."""
        if not word:
            return self.root
        node = self.root
        for char in word:
            idx = self._get_index(char)
            if not node.children[idx]:
                return None
            node = node.children[idx]
        return node
    
    def insert(self, word: str) -> None:
        """
            Inserts a word into the trie.
            Time: O(m) where m is word length.
        """
        if not word:
            # Empty string: mark root as terminal
            self.root.is_terminal = True
            return
        node = self.root
        for char in word:
            idx = self._get_index(char)
            if not node.children[idx]:
                node.children[idx] = TrieNode()
            node = node.children[idx]
        node.is_terminal = True
    
    def search(self, word: str) -> bool:
        """
            Checks if word exists in trie (must be complete word, not just prefix).
            Time: O(m) where m is word length.
        """
        node = self._find_node(word)
        return node.is_terminal if node else False
    
    def starts_with(self, prefix: str) -> bool:
        """
            Checks if any word in trie starts with given prefix.
            Time: O(m) where m is prefix length.
        """
        return self._find_node(prefix) is not None
    
    def soft_delete(self, word: str) -> bool:
        """
            Returns True if word was found and deleted, False otherwise.
            Time: O(m) where m is word length.
        """
        node = self._find_node(word)
        if not node or not node.is_terminal:
            return False
        
        node.is_terminal = False
        return True
    
    def autocomplete(self, prefix: str) -> list[str]:
        """
            Returns all words in trie that start with given prefix.
            Time: O(n) where n is number of nodes in subtree.
        """
        node = self._find_node(prefix)
        if not node:
            return []
        
        results = []
        
        def dfs(curr: TrieNode, path: str) -> None:
            if curr.is_terminal:
                results.append(prefix + path)
            for i, child in enumerate(curr.children):
                if child:
                    dfs(child, path + chr(ord('a') + i))
        
        dfs(node, '')
        return results
    
    def count_words_with_prefix(self, prefix: str) -> int:
        """
            Returns count of words that start with given prefix.
            Time: O(m + n) where m is prefix length, n is subtree size.
        """
        node = self._find_node(prefix)
        if not node:
            return 0
        
        def count(node: TrieNode) -> int:
            result = 1 if node.is_terminal else 0
            for child in node.children:
                if child:
                    result += count(child)
            return result
        
        return count(node)

    def longest_common_prefix(self) -> str:
        """
            Returns the longest common prefix of all words in the trie.
            Time: O(n) where n is number of nodes in trie.
        """
        node = self.root
        prefix = ''
        
        # Continue while there's exactly one child
        while True:
            # Count non-None children
            non_none_children = [i for i, child in enumerate(node.children) if child is not None]
            
            # Stop if there are 0 or more than 1 children
            if len(non_none_children) != 1:
                break
            
            # Get the single child
            idx = non_none_children[0]
            child_node = node.children[idx]
            
            # Stop if the child node is terminal (a complete word ends here)
            # This ensures we don't extend beyond the shortest word
            if child_node.is_terminal:
                prefix += chr(ord('a') + idx)
                break
            
            node = child_node
            prefix += chr(ord('a') + idx)
        
        return prefix

    def word_suggestions(self, word: str) -> list[list[str]]:
        """
            Returns suggestions for each prefix of the word.
            For each character position, returns all words that start with the prefix up to that point.
            Time: O(m * n) where m is word length, n is average subtree size.
            Space: O(m + h) where h is height of subtree (recursion stack).
        """
        if not word:
            return [self.autocomplete('')]
        
        ans = []
        for i in range(len(word)):
            prefix = word[:i+1]
            # Validate prefix characters before autocomplete
            try:
                suggestions = self.autocomplete(prefix)
                ans.append(suggestions)
            except ValueError:
                # Invalid character encountered, return empty suggestions for remaining prefixes
                ans.append([])
        return ans
