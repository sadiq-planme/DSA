"""   
    Implementation Details:
    • Data Structure: TrieNode dataclass with is_terminal flag and defaultdict-based children
    • Time Complexity: Insert/Search/Delete O(m), Space O(ALPHABET_SIZE × N × M)
    • Use cases: Autocomplete, prefix matching, word search, IP routing
    • Uses hashmap-based children (defaultdict) for flexible character support (Unicode-friendly)
    • Key design: is_terminal marks complete words, children defaultdict[str, TrieNode] for next chars
    
    Core Methods:
    • insert(word): Inserts word into trie, marks terminal node. O(m)
    • search(word): Checks if word exists as complete word (requires is_terminal=True). O(m)
    • starts_with(prefix): Checks if any word starts with prefix (empty prefix matches all). O(m)
    • _find_node(word): Helper to find node for word/prefix, returns None if path doesn't exist. O(m)
    
    Deletion Methods:
    • soft_delete(word): Removes terminal marker only, keeps nodes for memory efficiency. O(m)
    • hard_delete(word): Recursively deletes word and removes unused nodes (bottom-up). O(m)
    
    Advanced Methods:
    • autocomplete(prefix): Returns all words starting with prefix using DFS. O(n) where n is subtree size
    • count_words_with_prefix(prefix): Returns count of words starting with prefix. O(m + n)
    • longest_common_prefix(): Returns longest common prefix of all words by following single-child path. O(n)
    • word_suggestions(word): Returns autocomplete suggestions for each prefix of word. O(m * n)
    
    Helper Methods:
    • _get_prefix_node(prefix): Returns node for prefix, handles empty prefix (returns root). O(m)
    
    Tricky Questions:
    • Trie vs HashTable: Trie supports prefix search & lexicographic order; HashTable O(1) exact match
    • When to use: Need prefix matching → Trie; Only exact match → HashSet/HashMap
    • Deletion gotcha: Don't delete nodes shared by other words (check is_terminal & has children)
    • Search vs starts_with: search() requires is_terminal=True; starts_with() just checks path exists
    • Empty prefix: starts_with('') returns True (matches all words), autocomplete('') returns all words
    
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
from collections import defaultdict
from dataclasses import dataclass, field


@dataclass
class TrieNode:
    is_terminal: bool = False
    children: defaultdict[str, TrieNode] = field(default_factory=lambda: defaultdict(TrieNode))


class Trie:

    def __init__(self):
        # Root node represents empty prefix, all words start from here
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        """Inserts a word into the trie. Time: O(m) where m is word length."""
        if not word:
            return  # Empty words are not stored
        curr_node = self.root
        for char in word:
            # defaultdict automatically creates TrieNode() when accessing missing key (char)
            curr_node = curr_node.children[char]
        # Mark end of word so search() can distinguish complete words from prefixes
        curr_node.is_terminal = True

    def _find_node(self, word: str) -> TrieNode | None:
        """Finds node for given word/prefix. Returns None if path doesn't exist. Time: O(m)."""
        if not word:
            return None  # Empty string has no path in trie
        curr_node = self.root
        for char in word:
            if char not in curr_node.children:
                return None  # Path broken, word/prefix doesn't exist
            curr_node = curr_node.children[char]
        return curr_node

    def search(self, word: str) -> bool:
        """Checks if word exists as complete word. Time: O(m)."""
        node = self._find_node(word)
        # Must check is_terminal: "app" exists but "apple" path may continue beyond "app"
        return node.is_terminal if node else False

    def starts_with(self, prefix: str) -> bool:
        """Checks if any word starts with prefix. Time: O(m)."""
        # Empty prefix matches all words (common autocomplete behavior)
        current_node = self._find_node(prefix)
        return True if current_node is not None else False

    def soft_delete(self, word: str) -> bool:
        """Removes terminal marker only. Time: O(m)."""
        # Soft delete: keeps nodes for memory efficiency when words share prefixes
        # Use when you might re-insert the word later or want faster deletion
        node = self._find_node(word)
        if not node or not node.is_terminal:
            return False  # Word doesn't exist
        node.is_terminal = False
        return True

    # DSA Problems
    def hard_delete(self, word: str):
        """
            Deletes word and removes unused nodes (bottom-up cleanup).
            Time Complexity: O(m) where m is the word length.
            Space Complexity: O(m) because of the recursive call stack.
            
            Algorithm:
            1. Recursively traverse to the end of the word
            2. Unmark the terminal node
            3. On backtrack, delete nodes that are no longer needed
            4. A node can be deleted if it has no children and is not a terminal
        """
        if not word:
            return
        
        def is_it_a_leaf_node(current_node: TrieNode, word_char_iter: int) -> bool:
            # Base case: We reached the end of the word.
            if word_char_iter == len(word):
                # Word doesn't exist if current node is not marked as terminal
                # Since it is not a terminal node, it can not be a leaf node. Because all the LEAF nodes are terminal nodes, in a TRIE TREE if hard deletion is followed.
                # NOTE: the reverse is not true. That is not all the terminal nodes are leaf nodes.
                if not current_node.is_terminal:
                    return False
                
                # handle the soft deletion of the word.
                current_node.is_terminal = False
                
                # Node can be deleted if it has no children (leaf node)
                return len(current_node.children) == 0
            
            # Return False if Word doesn't exist in the trie
            child_char = word[word_char_iter]
            if child_char not in current_node.children:  # MISTAKE: i missed this block in self test
                return False
            
            # Recursively traverse to the end of the word
            child_node = current_node.children[child_char]
            child_is_leaf_flag = is_it_a_leaf_node(child_node, word_char_iter + 1)
            
            # If child is a leaf node remove it from the Trie.
            # After removing the child, check if the parent node is a leaf node and can be deleted.
            # Parent node is a leaf node and can be deleted if:
            # 1. It has no children left, AND
            # 2. It's not a terminal node (not part of another word)
            if child_is_leaf_flag:
                del current_node.children[child_char]
                return len(current_node.children) == 0 and not current_node.is_terminal
            
            return False
        
        is_it_a_leaf_node(self.root, 0)

    def word_matcher(self, paragraph: str, target_word: str) -> bool:
        """
            Given a paragraph, pre-process it into a Trie.
            Then given word as an input, return true or false if it's in the paragraph.

            Example:
            Paragraph: "The quick brown fox jumps over the lazy dog"
            Word: "fox"
            Output: True

            Paragraph: "The quick brown fox jumps over the lazy dog"
            Word: "cat"
            Output: False
        """
        for word in paragraph.split():
            self.insert(word)
        return self.search(target_word)

    def _get_prefix_node(self, prefix: str) -> TrieNode | None:
        """Helper to get node for prefix, handling empty prefix. Returns root if prefix is empty."""
        # Empty prefix means start from root (all words match)
        return self.root if not prefix else self._find_node(prefix)

    def autocomplete(self, prefix: str) -> list[str]:
        """Returns all words starting with prefix. Time: O(n) where n is subtree size."""
        node = self._get_prefix_node(prefix)
        if not node:
            return []  # Prefix doesn't exist
        
        results: list[str] = []
        def dfs(curr: TrieNode, path: list[str]) -> None:
            # DFS collects all terminal nodes (complete words) in subtree
            if curr.is_terminal:
                results.append(''.join(path))
            # Explore all children to find all words with this prefix
            for char, child in curr.children.items():
                path.append(char)
                dfs(child, path)
                path.pop()  # Backtrack: remove char before exploring sibling
        
        dfs(node, list(prefix))
        return results

    def count_words_with_prefix(self, prefix: str) -> int:
        """Returns count of words starting with prefix. Time: O(m + n)."""
        node = self._get_prefix_node(prefix)
        if not node:
            return 0
        
        def helper(curr: TrieNode) -> int:
            # Count this node if it's a word, plus recursively count all children
            total = 1 if curr.is_terminal else 0
            for child in curr.children:
                child_node = curr.children[child]
                total += helper(child_node)
            return total
        
        return helper(node)

    def longest_common_prefix(self) -> str:
        """Returns longest common prefix of all words. Time: O(n)."""
        # Algorithm: follow path while there's exactly one child (common path)
        # Stop when paths diverge (>1 child) or a word ends (is_terminal)
        prefix: list[str] = []
        curr = self.root
        while len(curr.children) == 1:
            char, child = next(iter(curr.children.items()))
            prefix.append(char)
            if child.is_terminal:
                break  # Shortest word ends here, can't extend further
            curr = child
        return ''.join(prefix)

    def word_suggestions(self, word: str) -> list[list[str]]:
        """Returns autocomplete suggestions for each prefix of word. Time: O(m * n), optimized single traversal."""
        # Example: "app" -> [[words with "a"], [words with "ap"], [words with "app"]]
        if not word:
            return [self.autocomplete('')]
        
        suggestions: list[list[str]] = []
        current_node = self.root
        current_prefix: list[str] = []
        
        # Single traversal: build suggestions as we traverse the word
        for char in word:
            # If prefix path doesn't exist, no suggestions for this and remaining prefixes
            if char not in current_node.children:
                suggestions.append([])
                break
            
            current_node = current_node.children[char]
            current_prefix.append(char)
            
            # Inline DFS to collect all words from current node (avoids autocomplete overhead)
            prefix_words: list[str] = []
            def collect_words(node: TrieNode, path: list[str]) -> None:
                if node.is_terminal:
                    prefix_words.append(''.join(path))
                for child_char, child_node in node.children.items():
                    path.append(child_char)
                    collect_words(child_node, path)
                    path.pop()  # Backtrack for sibling exploration
            
            collect_words(current_node, current_prefix)
            suggestions.append(prefix_words)
        
        # Fill remaining positions with empty lists if path broke early
        suggestions.extend([[]] * (len(word) - len(suggestions)))
        
        return suggestions
