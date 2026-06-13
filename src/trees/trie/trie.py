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
