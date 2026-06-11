"""
    Time Complexity: O(N^(N+1)) => O(N!*N) - backtracking with pruning. N is the number of queens. => O(N!)
    Space Complexity:
        - Input Space: O(1) - the input is the number of queens.
        - Auxiliary Space: O(N²) - one N×N board. => O(N)
        - Stack Space: O(N) - recursion depth = rows. => O(1)
        - Output Space: O(a * N²) - a is the number of solutions, N² is the size of a single board. => O(a * N)
"""

from typing import List, Tuple

def solve_n_queens(n: int) -> Tuple[List[List[List[str]]], int]:
    # Edge case
    if n <= 0:
        return [], 0

    # Initializing the variables
    board: List[List[str]] = [["."] * n for _ in range(n)]
    solutions: List[List[List[str]]] = []
    count: int = 0

    # Helper function to check if a queen can be placed at (row, col)
    # Time Complexity: O(N)
    def is_safe(row: int, col: int) -> bool:
        """Any queen in columns 0, ..., col-1 attacking (row, col)?"""
        # check this row on left side
        for c in range(col):
            if board[row][c] == "Q":
                return False

        # check upper diagonal on left side
        r, c = row, col
        while r >= 0 and c >= 0:
            if board[r][c] == "Q":
                return False
            r -= 1
            c -= 1

        # check lower diagonal on left side
        r, c = row, col
        while r < n and c >= 0:
            if board[r][c] == "Q":
                return False
            r += 1
            c -= 1

        # if no queens attacking, return True
        return True

    # Helper function to place a queen in the next column
    # Time Complexity: O(N^(N+1)) => O(N!)
    # Auxiliary Space Complexity: O(N²)
    # Stack Space Complexity: O(N)
    # Output Space Complexity: O(a * N²) - a is the number of solutions, N² is the size of a single board.
    def place_queen(col: int) -> None:
        nonlocal count
        # BASE CASE: if all queens are placed, add the board to the solutions
        # Time Complexity: O(N^2)
        if col == n:
            solutions.append([row.copy() for row in board])
            count += 1
            return

        for row in range(n):
            # check if safe
            if is_safe(row, col):
                # place queen
                board[row][col] = "Q"
                # recurse to next row
                place_queen(col + 1)
                # backtrack
                board[row][col] = "."

    place_queen(0)
    return solutions, count


for N in (1, 2, 3, 4, 5):
    boards, count = solve_n_queens(N)
    print(f"N = {N}: {count} solutions")
    for i, board in enumerate(boards):
        print(f"Solution {i + 1} : ")
        for row in board:
            print(" ".join(row))
        print()
    print()
