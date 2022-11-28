from __future__ import annotations
from typing import Optional, Sequence

__all__ = [
    "longest_common_subsequence",
]


def longest_common_subsequence(seq1: Sequence, seq2: Sequence) -> int:
    """
    References:
        https://www.programiz.com/dsa/longest-common-subsequence
        https://www.geeksforgeeks.org/python-program-for-longest-common-subsequence/

    Args:
        seq1: the first sequence
        seq2: the second sequence

    Returns: the size of the longest common subsequence

    """
    # find the length of the strings
    m = len(seq1)
    n = len(seq2)

    # init dp table
    # dp[i][j] the size of the longest subsequence between seq1[:i] and seq2[:j]
    dp: list[list[Optional[int]]] = [[0] * (n + 1) for _ in range(m + 1)]

    # build dp table in bottom-up fashion
    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 or j == 0:
                dp[i][j] = 0
            elif seq1[i - 1] == seq2[j - 1]:
                # if matches, move to next element
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                # if not match, use the max of the previous values
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

    return dp[m][n]
