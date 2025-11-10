"""
File name: similarity
Author: Fran Moreno
Last Updated: 11/6/2025
Version: 1.0
Description: TOFILL
"""
from difflib import SequenceMatcher


def string_inclusion_boost(a: str, b: str) -> float:
    """
    Computes a similarity boost based on whether one string includes the other
    string.

    :param a:
    :param b:
    :return:
    """
    if a in b or b in a:
        return 0.1  # Boost by 10%
    return 0.0


def normalized_similarity(a: str, b: str) -> float:
    """
    Returns a normalized similarity ratio result from the hybrid union of levenshtein distance and the
    'difflib.SequenceMatches' built-in class.

    :param a:
    :param b:
    :return:
    """
    base_score = SequenceMatcher(None, a, b).ratio()
    score_boost = string_inclusion_boost(a, b)
    return min(1.0, base_score + score_boost)  # Average of the two similarity ratios.


def get_dynamic_threshold(length1: int, length2: int, min_th: float = 0.6, max_th: float = 0.9) -> float:
    avg_len = (length1 + length2) / 2
    len_diff = abs(length1 - length2)

    # Threshold decay based on avg length (longer strings tolerate more differences)
    threshold = 0.8 - 0.02 * (avg_len ** 0.5)

    # Penalize large differences in length
    threshold -= 0.05 * (len_diff / max(length1, length2)) if any([length1, length2]) else 0.0

    # Clamp to min/max
    threshold = max(min_th, min(max_th, threshold))
    return threshold


def is_a_match(string1: str, string2: str):
    dynamic_threshold = get_dynamic_threshold(len(string1), len(string2))
    similarity_ratio = normalized_similarity(string1, string2)

    is_match = similarity_ratio > dynamic_threshold

    # print(f"['{string1}', '{string2}']: "
    #       f"Similarity = {similarity_ratio:.4f}, "
    #       f"Threshold = {dynamic_threshold:.4f} -> "
    #       f"{'MATCH' if is_match else 'NO MATCH'}")
    return is_match
