import random
import numpy as np


def random_sequence(min_len, max_len, legal_chars):
    len = np.random.randint(min_len, max_len + 1)
    seq = [random.choice(legal_chars) for _ in range(len)]
    return seq


def random_substitute(seq, desired_chars, num_substitutions):
    positions = random.sample(range(len(seq)), num_substitutions)
    for pos in positions:
        new_char = random.choice(desired_chars)
        seq[pos] = new_char

def quantifier_not(truth_value):
    T = (1, 0)
    F = (0, 1)
    return T if truth_value == F else F


def reverse_verify_fn(verify_fn):
    return lambda seq: quantifier_not(verify_fn(seq))