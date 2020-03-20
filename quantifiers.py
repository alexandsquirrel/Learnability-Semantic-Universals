"""
Copyright (C) 2017 Shane Steinert-Threlkeld

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>
"""

from builtins import object
import random

import numpy as np
import gc
from quantifiers_util import *

""" A generic, trail-and-error-based generator. Works for any
quantifier if the verification function works properly.
"""
def generic_generator(verify_fn, truth_value, max_length):
    CUTOFF = 20000
    counter = 0
    while True:
        counter += 1
        seq = random_sequence(1, max_length, Quantifier.legal_ids)
        chars = tuple(Quantifier.chars[idx] for idx in seq)
        if verify_fn(chars) == truth_value:
            return seq
        if counter > CUTOFF:
            counter = 0
            seq = random_sequence(1, max_length, Quantifier.legal_ids)


""" Verifies whether the seq evaluates to truth_value
under verify_fn. Raise an error if not.
"""
def verify_good_example(verify_fn, seq, truth_value):
    DEBUG = False
    if DEBUG:
        assert verify_fn(tuple(Quantifier.chars[idx] for idx in seq)) == truth_value


class Quantifier(object):

    # 4 chars: A cap B, A - B, B - A, M - (A cup B)
    num_chars = 4
    # chars = one-hot encoding
    chars = np.identity(num_chars)
    # zero char for padding
    zero_char = np.zeros(num_chars)

    # name the characters, for readability
    AB = chars[0]
    AnotB = chars[1]
    BnotA = chars[2]
    neither = chars[3]

    AB_id = 0
    AnotB_id = 1
    BnotA_id = 2
    neither_id = 3
    legal_ids = [0, 1, 2, 3]

    T = (1, 0)
    F = (0, 1)

    # TODO: other properties of quantifiers?

    def __init__(self, name, isom=True, cons=True, lcons=False,
            rmon=True, lmon=None, fn=None, gen_fn=generic_generator):

        if fn is None:
            raise ValueError("supply a function for verifying a quantifier!")

        if gen_fn is generic_generator:
            print("Warning: using generic generator:", name)

        self._name = name
        self._isom = isom
        self._cons = cons
        self._lcons = lcons
        self._rmon = rmon
        self._lmon = lmon
        self._verify = fn
        self._generate = gen_fn

    def __call__(self, seq):
        return self._verify(seq)
    
    def generateExample(self, truth_value, max_length):
        return self._generate(self._verify, truth_value, max_length + 1)



def all_ver(seq):
    """Verifies whether every A is a B in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there are no Quantifier.AnotBs in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.AnotB):
            return Quantifier.F
    return Quantifier.T


def all_gen(verify_fn, truth_value, max_length):
    # Note: verify_fn is useless :)
    seq = random_sequence(1, max_length, [Quantifier.AB_id, Quantifier.BnotA_id, Quantifier.neither_id])
    if truth_value == Quantifier.F:
        num_falsifying_char = np.random.randint(1, len(seq) / 2 + 2)
        random_substitute(seq, [Quantifier.AnotB_id], num_falsifying_char)
    verify_good_example(verify_fn, seq, truth_value)
    return seq


every = Quantifier("every",
        isom=True, cons=True, lcons=False, rmon=True, lmon=False,
        fn=all_ver, gen_fn=all_gen)


def notall_ver(seq):
    """Verifies whether not all As are Bs in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there is a Quantifier.AnotB in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.AnotB):
            return Quantifier.T
    return Quantifier.F


def notall_gen(verify_fn, truth_value, max_length):
    return all_gen(reverse_verify_fn(verify_fn), quantifier_not(truth_value), max_length)


nall = Quantifier("not_all",
        isom=True, cons=True, lcons=False, rmon=False, lmon=True,
        fn=notall_ver, gen_fn=notall_gen)


def no_ver(seq):
    """Verifies whether no As are Bs in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there is not a Quantifier.AB in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            return Quantifier.F
    return Quantifier.T


def no_gen(verify_fn, truth_value, max_length):
    seq = random_sequence(1, max_length, [Quantifier.AnotB_id, Quantifier.BnotA_id, Quantifier.neither_id])
    if truth_value == Quantifier.F:
        num_AB = np.random.randint(1, (len(seq) + 1) / 2 + 1)
        random_substitute(seq, [Quantifier.AB_id], num_AB)
    verify_good_example(verify_fn, seq, truth_value)
    return seq


no = Quantifier("no",
        isom=True, cons=True, lcons=False, rmon=False, lmon=False,
        fn=no_ver, gen_fn=no_gen)


def only_ver(seq):
    """Verifies whether only As are Bs in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there are no Quantifier.BnotAs in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.BnotA):
            return Quantifier.F
    return Quantifier.T


def only_gen(verify_fn, truth_value, max_length):
    seq = random_sequence(1, max_length, [Quantifier.AB_id, Quantifier.AnotB_id, Quantifier.neither_id])
    if truth_value == Quantifier.F:
        num_BnotA = np.random.randint(1, len(seq) / 2 + 2)
        random_substitute(seq, [Quantifier.BnotA_id], num_BnotA)
    verify_good_example(verify_fn, seq, truth_value)
    return seq


only = Quantifier("only",
        isom=True, cons=False, lcons=True, rmon=False, lmon=True,
        fn=only_ver, gen_fn=only_gen)


def notonly_ver(seq):
    """Verifies whether not only As are Bs in a sequence.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff there is a Quantifier.BnotA in seq
    """
    for item in seq:
        if np.array_equal(item, Quantifier.BnotA):
            return Quantifier.T
    return Quantifier.F


def notonly_gen(verify_fn, truth_value, max_length):
    return only_gen(reverse_verify_fn(verify_fn), quantifier_not(truth_value), max_length)


notonly = Quantifier("not_only",
        isom=True, cons=False, lcons=True, rmon=True, lmon=False,
        fn=notonly_ver, gen_fn=notonly_gen)


def even_ver(seq):
    """Verifies whether the number of As that are B is even.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff the number of Quantifier.ABs in seq is even
    """
    num_AB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            num_AB += 1
    if num_AB % 2 == 0:
        return Quantifier.T
    else:
        return Quantifier.F


def even_gen(verify_fn, truth_value, max_length):
    seq = random_sequence(3, max_length, [Quantifier.AnotB_id, Quantifier.BnotA_id, Quantifier.neither_id])
    num_half_AB = np.random.randint(0, (len(seq) - 1) / 2)
    num_AB = num_half_AB * 2 if truth_value == Quantifier.T else num_half_AB * 2 + 1
    random_substitute(seq, [Quantifier.AB_id], num_AB)
    verify_good_example(verify_fn, seq, truth_value)
    return seq


even = Quantifier("even",
        isom=True, cons=True, lcons=True, rmon=None, lmon=None,
        fn=even_ver, gen_fn=even_gen)


def odd_ver(seq):
    """Verifies whether the number of As that are B is odd.

    Args:
        seq: a sequence of elements of R^4

    Returns:
        Quantifier.T iff the number of Quantifier.ABs in seq is odd
    """
    return Quantifier.T if even_ver(seq) == Quantifier.F else Quantifier.F


def odd_gen(verify_fn, truth_value, max_length):
    return even_gen(reverse_verify_fn(verify_fn), quantifier_not(truth_value), max_length)


odd = Quantifier("odd",
        isom=True, cons=True, lcons=True, rmon=None, lmon=None,
        fn=odd_ver, gen_fn=odd_gen)


def at_least_n_ver(seq, n):
    """Verifies whether |A cap B| > n.

    Args:
        seq: a sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff at least n elements are Quantifier.AB
    """
    num_AB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            if num_AB == n-1:
                return Quantifier.T
            else:
                num_AB += 1
    return Quantifier.F


def at_least_n_gen(n, verify_fn, truth_value, max_length):
    seq = random_sequence(n, max_length, [Quantifier.AnotB_id, Quantifier.BnotA_id, Quantifier.neither_id])
    num = np.random.randint(n, len(seq) + 1) if truth_value == Quantifier.T else np.random.randint(0, n)
    random_substitute(seq, [Quantifier.AB_id], num)
    verify_good_example(verify_fn, seq, truth_value)
    return seq


def at_least_n(n):
    """Generates a Quantifier corresponding to at least n.

    Args:
        n: integer

    Returns:
        Quantifier, with at_least_n_ver(_, n) as its verifier
    """
    return Quantifier("at_least_{}".format(n),
            isom=True, cons=True, lcons=True, rmon=True, lmon=True,
            fn=lambda seq: at_least_n_ver(seq, n),
            gen_fn=lambda verify_fn, truth_value, max_length: at_least_n_gen(n, verify_fn, truth_value, max_length))


some = at_least_n(1)
at_least_three = at_least_n(3)


def at_most_n_ver(seq, n):
    """Verifies whether |A cap B| <= n.

    Args:
        seq: a sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff exactly n elements are Quantifier.AB
    """
    num_AB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            if num_AB == n:
                return Quantifier.F
            else:
                num_AB += 1
    return Quantifier.T


def at_most_n_gen(n, verify_fn, truth_value, max_length):
    return at_least_n_gen(n+1, reverse_verify_fn(verify_fn), quantifier_not(truth_value), max_length)


def at_most_n(n):
    """Generates a Quantifier corresponding to at most n.

    Args:
        n: integer

    Returns:
        Quantifier, with at_most_n_ver(_, n) as its verifier
    """
    return Quantifier("at_most_{}".format(n),
            isom=True, cons=True, lcons=True, rmon=False, lmon=False,
            fn=lambda seq: at_most_n_ver(seq, n),
            gen_fn=lambda verify_fn, truth_value, max_length: at_most_n_gen(n, verify_fn, truth_value, max_length))


def exactly_n_ver(seq, n):
    """Verifies whether |A cap B| = n.

    Args:
        seq: a sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff exactly n elements are Quantifier.AB
    """
    num_AB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            num_AB += 1
    return Quantifier.T if num_AB == n else Quantifier.F


def exactly_n_gen(n, verify_fn, truth_value, max_length):
    min_length = 1 if truth_value == Quantifier.F else n
    seq = random_sequence(min_length, max_length,
                          [Quantifier.AnotB_id, Quantifier.BnotA_id, Quantifier.neither_id])
    if truth_value == Quantifier.T:
        num_AB = n
    else:
        num_AB = np.random.randint(1, len(seq) + 1)
        if num_AB == n:
            num_AB = 0
    random_substitute(seq, [Quantifier.AB_id], num_AB)
    verify_good_example(verify_fn, seq, truth_value)
    return seq


def exactly_n(n):
    """Generates a Quantifier corresponding to at least n.

    Args:
        n: integer

    Returns:
        Quantifier, with exactly_n_ver(_, n) as its verifier
    """
    return Quantifier("exactly_{}".format(n),
            isom=True, cons=True, lcons=True, rmon=None, lmon=None,
            fn=lambda seq: exactly_n_ver(seq, n),
            gen_fn=lambda verify_fn, truth_value, max_length: exactly_n_gen(n, verify_fn, truth_value, max_length))


exactly_three = exactly_n(3)


def between_m_and_n_ver(seq, m, n):
    """Verifies whether m <= |A cap B| <= n.

    Args:
        seq: a sequence of elements from R^4
        m: an integer
        n: an integer

    Returns:
        Quantifier.T iff between m and n elements are Quantifier.AB
    """
    num_AB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            num_AB += 1
    return Quantifier.T if (m <= num_AB and num_AB <= n) else Quantifier.F


def between_m_and_n_gen(m, n, verify_fn, truth_value, max_length):
    lb =  m if truth_value == Quantifier.T else 1
    seq = random_sequence(lb, max_length, [Quantifier.AnotB_id, Quantifier.BnotA_id, Quantifier.neither_id])
    if truth_value == Quantifier.T:
        num_sub = np.random.randint(m, min(len(seq)+1, n))
    elif len(seq) <= n:
        num_sub = np.random.randint(0, min(len(seq), m))
    else:
        if np.random.randint(0, m + len(seq) - n) < m:
            num_sub = np.random.randint(0, m)
        else:
            num_sub = np.random.randint(n+1, len(seq)+1)
    random_substitute(seq, [Quantifier.AB_id], num_sub)
    verify_good_example(verify_fn, seq, truth_value)
    return seq


def between_m_and_n(m, n):
    return Quantifier("between_{}_and_{}".format(m,n),
            isom=True, cons=True, lcons=True, rmon=None, lmon=None,
            fn=lambda seq: between_m_and_n_ver(seq, m, n),
            gen_fn=lambda verify_fn, truth_value, max_length: between_m_and_n_gen(m, n, verify_fn, truth_value, max_length))


def all_but_n_ver(seq, n):
    """Verifies whether |A - B| = 4.

    Args:
        seq: a sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff exactly n elements are Quantifier.AnotB
    """
    num_AnotB = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AnotB):
            num_AnotB += 1
    return Quantifier.T if num_AnotB == n else Quantifier.F


def all_but_n(n):
    """Generates a Quantifier corresponding to all but n.

    Args:
        n: integer

    Returns:
        Quantifier, with all_but_n_ver(_, n) as its verifier
    """
    return Quantifier("all_but_{}".format(n),
            isom=True, cons=True, lcons=False, rmon=None, lmon=None,
            fn=lambda seq: all_but_n_ver(seq, n))


def most_ver(seq):
    """Verifies whether |A cap B| > |A - B|

    Args:
        seq: a sequence of elements from R^4

    Returns:
        Quantifier.T iff more elements are Quantifier.AB than are
        Quantifier.AnotB
    """
    diff = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            diff += 1
        elif np.array_equal(item, Quantifier.AnotB):
            diff -= 1
    return Quantifier.T if diff > 0 else Quantifier.F


def most_gen(verify_fn, truth_value, max_length):
    seq = random_sequence(1, max_length, [Quantifier.BnotA_id, Quantifier.neither_id])
    num_A = np.random.randint(1, len(seq) + 1)
    if truth_value == Quantifier.T:
        num_AB = np.random.randint(num_A / 2 + 1, num_A + 1)
        num_AnotB = num_A - num_AB
    else:
        num_AnotB = np.random.randint(num_A / 2 + 1, num_A + 1)
        num_AB = num_A - num_AnotB
    random_substitute(seq, [Quantifier.AB_id], num_AB)
    # Substitute Quantifier.BnotA_id, Quantifier.neither_id with Quantifier.AnotB_id
    AnotB_places = [idx for idx in range(len(seq))\
                    if seq[idx] == Quantifier.BnotA_id or seq[idx] == Quantifier.neither_id]
    sub_idx = random.sample(range(len(AnotB_places)), num_AnotB)
    for idx in sub_idx:
        place = AnotB_places[idx]
        seq[place] = Quantifier.AnotB_id
    verify_good_example(verify_fn, seq, truth_value)
    return seq


most = Quantifier("most",
        isom=True, cons=True, lcons=False, rmon=True, lmon=None,
        fn=most_ver, gen_fn=most_gen)


def M_ver(seq):
    """Verifies whether |A| > |B|, i.e. M from Keenan and Westerstahl 1997

    Args:
        seq: a sequence of elements from R^4

    Returns:
        Quantifier.T iff more elements are AB and AnotB together than are
        AB and BnotA
    """
    num_AB = 0
    num_AnotB = 0
    num_BnotA = 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            num_AB += 1
        if np.array_equal(item, Quantifier.AnotB):
            num_AnotB += 1
        if np.array_equal(item, Quantifier.BnotA):
            num_BnotA += 1
    return (Quantifier.T if num_AB + num_AnotB > num_AB + num_BnotA
            else Quantifier.F)


def M_gen(verify_fn, truth_value, max_length):
    length = np.random.randint(1, max_length)
    if truth_value == Quantifier.T:
        num_A = np.random.randint(1, length + 1)
        num_B = np.random.randint(0, num_A)
    else:
        num_B = np.random.randint(1, length + 1)
        num_A = np.random.randint(0, num_B)
    is_A = [False] * length
    random_substitute(is_A, [True], num_A)
    is_B = [False] * length
    random_substitute(is_B, [True], num_B)
    seq = []
    for idx in range(length):
        if is_A[idx] and is_B[idx]:
            seq.append(Quantifier.AB_id)
        elif is_A[idx] and not is_B[idx]:
            seq.append(Quantifier.AnotB_id)
        elif not is_A[idx] and is_B[idx]:
            seq.append(Quantifier.BnotA_id)
        else:
            seq.append(Quantifier.neither_id)
    verify_good_example(verify_fn, seq, truth_value)
    return seq


M = Quantifier("M",
        isom=True, cons=False, lcons=False, lmon=True, rmon=False,
        fn=M_ver, gen_fn=M_gen)


def first_n_ver(seq, n):
    """Verifies whether the first n As are also Bs.

    Args:
        seq: sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff the first three elements of seq that are either
        Quantifier.AB or Quantifier.AnotB are in fact Quantifier.AB.
        Quantifier.F if either seq has length less than n or there are
        fewer than n Quantifier.ABs in seq.
    """
    # TODO: more complicated presupposition handling instead of just false?
    if len(seq) < n:
        return Quantifier.F

    num_AB = 0
    for item in seq:
        if num_AB >= n:
            return Quantifier.T
        # if an A-not-B found before n ABs are, return F
        if np.array_equal(item, Quantifier.AnotB) and num_AB < n:
            return Quantifier.F
        elif np.array_equal(item, Quantifier.AB):
            num_AB += 1

    if num_AB >= n:
        return Quantifier.T

    # there are less than n ABs in total
    return Quantifier.F


def first_n_gen(n, verify_fn, truth_value, max_length):
    # First, generate the effective portion.
    effective_portion_len = np.random.randint(n - 1, max(max_length / 2, n - 1))
    effective_portion = random_sequence(effective_portion_len, effective_portion_len,
                                        [Quantifier.BnotA_id, Quantifier.neither_id])
    random_substitute(effective_portion, [Quantifier.AB_id], n - 1)
    if truth_value == Quantifier.F:
        num_AnotB = np.random.randint(1, effective_portion_len / 4 + 2)
        random_substitute(effective_portion, [Quantifier.AnotB_id], num_AnotB)

    effective_portion += [Quantifier.AB_id]

    padded_portion_len = np.random.randint(0, max_length - effective_portion_len)
    padded_portion = random_sequence(padded_portion_len, padded_portion_len,
                                     Quantifier.legal_ids)

    verify_good_example(verify_fn, effective_portion + padded_portion, truth_value)
    return effective_portion + padded_portion



def first_n(n):
    """Generates a Quantifier corresponding to `the first n'.

    Args:
        n: integer

    Returns:
        a Quantifier, with first_n_ver(_, n) as its verifier
    """
    return Quantifier("first_{}".format(n),
            isom=False, cons=True, lcons=False, rmon=True, lmon=None,
            fn=lambda seq: first_n_ver(seq, n),
            gen_fn=lambda verify_fn, truth_value, max_length: first_n_gen(n, verify_fn, truth_value, max_length))


first_three = first_n(3)


# TODO: document!
def last_n_ver(seq, n):
    """Verifies whether the last n As are also Bs.

    Args:
        seq: sequence of elements from R^4
        n: an integer

    Returns:
        Quantifier.T iff the final three elements of seq that are either
        Quantifier.AB or Quantifier.AnotB are in fact Quantifier.AB.
        Quantifier.F if either seq has length less than n or there are
        fewer than n Quantifier.ABs in seq.
    """
    return first_n_ver(list(reversed(seq)), n)


def last_n_gen(n, verify_fn, truth_value, max_length):
    return list(reversed(first_n_gen(n, lambda seq: first_n_ver(seq, n), truth_value, max_length)))


def last_n(n):
    """Generates a Quantifier corresponding to `the last n'.

    Args:
        n: integer

    Returns:
        a Quantifier, with last_n_ver(_, n) as its verifier
    """
    return Quantifier("last_{}".format(n),
            isom=False, cons=True, lcons=False, rmon=True, lmon=None,
            fn=lambda seq: last_n_ver(seq, n),
            gen_fn=lambda verify_fn, truth_value, max_length: last_n_gen(n, verify_fn, truth_value, max_length))


def equal_number_ver(seq):
    """Generates a Quantifier corresponding to
    `the number of As equals the number of Bs'.

    Args:
        seq: sequence of elts of R^4

    Returns:
        Quantifier.T iff the number of Quantifier.ABs plus Quantifier.AnotBs is
        the same as the number of Quanitifer.BnotAs plus Quantifier.ABs
    """
    num_AB, num_AnotB, num_BnotA = 0, 0, 0
    for item in seq:
        if np.array_equal(item, Quantifier.AB):
            num_AB += 1
        elif np.array_equal(item, Quantifier.AnotB):
            num_AnotB += 1
        elif np.array_equal(item, Quantifier.BnotA):
            num_BnotA += 1
    return Quantifier.T if num_AnotB == num_BnotA else Quantifier.F


def equal_number_gen(verify_fn, truth_value, max_length):
    length = np.random.randint(1, max_length)
    num_A = np.random.randint(0, length + 1)
    if truth_value == Quantifier.T:
        num_B = num_A
    else:
        num_B = np.random.randint(1, length + 1)
        if num_B == num_A:
            num_B = 0
    is_A = [False] * length
    random_substitute(is_A, [True], num_A)
    is_B = [False] * length
    random_substitute(is_B, [True], num_B)
    seq = []
    for idx in range(length):
        if is_A[idx] and is_B[idx]:
            seq.append(Quantifier.AB_id)
        elif is_A[idx] and not is_B[idx]:
            seq.append(Quantifier.AnotB_id)
        elif not is_A[idx] and is_B[idx]:
            seq.append(Quantifier.BnotA_id)
        else:
            seq.append(Quantifier.neither_id)
    verify_good_example(verify_fn, seq, truth_value)
    return seq


equal_number = Quantifier("equal_number",
        isom=True, cons=False, lcons=False, rmon=None, lmon=None,
        fn=equal_number_ver, gen_fn=equal_number_gen)


def or_ver(q1, q2, seq):

    if q1(seq) == Quantifier.T or q2(seq) == Quantifier.T:
        return Quantifier.T
    return Quantifier.F


def at_least_n_or_at_most_m_gen(n, m, verify_fn, truth_value, max_length):
    if truth_value == Quantifier.T:
        if np.random.randint(0, 2) == 0:
            return at_least_n_gen(n, verify_fn, Quantifier.T, max_length)
        else:
            return at_most_n_gen(m, verify_fn, Quantifier.T, max_length)
    else:
        assert m < n  # otherwise, there are no counterexamples
        return between_m_and_n_gen(m+1, n-1, reverse_verify_fn(verify_fn), Quantifier.T, max_length)


def at_least_n_or_at_most_m(n, m):
    return Quantifier("at_least_{}_or_at_most_{}".format(n, m),
            isom=True, cons=True, lcons=False, rmon=False, lmon=False,
            fn = lambda seq: or_ver(
                lambda seq: at_least_n_ver(seq, n),
                lambda seq: at_most_n_ver(seq, m), seq),
            gen_fn=lambda verify_fn, truth_value, max_length: at_least_n_or_at_most_m_gen(n, m, verify_fn, truth_value, max_length))


def get_all_quantifiers():
    """Returns: a list of all Quantifiers that have been created so far.
    """
    return [quant for quant in gc.get_objects()
            if isinstance(quant, Quantifier)]


def get_nonparity_quantifiers():

    quants = get_all_quantifiers()
    quants.remove(even)
    quants.remove(odd)
    return quants
