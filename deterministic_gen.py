import itertools
import math

from quantifiers import *
import numpy as np
from data_gen import DataGenerator


class DeterministicGenerator(DataGenerator):

    """ Generates a random, specific (sequence, quantifier_index) tuple
    that guarantees a truth value evaluated under the specified quantifier.

    Args:
        quant_idx: the index of quantifier to be evaluated under.
        truth_val: the desired truth value of the generated sequence.
        seq_length: the maximum length of the sequence.

    Returns:
        a pair seq, quant_idx, where seq is a random sequence of characters
        of a random length up to seq_length and quant_idx is a random
        integer up to self._num_quants.
    """
    def _generate_specific_tuple(self, quant_idx, truth_val, seq_length):
        quant = self._quantifiers[quant_idx]
        seq = quant.generateExample(truth_val, seq_length)
        return seq, quant_idx


    """ Generates labled data with a specific number of data points per bucket.
    Each bucket may contain duplicates, but is unlikely to contain many.
    
    Args:
        num_data_points_per_bucket: number of data points per bucket.
        
    Returns:
        The generated data points in the same format as DataGenerator._generate_labeled_data.
    """
    def _generate_labled_data_by_bucket(self, num_data_points_per_bucket):
        self._labeled_data = []
        for (quant, truth_val) in itertools.product(range(self._num_quants), [Quantifier.T, Quantifier.F]):
            print("Generating ", self._quantifiers[quant]._name, " ", truth_val)
            for idx in range(num_data_points_per_bucket):
                tup = self._generate_specific_tuple(quant, truth_val, self._max_len)
                data_point = self._point_from_tuple(tup)
                self._labeled_data.append(data_point)

        np.random.shuffle(self._labeled_data)
        return self._labeled_data


    """ Similar functionality as _generate_labled_data_by_bucket, but only
    specify the total number of data points but not per-bucket number.
    The parameter "balanced" is useless; it's just for overriding.
    """
    def _generate_labeled_data(self, num_data_points, balanced=True):
        total_possible = self._num_quants * sum(
            Quantifier.num_chars ** i
            for i in range(1, self._max_len + 1))
        if total_possible <= num_data_points:
            return super()._generate_labeled_data(num_data_points, balanced=True)
        else:
            num_buckets = self._num_quants * 2
            points_per_bucket = math.ceil(num_data_points / num_buckets)
            print(num_data_points)
            print("Num of buckets: ", num_buckets, ", points per bucket: ", points_per_bucket)
            return self._generate_labled_data_by_bucket(points_per_bucket)
