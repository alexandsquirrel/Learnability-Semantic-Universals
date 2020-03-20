
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

from __future__ import division
from __future__ import print_function

from builtins import zip
from builtins import str
from builtins import range
from builtins import object
import itertools
import math
import os
import time

from collections import defaultdict

import numpy as np

import quantifiers

# TODO: move batching logic from quant_verify.run_experiment to here?
# TODO: roll-back the writing to files logic?


class DataGenerator(object):

    # TODO: document; mode = r, w, g [generate]; remove r, w?
    def __init__(self, max_len, quants=quantifiers.get_all_quantifiers(),
                 training_split=0.7, mode='g', file_path='/tmp/quantexp/data/',
                 bin_size=1e6, num_data_points=100000):

        self._max_len = max_len
        self._quantifiers = quants
        self._num_quants = len(quants)
        self._quant_labels = np.identity(self._num_quants)
        self._training_split = training_split
        self._training_data = None
        self._test_data = None

        if mode == 'g':
            self._labeled_data = self._generate_labeled_data(num_data_points)
        elif mode == 'w':
            self.write_labeled_data(file_path, bin_size)
        elif mode == 'r':
            pass
        else:
            raise ValueError("mode must be one of g, w, r")

    def _generate_sequences(self):
        """Generates (sequence, quantifier_index) pairs for all sequences
        up to length max_len.
        These correspond to finite models.

        Args:
            max_len: the maximum length of a sequence (aka size of a model)

        Returns:
            a generator, generating all relevant pairs
        """

        num_quants = len(self._quantifiers)
        num_chars = quantifiers.Quantifier.num_chars

        all_gens = []
        for n in range(1, self._max_len + 1):
            seqs = itertools.product(range(num_chars), repeat=n)
            data_n = ((seq, quant) for seq in seqs
                      for quant in range(num_quants))
            all_gens.append(data_n)

        return itertools.chain(*all_gens)

    def _generate_random_tuple(self):
        """Generates a random tuple corresponding to an input example.

        Returns:
            a pair seq, quant, where seq is a random sequence of characters
            of a random length up to self._max_len and quant is a random
            integer up to self._num_quants
        """
        quant = np.random.randint(self._num_quants)
        length = np.random.randint(1, self._max_len+1)
        seq = tuple((np.random.randint(quantifiers.Quantifier.num_chars)
                     for _ in range(length)))
        return seq, quant

    def _tuple_to_idx(self, tup):
        # TODO: document better
        """Takes a tuple of (seq, quant) and generates its index in the
        lexicographic order of all such sequences.

        This could probably be done more cleanly, but it works.

        Args:
            tup: a tuple of a sequence and a quantifier index,
                    as generated by _generate_sequences()

        Returns:
            an integer, corresponding to this tuple's position
            in lexicographic order
        """
        seq, quant = tup
        upper_bound = len(seq) - 1
        return (quant +
                self._num_quants *
                sum(seq[i] *
                    (quantifiers.Quantifier.num_chars ** (upper_bound - i))
                    for i in range(len(seq))) +
                self._num_quants * sum(4**i for i in range(1, len(seq))))

    def _point_from_tuple(self, tup):
        """Generates a labeled data point from a tuple generated by
        _generate_sequences.
        To do so, it converts character indices into one-hot vectors,
        pads the length to _max_len, and augments each character with
        the one-hot vector corresponding to the quantifier.
        It then runs the quantifier on the sequence and outputs
        the generated label as well.

        Args:
            tup: a pair, the first element of which is a tuple of
                    elements of range(num_chars),
                    the second element of which is an element of
                    range(num_quants)

        Returns:
            a pair, the first element of which is a max_len
            length tuple of numpy arrays of length num_chars + num_quants,
            corresponding to the characters in the sequence, the second
            element of which is a label, generated by running the quantifier
            on the input sequence.
        """

        char_seq, quant_idx = tup

        chars = tuple(quantifiers.Quantifier.chars[idx] for idx in char_seq)
        padded_seq = (chars +
                      (quantifiers.Quantifier.zero_char,)
                      * (self._max_len - len(chars)))
        padded_with_quant = tuple(
            np.concatenate([char, self._quant_labels[quant_idx]])
            for char in padded_seq)
        label = self._quantifiers[quant_idx](chars)

        return padded_with_quant, label

    def _generate_labeled_data(self, num_data_points, balanced=True):
        """Generates a complete list of labeled data.  Iterates through
        _generate_sequences, calling _point_from_tuple on each tuple generated.
        At the end, the list is shuffled so that the data is in random order.
        Note that this returns the entire dataset, not split into train/test.

        Args:
            num_data_points: maximum possible data points to generate from
            balanced: Boolean; if true, under-samples from dominant truth-value
                        for each quantifier, so that data is balanced for each
                        value by quantifier

        Returns:
            a list of all labeled data, in random order.
        """

        self._labeled_data = []
        total_possible = self._num_quants*sum(
            quantifiers.Quantifier.num_chars**i
            for i in range(1, self._max_len+1))

        # if the total possible data pool is smaller than requested,
        # just generate all of it
        if total_possible <= num_data_points:
            print('generating all')
            for tup in self._generate_sequences():
                self._labeled_data.append(
                    self._point_from_tuple(tup))
        else:
            # otherwise, generate num_data_points randomly
            # store which data points have already been generated
            # generated_idxs = bitarray(total_possible)
            generated_idxs = set()
            to_generate = min(total_possible, num_data_points)
            # tups: a dictionary, keys: (quant_idx, label) pairs
            # values: sequences.  Will be used for balancing data
            tups = defaultdict(list)

            while to_generate > 0:
                # generate random tuple
                tup = self._generate_random_tuple()
                tup_idx = self._tuple_to_idx(tup)
                # have not generated this data point yet, so add it
                if tup_idx not in generated_idxs:
                    generated_idxs.add(tup_idx)
                    to_generate -= 1
                    seq, label = self._point_from_tuple(tup)
                    if balanced:
                        tups[(tup[1], label)].append(seq)
                    else:
                        self._labeled_data.append((seq, label))

            if balanced:
                # balance across (Q, T/F), instead of just T/F
                num_to_sample = min([len(tups[k]) for k in tups])
                for (qidx, label) in tups:
                    # randomly sample right number of sequences
                    idxs = np.random.choice(len(tups[(qidx, label)]),
                                            num_to_sample,
                                            replace=False)
                    # add to data
                    for idx in np.nditer(idxs):
                        seq = tups[(qidx, label)][idx]
                        self._labeled_data.append(
                            (seq, label))

        np.random.shuffle(self._labeled_data)
        return self._labeled_data

    def get_training_data(self):
        """Gets training data, based on the percentage self._training_split.
        Shuffles the training data every time it is called.
        Must be called only after _generate_labeled_data has been.
        """

        if self._training_data is None:
            idx = int(math.ceil(
                self._training_split * len(self._labeled_data)))
            self._training_data = self._labeled_data[:idx]

        np.random.shuffle(self._training_data)
        return self._training_data

    def get_test_data(self):
        """Gets test data, based on the percentage 1 - self._training_split.
        Must be called only after _generate_labeled_data has been.
        """

        if self._test_data is None:
            idx = int(math.ceil(
                self._training_split * len(self._labeled_data)))
            self._test_data = self._labeled_data[idx:]

        return self._test_data

    def write_labeled_data(self, file_path, num_files=256):

        split = self._training_split

        num_train_bins = max(1, int(split*num_files))
        num_test_bins = max(1, int((1-split)*num_files))

        train_input_filenames = ['{}train_input_{}.txt'.format(file_path, idx)
                                 for idx in range(num_train_bins)]
        train_label_filenames = ['{}train_labels_{}.txt'.format(file_path, idx)
                                 for idx in range(num_train_bins)]
        test_input_filenames = ['{}test_input_{}.txt'.format(file_path, idx)
                                for idx in range(num_train_bins)]
        test_label_filenames = ['{}test_labels_{}.txt'.format(file_path, idx)
                                for idx in range(num_train_bins)]

        train_input_files = [open(fn, 'w+') for fn in train_input_filenames]
        train_label_files = [open(fn, 'w+') for fn in train_label_filenames]
        test_input_files = [open(fn, 'w+') for fn in test_input_filenames]
        test_label_files = [open(fn, 'w+') for fn in test_label_filenames]

        t0 = time.time()
        print('files opened...')

        for tup in self._generate_sequences():

            eg_input, eg_label = self._point_from_tuple(tup)
            if np.random.random() < split:
                # training example
                train_idx = np.random.randint(num_train_bins)
                train_input_files[train_idx].write(
                    self._input_to_str(eg_input) + '\n')
                train_label_files[train_idx].write(
                    self._label_to_str(eg_label) + '\n')
            else:
                # test example
                test_idx = np.random.randint(num_test_bins)
                test_input_files[test_idx].write(
                    self._input_to_str(eg_input) + '\n')
                test_label_files[test_idx].write(
                    self._label_to_str(eg_label) + '\n')

        t1 = time.time()
        print('initial loop took: {} seconds'.format(t1-t0))

        # make sure all the data has been written, move buffers back to start
        for f in (train_input_files + train_label_files +
                  test_input_files + test_label_files):
            f.flush()
            os.fsync(f)
            f.seek(0)

        t2 = time.time()
        print('randomizing each file')
        # randomize each file
        for infile, label_file in (list(zip(train_input_files, train_label_files))
                                   + list(zip(test_input_files, test_label_files))):
            inputs = infile.readlines()
            labels = label_file.readlines()
            assert len(inputs) == len(labels)
            idxs = np.arange(len(inputs))
            np.random.shuffle(idxs)
            infile.seek(0)
            label_file.seek(0)
            for i in idxs:
                infile.write(inputs[i])
                label_file.write(labels[i])
            # now, close for good
            infile.close()
            label_file.close()
        t3 = time.time()

        print('randomization took: {} seconds'.format(t3-t2))
        print('total time to write data: {} seconds'.format(t3-t0))

    def _input_to_str(self, seq):
        """Generates string for a nested list, corresponding to one input
        for the model.

        Args:
            seq: a sequence, corresponding to a model tagged with a quantifier

        Returns:
            a string, with tab-separated sub-items,
            each item being space separated
        """
        return '\t'.join(' '.join(str(item) for item in ls) for ls in seq)

    def _str_to_input(self, string):
        return tuple([np.array(item.split(' '), dtype=float)
                      for item in string.split('\t')])

    def _label_to_str(self, label):
        return '\t'.join(str(i) for i in label)

    def _str_to_label(self, string):
        return tuple([int(i) for i in string.split('\t')])