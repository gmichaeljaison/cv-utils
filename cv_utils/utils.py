from __future__ import division

import os

import theano

from theano import tensor
from theano.tensor.signal import downsample as ds


def remove_missing_keys(src_dict, target_dict):
    """
    Remove keys from target_dict that are not available in src_dict

    :param src_dict: source dictionary to search for
    :param target_dict: target dictionary where keys should be removed
    """
    for key in list(target_dict.keys()):
        if key not in src_dict:
            target_dict.pop(key)


def max_pooling(matrix, pool_size):
    """
    Applies max-pooling for the given matrix for specified pool_size.
        Only the maximum value in the given pool size is chosen to construct the result.

    :param matrix: Input matrix
    :param pool_size: pooling cell size
    :return: max-pooled output
    """
    t_input = tensor.dmatrix('input')

    pool_out = ds.max_pool_2d(t_input, pool_size, ignore_border=True)
    pool_f = theano.function([t_input], pool_out)

    return pool_f(matrix)


def min_pooling(matrix, pool_size):
    """ Applies min-pooling by negating the input matrix and using max-pooling function """
    matrix *= -1
    return max_pooling(matrix, pool_size)


def create_dirs(path):
    """
    Creates all directories mentioned in the given path.
        Useful to write a new file with the specified path.
        It carefully skips the file-name in the given path.

    :param path: Path of a file or directory
    """
    fname = os.path.basename(path)

    # if file name exists in path, skip the filename
    if fname.__contains__('.'):
        path = os.path.dirname(path)

    if not os.path.exists(path):
        os.makedirs(path)
